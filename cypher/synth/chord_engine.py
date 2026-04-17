"""ChordEngine — chord sequencer / controller.

Makes no sound itself. Generates note-on/note-off events from chord triggers,
respecting the current mode (CHORD / STRUM / ARP) and sends them to a target
voice (currently PolySynthVoice; will also handle sampler and MIDI-out).

Modes:
    CHORD      — all chord notes fire simultaneously (default)
    STRUM UP   — notes trigger low→high with a small delay between
    STRUM DOWN — high→low
    ARP UP     — cycle one note at a time, low→high, repeating until release
    ARP DOWN   — high→low
    ARP U/D    — up then down, back and forth
    ARP RAND   — random note from the chord each step

Parameters (8, 2 pages):
    KEY:  KEY | SCALE | PROGRESSION | STEP
    PLAY: MODE | RATE | OCTAVE | GATE

C++ portability notes:
    - Note scheduling is a fixed-size array of (frame, note, vel, on) tuples.
    - All state in plain ints/floats/lists. No closures, no generators.
    - Frame-accurate event fire times at block boundaries (block-rate quantized).
"""

from __future__ import annotations

import random

from ..core.parameter import Curve, Parameter
from ..core.types import DEFAULT_SAMPLE_RATE
from .chords import (
    PROGRESSIONS, PROGRESSION_LIST,
    SCALES, SCALE_LIST,
    build_progression_chord, progression_length,
)


# ── Trigger modes ────────────────────────────────────────────────────
MODE_CHORD = 0
MODE_STRUM_UP = 1
MODE_STRUM_DOWN = 2
MODE_ARP_UP = 3
MODE_ARP_DOWN = 4
MODE_ARP_UP_DOWN = 5
MODE_ARP_RANDOM = 6

MODE_NAMES = ["CHORD", "STRUM UP", "STRUM DN", "ARP UP", "ARP DN", "ARP U/D", "ARP RND"]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class ChordEngine:
    """Chord sequencer / controller. Emits note events to a target voice.

    The Player's audio callback calls advance(frames) each block, then routes
    returned events to self.target (set by Player after construction).
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

        # Target voice (set by Player). Must have trigger(note, vel) + release(note).
        self.target = None

        # Cached params
        self._key: int = 0           # 0=C … 11=B
        self._scale_idx: int = 0     # index into SCALE_LIST
        self._prog_idx: int = 0      # index into PROGRESSION_LIST
        self._step: int = 0          # current chord step
        self._mode: int = 0
        self._rate_ms: float = 30.0  # strum/arp rate
        self._octave: int = 3        # root octave (C3 = MIDI 60 == octave 3 * 12 + 24)
        self._gate: float = 0.7      # arp note duration as fraction of rate

        # Scheduled note events: list of (fire_frame, note, velocity, is_on)
        self._scheduled: list[tuple[int, int, float, bool]] = []
        self._frame_count: int = 0
        self._held_notes: set[int] = set()

        # Arp loop state (when an arp is "running")
        self._arp_running: bool = False
        self._arp_notes: list[int] = []
        self._arp_idx: int = 0
        self._arp_dir: int = 1       # for ARP_UP_DOWN: +1 or -1
        self._arp_next_frame: int = 0
        self._arp_last_note: int | None = None
        self._arp_velocity: float = 0.9

        self._params: list[Parameter] = [
            # Page 1: KEY
            Parameter("key",      "KEY",         0.0, 11.0, 0.0, "",   snap=12),
            Parameter("scale",    "SCALE",       0.0, len(SCALE_LIST) - 1, 0.0, "",
                      snap=len(SCALE_LIST)),
            Parameter("prog",     "PROGRESSION", 0.0, len(PROGRESSION_LIST) - 1, 0.0, "",
                      snap=len(PROGRESSION_LIST)),
            Parameter("step",     "STEP",        0.0, 7.0,  0.0, "",   snap=8),
            # Page 2: PLAY
            Parameter("mode",     "MODE",        0.0, len(MODE_NAMES) - 1, 0.0, "",
                      snap=len(MODE_NAMES)),
            Parameter("rate",     "RATE",        10.0, 500.0, 0.2, "ms", Curve.EXPONENTIAL),
            Parameter("octave",   "OCTAVE",      1.0, 6.0,  0.4, "",   snap=6),
            Parameter("gate",     "GATE",        0.1, 1.0,  0.7, "%"),
        ]

    # ──────────────────────────────────────────────────────────────────
    # Voice-like interface (for UI)
    # ──────────────────────────────────────────────────────────────────

    @property
    def params(self) -> list[Parameter]:
        return self._params

    @property
    def is_active(self) -> bool:
        return self._arp_running or bool(self._held_notes) or bool(self._scheduled)

    def trigger(self, note: int, velocity: float) -> None:
        # Pressing a chromatic note while on the ChordEngine tab fires a chord
        # rooted at that note, respecting the current progression step.
        self.trigger_chord_at_step(velocity, note_root=note)

    def release(self, note: int) -> None:
        # A chromatic release ends the chord
        self.release_all()

    def all_notes_off(self) -> None:
        self.release_all()
        self._scheduled.clear()

    # ──────────────────────────────────────────────────────────────────
    # Public chord control
    # ──────────────────────────────────────────────────────────────────

    @property
    def mode_name(self) -> str:
        self._update_params()
        return MODE_NAMES[self._mode]

    @property
    def scale_name(self) -> str:
        self._update_params()
        return SCALE_LIST[self._scale_idx]

    @property
    def progression_name(self) -> str:
        self._update_params()
        return PROGRESSION_LIST[self._prog_idx]

    @property
    def key_name(self) -> str:
        self._update_params()
        return NOTE_NAMES[self._key]

    @property
    def root_midi(self) -> int:
        """MIDI note of the current key root (from OCTAVE param)."""
        self._update_params()
        return self._octave * 12 + 24 + self._key

    def set_key(self, key_index: int) -> None:
        """Set key by 0-11 chromatic index."""
        key_index = max(0, min(11, key_index))
        self._params[0].value = key_index / 11.0

    def set_progression(self, prog_index: int) -> None:
        prog_index = max(0, min(len(PROGRESSION_LIST) - 1, prog_index))
        self._params[2].value = prog_index / max(1, len(PROGRESSION_LIST) - 1)
        self._params[3].value = 0.0  # reset step

    def step_progression(self, delta: int) -> None:
        self._update_params()
        plen = progression_length(self.progression_name)
        new_step = (self._step + delta) % plen
        self._params[3].value = new_step / 7.0

    def current_chord(self) -> tuple[list[int], str]:
        """Return (midi_notes, label) of the current progression chord."""
        self._update_params()
        plen = progression_length(self.progression_name)
        step = self._step % max(1, plen)
        return build_progression_chord(self.root_midi, self.progression_name, step)

    def trigger_chord_at_step(self, velocity: float = 0.9,
                              note_root: int | None = None) -> None:
        """Trigger the chord at the current progression step."""
        self._update_params()
        if note_root is not None:
            # Treat note_root as the key root for this trigger
            plen = progression_length(self.progression_name)
            step = self._step % max(1, plen)
            notes, _ = build_progression_chord(note_root, self.progression_name, step)
        else:
            notes, _ = self.current_chord()
        self._schedule_notes(notes, velocity)

    def release_all(self) -> None:
        """Release everything immediately."""
        for n in list(self._held_notes):
            if self.target is not None:
                self.target.release(n)
        self._held_notes.clear()
        self._arp_running = False
        self._arp_last_note = None

    # ──────────────────────────────────────────────────────────────────
    # Block-rate event dispatch (called from audio callback)
    # ──────────────────────────────────────────────────────────────────

    def advance(self, num_frames: int) -> None:
        """Advance the engine by num_frames and fire scheduled events.

        Events are dispatched to self.target (set by Player). Block-rate
        quantized — fine for strum/arp timings down to ~20ms.
        """
        if self.target is None:
            # No destination — drop any scheduled events so they don't stack up
            self._scheduled.clear()
            self._arp_running = False
            self._held_notes.clear()
            self._frame_count += num_frames
            return

        end_frame = self._frame_count + num_frames

        # Fire due scheduled events
        remaining = []
        for frame, note, vel, is_on in self._scheduled:
            if frame <= end_frame:
                if is_on:
                    self.target.trigger(note, vel)
                    self._held_notes.add(note)
                else:
                    self.target.release(note)
                    self._held_notes.discard(note)
            else:
                remaining.append((frame, note, vel, is_on))
        self._scheduled = remaining

        # Advance arp loop
        if self._arp_running and self._arp_notes:
            interval = max(1, int(self._rate_ms * self.sample_rate / 1000.0))
            gate_samples = max(1, int(interval * self._gate))
            while self._arp_next_frame <= end_frame:
                # Release previous arp note
                if self._arp_last_note is not None:
                    self.target.release(self._arp_last_note)
                    self._held_notes.discard(self._arp_last_note)

                # Pick next note based on direction
                mode = self._mode
                if mode == MODE_ARP_UP:
                    n = self._arp_notes[self._arp_idx]
                    self._arp_idx = (self._arp_idx + 1) % len(self._arp_notes)
                elif mode == MODE_ARP_DOWN:
                    n = self._arp_notes[self._arp_idx]
                    self._arp_idx = (self._arp_idx - 1) % len(self._arp_notes)
                elif mode == MODE_ARP_UP_DOWN:
                    n = self._arp_notes[self._arp_idx]
                    self._arp_idx += self._arp_dir
                    if self._arp_idx >= len(self._arp_notes) - 1:
                        self._arp_idx = len(self._arp_notes) - 1
                        self._arp_dir = -1
                    elif self._arp_idx <= 0:
                        self._arp_idx = 0
                        self._arp_dir = 1
                else:  # ARP_RANDOM
                    n = random.choice(self._arp_notes)

                self.target.trigger(n, self._arp_velocity)
                self._held_notes.add(n)
                self._arp_last_note = n

                # Schedule the note-off for gate time later
                self._scheduled.append(
                    (self._arp_next_frame + gate_samples, n, 0.0, False)
                )
                self._arp_next_frame += interval

        self._frame_count = end_frame

    # ──────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────

    def _update_params(self) -> None:
        self._key = max(0, min(11, int(round(self._params[0].mapped))))
        self._scale_idx = max(0, min(len(SCALE_LIST) - 1,
                                     int(round(self._params[1].mapped))))
        self._prog_idx = max(0, min(len(PROGRESSION_LIST) - 1,
                                    int(round(self._params[2].mapped))))
        self._step = max(0, min(7, int(round(self._params[3].mapped))))
        self._mode = max(0, min(len(MODE_NAMES) - 1,
                                int(round(self._params[4].mapped))))
        self._rate_ms = self._params[5].mapped
        self._octave = max(1, min(6, int(round(self._params[6].mapped))))
        self._gate = self._params[7].mapped

    def _schedule_notes(self, notes: list[int], velocity: float) -> None:
        """Schedule the given notes according to the current mode."""
        self.release_all()

        if not notes:
            return

        interval = max(1, int(self._rate_ms * self.sample_rate / 1000.0))

        if self._mode == MODE_CHORD:
            for n in notes:
                self._scheduled.append((self._frame_count, n, velocity, True))

        elif self._mode == MODE_STRUM_UP:
            ordered = sorted(notes)
            for i, n in enumerate(ordered):
                self._scheduled.append(
                    (self._frame_count + i * interval, n, velocity, True)
                )

        elif self._mode == MODE_STRUM_DOWN:
            ordered = sorted(notes, reverse=True)
            for i, n in enumerate(ordered):
                self._scheduled.append(
                    (self._frame_count + i * interval, n, velocity, True)
                )

        else:  # ARP modes
            self._arp_notes = sorted(notes)
            if self._mode == MODE_ARP_DOWN:
                self._arp_idx = len(self._arp_notes) - 1
            else:
                self._arp_idx = 0
            self._arp_dir = 1
            self._arp_running = True
            self._arp_next_frame = self._frame_count
            self._arp_velocity = velocity
            self._arp_last_note = None

    # ──────────────────────────────────────────────────────────────────
    # State (UI)
    # ──────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        self._update_params()
        plen = progression_length(self.progression_name)
        step = self._step % max(1, plen)
        _, label = build_progression_chord(self.root_midi, self.progression_name, step)
        return {
            "active": self.is_active,
            "key": self.key_name,
            "scale": self.scale_name,
            "progression": self.progression_name,
            "step": step,
            "progression_length": plen,
            "mode": self.mode_name,
            "rate_ms": self._rate_ms,
            "octave": self._octave,
            "gate": self._gate,
            "chord_label": label,
            "held_count": len(self._held_notes),
        }
