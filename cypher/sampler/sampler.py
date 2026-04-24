"""SamplerEngine — 16 pad slots, 8-voice polyphonic, phase-vocoder pitched.

Each slot has a MODE:
    PAD     — plays only on its own pad, no key tracking
    CLASSIC — one sample spreads across the pad grid. Trigger pad P while
              slot N has MODE=CLASSIC → play slot N pitched by (P - N)
              semitones, preserving tempo via phase vocoder.
    CHOP    — slot N's sample is divided into M slices (SLICES param).
              Pads N, N+1, …, N+M-1 play slices 0..M-1 (no pitch shift).

Dispatch priority for a pad trigger:
    1. If slot P is loaded in PAD mode → play slot P.
    2. Else walk slots N < P; if slot N is CLASSIC and covers P → pitched.
    3. Else if slot N is CHOP and (P - N) < slot.slices → that slice.

Pitch-shifted buffers are cached per (slot_idx, semitones_int) so repeated
triggers at the same pitch are instant.

C++ portability notes:
    - Cache is a flat array keyed by (slot * 25 + semi_offset + 12).
    - Slices are a list of (start_frame, end_frame) pairs per slot.
    - Per-voice playback uses raw float32 reads, no Python loops in hot path.
"""

from __future__ import annotations

import numpy as np

from ..core.envelope import ADEnvelope
from ..core.filters import BiquadFilter
from ..core.parameter import Curve, Parameter
from ..core.types import AudioBuffer, DEFAULT_SAMPLE_RATE
from ..core.voice import Voice
from .pitch_shift import pitch_shift
from .freeze import (
    FreezeState, FreezeProcessor,
    MOTION_HOLD, MOTION_DRIFT, MOTION_OSCILLATE, MOTION_DECAY, MOTION_NAMES,
)
from .gross_beat import GrossBeat, PRESET_NAMES as GROSS_PRESET_NAMES


PAD_COUNT = 16
VOICE_COUNT = 8
PAD_MIDI_START = 36   # C2
PAD_MIDI_END = 51     # D#3 (inclusive)

# Slot MODE values
MODE_PAD = 0
MODE_CLASSIC = 1
MODE_CHOP = 2
MODE_NAMES = ["PAD", "CLASSIC", "CHOP"]

# Beat divisions (slices per beat)
DIVISION_NAMES = ["1/4", "1/8", "1/16", "1/32", "1/4T", "1/8T", "1/16T"]
DIVISION_VALUES = [1, 2, 4, 8, 3, 6, 12]  # slices per beat

# Slot params: indices (kept in sync with _make_slot_params)
P_MODE = 0
P_PITCH = 1
P_REVERSE = 2
P_GAIN = 3
P_START = 4
P_END = 5
P_ATTACK = 6
P_DECAY = 7
P_SLICES = 8
P_FILTER = 9
# Freeze params
P_FZ_POS = 10
P_FZ_GRAIN = 11
P_FZ_MOTION = 12
P_FZ_RATE = 13


def _make_slot_params() -> list[Parameter]:
    """14 params per slot, 4 pages of 4 (+ 2 extras)."""
    return [
        # Page 1 — PLAY
        Parameter("mode",    "MODE",    0.0, 2.0,  0.0,  "",   snap=3),
        Parameter("pitch",   "PITCH",   -24.0, 24.0, 0.5, "st"),
        Parameter("reverse", "REVERSE", 0.0, 1.0,  0.0,  "",   snap=2),
        Parameter("gain",    "GAIN",    0.0, 2.0,  0.5,  "",   Curve.LINEAR),
        # Page 2 — TRIM
        Parameter("start",   "START",   0.0, 1.0,  0.0,  "%"),
        Parameter("end",     "END",     0.0, 1.0,  1.0,  "%"),
        Parameter("attack",  "ATTACK",  0.1, 2000.0, 0.0, "ms", Curve.EXPONENTIAL),
        Parameter("decay",   "DECAY",   10.0, 10000.0, 1.0, "ms", Curve.EXPONENTIAL),
        # Page 3 — CHOP
        Parameter("slices",  "SLICES",  1.0, 32.0, 0.1,  "",   snap=32),
        Parameter("filter",  "FILTER",  20.0, 20000.0, 1.0, "Hz", Curve.EXPONENTIAL),
        # Page 4 — FREEZE
        Parameter("fz_pos",    "FZ POS",    0.0,  1.0,  1.0, "%"),
        Parameter("fz_grain",  "FZ GRAIN",  20.0, 500.0, 0.4, "ms", Curve.EXPONENTIAL),
        Parameter("fz_motion", "FZ MOTION", 0.0,  3.0,  0.0, "",  snap=4),
        Parameter("fz_rate",   "FZ RATE",   0.0,  1.0,  0.5, ""),  # motion-dependent
    ]


class SampleSlot:
    """One pad slot: loaded sample + params + mode + slice points."""

    __slots__ = (
        "name", "path", "data", "source_rate", "_params", "loaded",
        "_pitch_cache", "_slice_count_cache", "_slice_points",
        "freeze_armed",
    )

    def __init__(self) -> None:
        self.name: str = ""
        self.path: str = ""
        self.data: AudioBuffer = np.zeros(1, dtype=np.float32)
        self.source_rate: int = DEFAULT_SAMPLE_RATE
        self._params: list[Parameter] = _make_slot_params()
        self.loaded: bool = False
        # Cache: semitone_int → pitch-shifted buffer (int between -24 and 24)
        self._pitch_cache: dict[int, AudioBuffer] = {}
        self._slice_count_cache: int = 0
        self._slice_points: list[tuple[int, int]] = []
        # Granular Freeze armed (per-slot flag, not a parameter)
        self.freeze_armed: bool = False

    @property
    def params(self) -> list[Parameter]:
        return self._params

    @property
    def mode(self) -> int:
        return max(0, min(2, int(round(self._params[P_MODE].mapped))))

    @property
    def slices(self) -> int:
        return max(1, min(32, int(round(self._params[P_SLICES].mapped))))

    @property
    def length(self) -> int:
        return len(self.data) if self.loaded else 0

    def load(self, name: str, path: str, data: AudioBuffer, source_rate: int) -> None:
        self.name = name
        self.path = path
        self.data = data
        self.source_rate = source_rate
        self.loaded = True
        self._pitch_cache.clear()
        self._slice_count_cache = 0
        self._slice_points = []

    def clear(self) -> None:
        self.name = ""
        self.path = ""
        self.data = np.zeros(1, dtype=np.float32)
        self.loaded = False
        self._pitch_cache.clear()
        self._slice_points = []

    def get_pitched_buffer(self, semitones: int) -> AudioBuffer:
        """Return the sample pitch-shifted by `semitones` (integer semitones).

        Cached — first call for a given semitone may take a few ms; subsequent
        calls are instant.
        """
        if semitones == 0 or not self.loaded:
            return self.data
        if semitones in self._pitch_cache:
            return self._pitch_cache[semitones]
        shifted = pitch_shift(self.data, float(semitones))
        self._pitch_cache[semitones] = shifted
        return shifted

    def refresh_slices(self) -> None:
        """Recompute slice boundaries when SLICES param changes."""
        n = self.slices
        if n == self._slice_count_cache or not self.loaded:
            return
        length = self.length
        slice_w = length // max(1, n)
        self._slice_points = [
            (i * slice_w, (i + 1) * slice_w if i < n - 1 else length)
            for i in range(n)
        ]
        self._slice_count_cache = n


class SamplerVoice:
    """One polyphonic voice. Plays a static buffer chosen at trigger time."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.buffer: AudioBuffer = np.zeros(0, dtype=np.float32)
        self.slot: SampleSlot | None = None
        self._position: float = 0.0
        self._rate: float = 1.0
        self._active: bool = False
        self._note: int = -1
        self._velocity: float = 0.0
        self._reversed: bool = False
        self._start_pos: float = 0.0
        self._end_pos: float = 0.0
        self._gain: float = 1.0

        self._amp_env = ADEnvelope(sample_rate)
        self._amp_env.sustain_level = 1.0
        self._filter = BiquadFilter(sample_rate)
        self._cutoff_hz: float = 20000.0

        # Granular Freeze state. Non-None when voice is in freeze mode.
        self._freeze: FreezeProcessor | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    def trigger(
        self,
        slot: SampleSlot,
        note: int,
        velocity: float,
        *,
        buffer: AudioBuffer | None = None,
        slice_start: int = 0,
        slice_end: int = -1,
    ) -> None:
        """Trigger playback. `buffer` can override slot.data (for pitch-shifted
        buffers in CLASSIC mode). `slice_start`/`slice_end` can scope to a slice."""
        if not slot.loaded or slot.length < 2:
            return

        self.slot = slot
        self._note = note
        self._velocity = max(0.0, min(1.0, velocity))

        # Pick the source buffer
        self.buffer = buffer if buffer is not None else slot.data
        total = len(self.buffer)

        # Determine play range
        p = slot.params
        if slice_end < 0:  # whole sample
            start_f = max(0.0, min(1.0, p[P_START].mapped))
            end_f = max(0.0, min(1.0, p[P_END].mapped))
            if end_f <= start_f:
                end_f = min(1.0, start_f + 0.001)
            self._start_pos = start_f * total
            self._end_pos = end_f * total
        else:
            self._start_pos = float(slice_start)
            self._end_pos = float(min(total, slice_end))

        self._reversed = int(round(p[P_REVERSE].mapped)) == 1
        self._position = self._end_pos - 1 if self._reversed else self._start_pos
        # No resampling in the voice — pitch shift done offline via phase vocoder.
        self._rate = slot.source_rate / self.sample_rate

        self._gain = p[P_GAIN].mapped * self._velocity
        self._cutoff_hz = p[P_FILTER].mapped

        self._amp_env.attack_time = p[P_ATTACK].mapped / 1000.0
        self._amp_env.decay_time = p[P_DECAY].mapped / 1000.0
        self._amp_env.sustain_level = 1.0
        self._amp_env.release_time = 0.02
        self._amp_env.trigger()

        if self._cutoff_hz < 18000.0:
            self._filter.reset()
            self._filter.set_lowpass(self._cutoff_hz, 0.707)

        self._active = True

    def release(self, note: int) -> None:
        if note == self._note:
            self._amp_env.release()
            # Gate mode: freeze ends with note-off.
            if self._freeze is not None:
                self._freeze.state.active = False

    def stop(self) -> None:
        self._active = False
        self._amp_env._stage = "idle"
        self._amp_env._level = 0.0
        self._note = -1
        self._freeze = None

    def _engage_freeze(self) -> None:
        """Switch to freeze mode at the end of normal playback."""
        if self.slot is None or not self.slot.loaded:
            return
        p = self.slot.params
        pos_frac = max(0.0, min(1.0, p[P_FZ_POS].mapped))
        grain_ms = p[P_FZ_GRAIN].mapped
        grain = max(64, int(grain_ms * self.sample_rate / 1000.0))
        motion = max(0, min(3, int(round(p[P_FZ_MOTION].mapped))))
        rate_norm = p[P_FZ_RATE].mapped  # 0..1, motion-dependent meaning

        # Interpret RATE per MOTION
        if motion == MOTION_DRIFT:
            # 0..1 → 0..1.0 fraction-of-sample / second (tunable)
            rate = rate_norm * 0.5
            depth = 0.0
        elif motion == MOTION_OSCILLATE:
            # 0..1 → 0.1..6Hz (exponential)
            rate = 0.1 * (6.0 / 0.1) ** rate_norm
            depth = 0.15  # fixed-ish DEPTH for v1
        elif motion == MOTION_DECAY:
            # 0..1 → 0.1..30s fade time (exponential)
            rate = 0.1 * (30.0 / 0.1) ** rate_norm
            depth = 0.0
        else:  # HOLD
            rate = 0.0
            depth = 0.0

        state = FreezeState(
            source=self.buffer,
            position_frac=pos_frac,
            grain_size=grain,
            motion=motion,
            rate=rate,
            depth=depth,
            sample_rate=self.sample_rate,
        )
        self._freeze = FreezeProcessor(state, self.sample_rate)
        # Keep amp envelope sustaining while freeze runs
        if self._amp_env._stage == "release":
            self._amp_env._stage = "sustain"
            self._amp_env._level = max(self._amp_env._level, 0.3)

    def process(self, num_frames: int) -> AudioBuffer:
        if not self._active:
            return np.zeros(num_frames, dtype=np.float32)

        # Freeze takeover: once engaged, all output comes from the freeze
        # processor. No sample playback logic runs.
        if self._freeze is not None and self._freeze.state.active:
            out = self._freeze.process(num_frames)
            if self._cutoff_hz < 18000.0:
                out = self._filter.process(out)
            env = self._amp_env.process(num_frames)
            out *= env * self._gain
            if not self._amp_env.is_active and not self._freeze.state.active:
                self._active = False
            return out

        buf = self.buffer
        length = len(buf)
        out = np.zeros(num_frames, dtype=np.float32)
        rate = self._rate
        pos = self._position
        reversed_play = self._reversed
        start = self._start_pos
        end = self._end_pos
        hit_end = False

        for i in range(num_frames):
            if reversed_play:
                if pos <= start:
                    hit_end = True
                    break
            else:
                if pos >= end - 1:
                    hit_end = True
                    break
            idx = int(pos)
            frac = pos - idx
            if 0 <= idx < length - 1:
                a = buf[idx]
                b = buf[idx + 1]
                out[i] = a + frac * (b - a)
            pos = pos - rate if reversed_play else pos + rate

        self._position = pos

        if hit_end:
            # End of sample/slice — engage freeze if armed, else start release
            if self.slot is not None and self.slot.freeze_armed:
                self._engage_freeze()
            elif self._amp_env.is_active:
                self._amp_env.release()

        if self._cutoff_hz < 18000.0:
            out = self._filter.process(out)

        env = self._amp_env.process(num_frames)
        out *= env * self._gain

        if not self._amp_env.is_active and self._freeze is None:
            self._active = False

        return out


class SamplerEngine(Voice):
    """16-slot / 8-voice polyphonic sampler with MODE dispatch."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, project=None) -> None:
        super().__init__(sample_rate)
        self.project = project
        self._slots: list[SampleSlot] = [SampleSlot() for _ in range(PAD_COUNT)]
        self._voices: list[SamplerVoice] = [
            SamplerVoice(sample_rate) for _ in range(VOICE_COUNT)
        ]
        self._alloc_order: list[SamplerVoice] = []
        self._note_map: dict[int, SamplerVoice] = {}
        self._focused_slot: int = 0

        # Sampler-global Gross Beat volume FX (one preset applies to all pads)
        self.gross_beat = GrossBeat(sample_rate)

    # ── Voice interface ─────────────────────────────────────────────

    @property
    def params(self) -> list[Parameter]:
        return self._slots[self._focused_slot].params

    @property
    def is_active(self) -> bool:
        return any(v.is_active for v in self._voices)

    def trigger(self, note: int, velocity: float) -> None:
        pad_idx = self._note_to_pad(note)
        if pad_idx < 0:
            return
        self.trigger_pad(pad_idx, note, velocity)

    def release(self, note: int) -> None:
        voice = self._note_map.get(note)
        if voice is not None:
            voice.release(note)
            del self._note_map[note]

    def all_notes_off(self) -> None:
        for v in self._voices:
            v.stop()
        self._note_map.clear()
        self._alloc_order.clear()

    def process(self, num_frames: int) -> AudioBuffer:
        # Sync Gross Beat tempo to project BPM if available
        if self.project is not None:
            self.gross_beat.bpm = self.project.bpm

        buf = np.zeros(num_frames, dtype=np.float32)
        active = 0
        for v in self._voices:
            if v.is_active:
                buf += v.process(num_frames)
                active += 1
        if active > 1:
            buf /= np.sqrt(active)

        # Gross Beat volume preset (no-op when preset == OFF)
        buf = self.gross_beat.process(buf)
        return buf

    def get_state(self) -> dict:
        slot = self._slots[self._focused_slot]
        return {
            "active": self.is_active,
            "focused_slot": self._focused_slot,
            "focused_name": slot.name if slot.loaded else "",
            "focused_mode": MODE_NAMES[slot.mode] if slot.loaded else "-",
            "active_voices": sum(1 for v in self._voices if v.is_active),
            "max_voices": VOICE_COUNT,
            "slots_loaded": sum(1 for s in self._slots if s.loaded),
        }

    # ── Sampler-specific ────────────────────────────────────────────

    @property
    def slots(self) -> list[SampleSlot]:
        return self._slots

    @property
    def focused_slot_idx(self) -> int:
        return self._focused_slot

    def focus_slot(self, idx: int) -> None:
        self._focused_slot = max(0, min(PAD_COUNT - 1, idx))

    def load_into_slot(
        self, slot_idx: int, name: str, path: str,
        data: AudioBuffer, source_rate: int,
    ) -> None:
        self._slots[slot_idx].load(name, path, data, source_rate)

    def clear_slot(self, slot_idx: int) -> None:
        self._slots[slot_idx].clear()

    def trigger_pad(self, pad_idx: int, note: int, velocity: float) -> None:
        """Resolve what slot + playback mode applies for this pad and trigger."""
        resolved = self._resolve_pad(pad_idx)
        if resolved is None:
            return
        slot, mode, slice_idx, semi_offset = resolved

        # Apply user-tweak PITCH from the slot's params, on top of the
        # key-tracking offset that CLASSIC mode contributes.
        manual_pitch = slot.params[P_PITCH].mapped
        total_semi = int(round(semi_offset + manual_pitch))

        existing = self._note_map.get(note)
        voice = existing if existing is not None else self._allocate_voice()

        if mode == MODE_CLASSIC and total_semi != 0:
            buf = slot.get_pitched_buffer(total_semi)
            voice.trigger(slot, note, velocity, buffer=buf)
        elif mode == MODE_CHOP:
            slot.refresh_slices()
            if slice_idx < 0 or slice_idx >= len(slot._slice_points):
                return
            s_start, s_end = slot._slice_points[slice_idx]
            voice.trigger(slot, note, velocity, slice_start=s_start, slice_end=s_end)
        else:
            # PAD mode: optionally apply user PITCH as phase-vocoder shift
            if total_semi != 0:
                buf = slot.get_pitched_buffer(total_semi)
                voice.trigger(slot, note, velocity, buffer=buf)
            else:
                voice.trigger(slot, note, velocity)

        self._note_map[note] = voice
        self._touch_alloc(voice)

    def _resolve_pad(
        self, pad_idx: int
    ) -> tuple[SampleSlot, int, int, int] | None:
        """Return (slot, mode, slice_idx, semi_offset) for a pad trigger.

        Dispatch order:
            1) PAD-mode slot loaded at pad_idx
            2) CLASSIC or CHOP slot at a lower pad_idx whose range covers this pad
        """
        # 1. Direct slot
        direct = self._slots[pad_idx]
        if direct.loaded and direct.mode == MODE_PAD:
            return direct, MODE_PAD, -1, 0

        # 2. Scan backward for CLASSIC or CHOP coverage
        for home in range(pad_idx, -1, -1):
            s = self._slots[home]
            if not s.loaded:
                continue
            offset = pad_idx - home
            if s.mode == MODE_CLASSIC:
                # Covers the whole pad grid from home upward
                if offset >= 0:
                    return s, MODE_CLASSIC, -1, offset
            elif s.mode == MODE_CHOP:
                if 0 <= offset < s.slices:
                    return s, MODE_CHOP, offset, 0
            elif s.mode == MODE_PAD and home != pad_idx:
                # A PAD slot below us doesn't fall through — stop looking
                continue

        # Fallback: if the direct slot is loaded in CLASSIC/CHOP but we're
        # asking for the home pad, play at zero offset
        if direct.loaded:
            if direct.mode == MODE_CLASSIC:
                return direct, MODE_CLASSIC, -1, 0
            if direct.mode == MODE_CHOP:
                direct.refresh_slices()
                return direct, MODE_CHOP, 0, 0

        return None

    def find_free_slot(self) -> int:
        for i, s in enumerate(self._slots):
            if not s.loaded:
                return i
        return -1

    @staticmethod
    def _note_to_pad(note: int) -> int:
        if PAD_MIDI_START <= note <= PAD_MIDI_END:
            return note - PAD_MIDI_START
        return -1

    # ── Voice allocation ────────────────────────────────────────────

    def _allocate_voice(self) -> SamplerVoice:
        for v in self._voices:
            if not v.is_active:
                return v
        if self._alloc_order:
            stolen = self._alloc_order.pop(0)
            for note, voice in list(self._note_map.items()):
                if voice is stolen:
                    del self._note_map[note]
                    break
            return stolen
        return self._voices[0]

    def _touch_alloc(self, v: SamplerVoice) -> None:
        if v in self._alloc_order:
            self._alloc_order.remove(v)
        self._alloc_order.append(v)
