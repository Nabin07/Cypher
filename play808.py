#!/usr/bin/env python3
"""Interactive CYPHER tweaker — pygame UI with proper key hold/release.

Controls:
  TAB          Switch voice (808 / KICK / SYNTH / FX / CHORD)
  SPACE        Trigger root note (hold to sustain)
  Z/X          Octave down/up
  A-K keys     Chromatic notes (hold to sustain, polyphonic on synth)
  UP/DOWN      Select parameter
  LEFT/RIGHT   Adjust parameter
  [/]          Fine-tune parameter
  1/2/3/4      Switch parameter page
  P            Pair kick with 808 / Unpair
  M            Toggle trigger mode (Classic / One-shot)
  R            Reset all params to defaults
  C            Trigger chord (hold to sustain)
  ,/.          Step through chord progression
  N            Cycle progression
  V            Toggle reverb (global)
  Q / ESC      Quit
"""

import argparse
import math
import sys
import threading
import time

import os

import numpy as np
os.environ['SDL_VIDEO_ALLOW_HIGHDPI'] = '1'  # must be set before pygame.init()
import pygame
import sounddevice as sd

sys.path.insert(0, ".")
from cypher.drum.sub808 import Sub808Voice
from cypher.drum.kick import KickVoice
from cypher.synth.poly import PolySynthVoice
from cypher.synth.mono import WAVE_NAMES
from cypher.synth.chords import (
    build_progression_chord, progression_length,
    PROGRESSION_LIST, CHORD_TYPES, NOTE_NAMES as CHORD_NOTE_NAMES,
    build_chord,
)
from cypher.core.types import DEFAULT_SAMPLE_RATE
from cypher.core.reverb import DattorroPlateReverb
from cypher.core.parameter import Parameter, Curve
from cypher.core.project import Project
from cypher.sampler.sampler import (
    SamplerEngine, PAD_MIDI_START, PAD_MIDI_END,
)
from cypher.sampler.loader import (
    pick_folder_dialog, scan_folder, load_sample, load_sample_with_meta,
)
from cypher.sampler.sidecar import write_sidecar
from cypher.midi import NoteOn, NoteOff, ControlChange, MidiMessage
from cypher import midi_input

SR = DEFAULT_SAMPLE_RATE
BLOCK_SIZE = 1024

# --- Colors ---
BG = (18, 18, 36)
PANEL = (28, 28, 50)
TEXT = (180, 180, 200)
DIM = (80, 80, 100)
CYAN = (0, 210, 255)
YELLOW = (255, 215, 0)
GREEN = (0, 255, 130)
RED = (255, 68, 68)
MAGENTA = (255, 100, 255)
ORANGE = (255, 165, 0)
SLIDER_BG = (40, 40, 65)
KEY_WHITE = (220, 220, 230)
KEY_BLACK = (55, 50, 75)
KEY_BLACK_OUT = (20, 20, 25)
KEY_ACTIVE = (0, 255, 130)
KEY_GREY = (40, 40, 50)
KEY_DOT = (255, 100, 255)

# --- Window ---
WIN_W, WIN_H = 800, 830
METRO_STRIP_H = 50
WAVE_H = 100          # waveform display height
WAVE_W = 230          # waveform display width
PREVIEW_SAMPLES = 4096
FPS = 60

# MIDI keyboard dimensions
MIDI_KB_H = 52
MIDI_KB_KEYS = 25     # 2 octaves + 1
MIDI_KB_WHITE_W = 28
MIDI_KB_BLACK_W = 18
MIDI_KB_BLACK_H = 32

# Chromatic keyboard: A=C, W=C#, S=D, E=D#, D=E, F=F, T=F#, G=G, Y=G#, H=A, U=A#, J=B, K=C+1
CHROMATIC_KEYS = {
    pygame.K_a: 0,  pygame.K_w: 1,  pygame.K_s: 2,  pygame.K_e: 3,
    pygame.K_d: 4,  pygame.K_f: 5,  pygame.K_t: 6,  pygame.K_g: 7,
    pygame.K_y: 8,  pygame.K_h: 9,  pygame.K_u: 10, pygame.K_j: 11,
    pygame.K_k: 12,
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# --- Scale definitions (semitones from root) ---
SCALES = {
    "MAJOR":     [0, 2, 4, 5, 7, 9, 11],
    "MINOR":     [0, 2, 3, 5, 7, 8, 10],
    "DORIAN":    [0, 2, 3, 5, 7, 9, 10],
    "MIXOLYD":   [0, 2, 4, 5, 7, 9, 10],
    "PHRYGIAN":  [0, 1, 3, 5, 7, 8, 10],
    "PENTATONIC":[0, 2, 4, 7, 9],
    "BLUES":     [0, 3, 5, 6, 7, 10],
    "CHROMATIC": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}
SCALE_NAMES = list(SCALES.keys())

VOICE_PAGES = {
    "808": [
        {"name": "SIMPLE", "params": [0, 1, 2, 3]},
        {"name": "ADVANCED", "params": [4, 5, 6, 7]},
        {"name": "ADVANCED 2", "params": [8, 9, 10, 11]},
    ],
    "KICK": [
        {"name": "SIMPLE", "params": [0, 1, 2, 3]},
        {"name": "ADVANCED", "params": [4, 5, 6, 7]},
    ],
    "SYNTH": [
        {"name": "TONE", "params": [0, 1, 2, 3]},
        {"name": "FILTER", "params": [4, 5, 6, 7]},
        {"name": "AMP", "params": [8, 9, 10, 11]},
        {"name": "MOD", "params": [12, 13, 14, 15]},
    ],
    "FX": [],        # FX uses custom slot-based UI
    "CHORD": [],     # CHORD uses custom UI
    "SAMPLER": [],   # SAMPLER uses custom UI
}

VOICE_NAMES = ["808", "KICK", "SYNTH", "FX", "CHORD", "SAMPLER"]
SAMPLER_IDX = 5

# Tab index → voice-list index (None for FX/CHORD which aren't sound voices)
TAB_TO_VOICE = {0: 0, 1: 1, 2: 2, 3: None, 4: None, 5: 3}


def tab_voice_idx(tab_idx):
    return TAB_TO_VOICE.get(tab_idx)


# MPC-style 4x4 pad keyboard layout for the SAMPLER tab
SAMPLER_PAD_KEYS = {
    pygame.K_z: 0,  pygame.K_x: 1,  pygame.K_c: 2,  pygame.K_v: 3,
    pygame.K_a: 4,  pygame.K_s: 5,  pygame.K_d: 6,  pygame.K_f: 7,
    pygame.K_q: 8,  pygame.K_w: 9,  pygame.K_e: 10, pygame.K_r: 11,
    pygame.K_1: 12, pygame.K_2: 13, pygame.K_3: 14, pygame.K_4: 15,
}

# FX reverb parameters
FX_REVERB_PARAMS = [
    Parameter(name="predelay", label="PREDELAY", min_val=0.0, max_val=150.0,
             default=0.13, unit="ms", curve=Curve.LINEAR),
    Parameter(name="decay", label="DECAY", min_val=0.1, max_val=70.0,
             default=0.35, unit="s", curve=Curve.EXPONENTIAL),
    Parameter(name="mix", label="MIX", min_val=0.0, max_val=1.0,
             default=0.30, unit="%", curve=Curve.LINEAR),
    Parameter(name="high_cut", label="HIGH CUT", min_val=1000.0, max_val=20000.0,
             default=0.55, unit="Hz", curve=Curve.EXPONENTIAL),
    Parameter(name="low_cut", label="LOW CUT", min_val=20.0, max_val=500.0,
             default=0.25, unit="Hz", curve=Curve.EXPONENTIAL),
]

# ── MIDI pad mapping (KeyLab 61 Essential defaults) ──────────────────
PAD_CHANNEL = 9
PAD_ENGINE_MAP = {
    36: 0,   # Pad 1 -> 808
    37: 1,   # Pad 2 -> KICK
    38: 2,   # Pad 3 -> SYNTH
}
KNOB_CCS = [10, 74, 71, 76, 77, 93, 73, 75, 18]

# Strum/arp rhythm divisions (in beats)
RHYTHM_DIVS = {
    "1/4": 1.0,
    "1/8": 0.5,
    "1/16": 0.25,
    "1/32": 0.125,
}
RHYTHM_NAMES = list(RHYTHM_DIVS.keys())


# ── Audio engine ──────────────────────────────────────────────────────

class Player:
    def __init__(self):
        self.project = Project(key=0, scale_idx=1, bpm=120.0)

        self.sub808 = Sub808Voice(SR)
        self.kick = KickVoice(SR)
        self.synth = PolySynthVoice(SR)
        self.sampler = SamplerEngine(SR, project=self.project)
        from cypher.core.metronome import Metronome
        self.metronome = Metronome(self.project, SR)
        self.voices = [self.sub808, self.kick, self.synth, self.sampler]
        self.voice_idx = 0
        self.lock = threading.Lock()
        self.octave = 2
        self.current_page = 0
        self.selected_param = 0
        self.peak_level = 0.0
        self.stream = None

        # Chord state
        self.chord_prog_idx = 0
        self.chord_step = 0
        self.chord_label = ""

        # MIDI note tracking
        self._midi_note_voice: dict[int, int] = {}

        # Global reverb
        self.reverb = DattorroPlateReverb(SR)
        self.reverb_on = False

        # FX state
        self.fx_reverb_params = [Parameter(**{
            'name': p.name, 'label': p.label, 'min_val': p.min_val,
            'max_val': p.max_val, 'default': p.default,
            'unit': p.unit, 'curve': p.curve, 'snap': p.snap,
        }) for p in FX_REVERB_PARAMS]
        # Indices align with self.voices: [sub808, kick, synth, sampler]
        self.fx_sends = [False, False, True, False]
        self.fx_send_amounts = [0.5, 0.5, 0.5, 0.5]
        self.fx_reverb_expanded = True
        self.fx_selected_param = 0
        self.fx_reverb_mode = 2  # HALL default
        self._sync_reverb_from_params()

        # Global key/scale
        self.global_key = 0      # 0=C, 1=C#, ... 11=B
        self.global_scale_idx = 1  # default MINOR

        # Chord engine state
        self.chord_mode = 0      # 0=CHORD (normal), 1=STRUM DOWN, 2=STRUM UP, 3=ARPEGGIO
        self.chord_rhythm_idx = 2  # default 1/16
        self.chord_swing = 0.0   # 0.0-1.0
        self.chord_bpm = 120.0
        self._strum_thread = None
        self._strum_stop = threading.Event()
        self._arp_thread = None
        self._arp_stop = threading.Event()
        self._arp_notes: list[int] = []
        self._arp_running = False

        # Active notes for MIDI keyboard display
        self.active_midi_notes: set[int] = set()

        # Waveform preview
        self.preview_buf = np.zeros(PREVIEW_SAMPLES, dtype=np.float32)
        self._preview_param_hash: int = 0
        self._preview_voice_idx: int = -1

        # Sampler UI state
        self.sampler_folder: str = ""
        self.sampler_files: list = []
        self.sampler_selected_file: int = 0
        # BPM-edit debounce: slot_idx -> monotonic timestamp of last edit
        self.sampler_bpm_dirty: dict[int, float] = {}
        # Tap-tempo timestamps per slot (last few)
        self.sampler_tap_history: dict[int, list[float]] = {}

    def _sync_reverb_from_params(self):
        """Push FX param values into the reverb DSP."""
        p = self.fx_reverb_params
        self.reverb.predelay_ms = p[0].mapped
        decay_time = max(0.1, p[1].mapped)
        tank_period = 0.15
        self.reverb.decay = min(0.999, 10.0 ** (-3.0 * tank_period / decay_time))
        self.reverb.mix = p[2].mapped
        freq = p[3].mapped
        self.reverb.damping = math.exp(-2.0 * math.pi * freq / SR)
        self.reverb.low_cut_hz = p[4].mapped
        self.reverb.set_mode(self.fx_reverb_mode)

    def get_scale_notes(self) -> set[int]:
        """Get all MIDI notes in the current global key/scale."""
        scale = SCALES[SCALE_NAMES[self.global_scale_idx]]
        notes = set()
        for octave in range(-1, 9):
            for interval in scale:
                n = self.global_key + interval + (octave + 1) * 12
                if 0 <= n <= 127:
                    notes.add(n)
        return notes

    def note_in_scale(self, midi_note: int) -> bool:
        """Check if a MIDI note is in the current global scale."""
        pc = (midi_note - self.global_key) % 12
        scale = SCALES[SCALE_NAMES[self.global_scale_idx]]
        return pc in scale

    @property
    def voice(self):
        # Tabs: 0=808 1=KICK 2=SYNTH 3=FX 4=CHORD 5=SAMPLER
        if self.voice_idx == SAMPLER_IDX:
            return self.sampler
        if self.voice_idx in (3, 4):
            return self.synth  # fallback for FX/CHORD (no own voice)
        return self.voices[self.voice_idx]

    # ── Sampler helpers ──
    def sampler_open_folder(self, folder=None) -> bool:
        if folder is None:
            folder = pick_folder_dialog(initial=self.sampler_folder or None)
        if not folder:
            return False
        self.sampler_folder = folder
        self.sampler_files = scan_folder(folder)
        self.sampler_selected_file = 0
        return True

    def sampler_load_selected_to(self, slot_idx: int) -> bool:
        if not self.sampler_files:
            return False
        idx = max(0, min(len(self.sampler_files) - 1, self.sampler_selected_file))
        path = self.sampler_files[idx]
        try:
            data, sr, meta = load_sample_with_meta(path)
        except Exception:
            return False
        with self.lock:
            slot = self.sampler.slots[slot_idx]
            slot.load(
                path.name, str(path), data, sr,
                sample_bpm=float(meta.get("bpm", 0.0)),
                bpm_confidence=float(meta.get("confidence", 0.0)),
                user_corrected=bool(meta.get("user_corrected", False)),
            )
            # Auto-pre-select Match when confidence high + ratio reasonable
            proj_bpm = self.project.bpm
            if (slot.sample_bpm > 0 and proj_bpm > 0
                    and slot.bpm_confidence >= 0.6):
                ratio = slot.sample_bpm / proj_bpm
                if 0.85 <= ratio <= 1.15:
                    slot.match_mode = True
        return True

    def sampler_persist_slot_meta(self, slot_idx: int) -> None:
        """Write current slot BPM/user_corrected back to sidecar JSON."""
        slot = self.sampler.slots[slot_idx]
        if not slot.loaded or not slot.path:
            return
        write_sidecar(slot.path, {
            "bpm": float(slot.sample_bpm),
            "confidence": float(slot.bpm_confidence),
            "user_corrected": bool(slot.user_corrected),
        })

    @property
    def voice_name(self):
        return VOICE_NAMES[self.voice_idx]

    def audio_callback(self, outdata, frames, time_info, status):
        with self.lock:
            buf = np.zeros(frames, dtype=np.float32)
            fx_bus = np.zeros(frames, dtype=np.float32)
            for i, v in enumerate(self.voices):
                if v.is_active:
                    voice_out = v.process(frames)
                    if self.reverb_on and i < len(self.fx_sends) and self.fx_sends[i]:
                        send_amt = self.fx_send_amounts[i]
                        fx_bus += voice_out * send_amt
                        buf += voice_out * (1.0 - send_amt)
                    else:
                        buf += voice_out
            if self.reverb_on:
                buf += self.reverb.process(fx_bus)
            # Metronome click: dry, no reverb send
            buf += self.metronome.process(frames)
            peak = np.max(np.abs(buf))
            if peak > 1.0:
                buf /= peak
        p = float(np.max(np.abs(buf)))
        self.peak_level = max(p, self.peak_level * 0.95)
        outdata[:, 0] = buf

    def start(self):
        self.stream = sd.OutputStream(
            samplerate=SR, blocksize=BLOCK_SIZE, channels=1,
            dtype='float32', callback=self.audio_callback, latency='low',
        )
        self.stream.start()

    def stop(self):
        self._strum_stop.set()
        self._arp_stop.set()
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def switch_voice(self):
        self.voice_idx = (self.voice_idx + 1) % len(VOICE_NAMES)
        self.current_page = 0
        self.selected_param = 0

    def select_engine(self, idx):
        if idx == self.voice_idx or idx < 0 or idx >= len(VOICE_NAMES):
            return
        self.voice_idx = idx
        self.current_page = 0
        self.selected_param = 0

    def trigger(self, note, velocity=0.9):
        with self.lock:
            if self.sub808.is_active:
                self.kick.set_linked_808_freq(self.sub808._current_pitch_hz)
            else:
                self.kick.set_linked_808_freq(0.0)
            self.voice.trigger(note, velocity)
        self.active_midi_notes.add(note)

    def release_note(self, note):
        with self.lock:
            self.voice.release(note)
        self.active_midi_notes.discard(note)

    def trigger_chord(self, notes, velocity=0.9):
        with self.lock:
            self.synth.release_all()
            for note in notes:
                self.synth.trigger(note, velocity)
        self.active_midi_notes.update(notes)

    def release_chord(self, notes):
        with self.lock:
            for note in notes:
                self.synth.release(note)
        for n in notes:
            self.active_midi_notes.discard(n)

    def trigger_chord_humanized(self, notes, velocity=0.9):
        """Trigger all notes near-simultaneously with tiny random offsets (5-20ms)."""
        import random
        self._strum_stop.set()
        if self._strum_thread and self._strum_thread.is_alive():
            self._strum_thread.join(timeout=0.5)
        self._strum_stop = threading.Event()

        def _humanize():
            self.active_midi_notes.clear()
            for i, note in enumerate(notes):
                if self._strum_stop.is_set():
                    return
                with self.lock:
                    self.synth.trigger(note, velocity * random.uniform(0.85, 1.0))
                self.active_midi_notes.add(note)
                if i < len(notes) - 1:
                    self._strum_stop.wait(random.uniform(0.005, 0.020))

        self._strum_thread = threading.Thread(target=_humanize, daemon=True)
        self._strum_thread.start()

    def trigger_strum(self, notes, velocity=0.9, direction="down"):
        """Trigger notes with strum delay. Direction: 'down' low-to-high, 'up' high-to-high."""
        self._strum_stop.set()
        if self._strum_thread and self._strum_thread.is_alive():
            self._strum_thread.join(timeout=0.5)
        self._strum_stop = threading.Event()

        ordered = sorted(notes) if direction == "down" else sorted(notes, reverse=True)
        div = RHYTHM_DIVS[RHYTHM_NAMES[self.chord_rhythm_idx]]
        delay_per_note = (60.0 / self.chord_bpm) * div / max(1, len(ordered) - 1) if len(ordered) > 1 else 0

        def _strum():
            self.active_midi_notes.clear()
            for i, note in enumerate(ordered):
                if self._strum_stop.is_set():
                    return
                with self.lock:
                    self.synth.trigger(note, velocity)
                self.active_midi_notes.add(note)
                if i < len(ordered) - 1 and delay_per_note > 0:
                    # Apply swing to even-numbered notes
                    actual_delay = delay_per_note
                    if i % 2 == 1 and self.chord_swing > 0:
                        actual_delay *= (1.0 + self.chord_swing * 0.5)
                    self._strum_stop.wait(actual_delay)

        self._strum_thread = threading.Thread(target=_strum, daemon=True)
        self._strum_thread.start()

    def start_arpeggio(self, notes, velocity=0.9):
        """Start arpeggio cycling through notes at tempo."""
        self.stop_arpeggio()
        self._arp_stop = threading.Event()
        self._arp_notes = sorted(notes)
        self._arp_running = True

        div = RHYTHM_DIVS[RHYTHM_NAMES[self.chord_rhythm_idx]]
        note_duration = (60.0 / self.chord_bpm) * div

        def _arp():
            idx = 0
            last_note = -1
            while not self._arp_stop.is_set():
                note = self._arp_notes[idx % len(self._arp_notes)]
                # Don't hard-release previous note — just trigger the new one.
                # The poly pool handles voice reuse (same note = same voice,
                # soft retrigger via envelope). No gap = no pop.
                with self.lock:
                    self.synth.trigger(note, velocity)
                if last_note >= 0 and last_note != note:
                    self.active_midi_notes.discard(last_note)
                self.active_midi_notes.add(note)
                last_note = note
                actual_dur = note_duration
                if idx % 2 == 1 and self.chord_swing > 0:
                    actual_dur *= (1.0 + self.chord_swing * 0.5)
                self._arp_stop.wait(actual_dur)
                idx += 1
            self._arp_running = False

        self._arp_thread = threading.Thread(target=_arp, daemon=True)
        self._arp_thread.start()

    def stop_strum(self):
        """Stop any running strum thread."""
        self._strum_stop.set()
        if self._strum_thread and self._strum_thread.is_alive():
            self._strum_thread.join(timeout=0.5)

    def stop_arpeggio(self):
        if self._arp_running:
            self._arp_stop.set()
            if self._arp_thread and self._arp_thread.is_alive():
                self._arp_thread.join(timeout=0.5)
            self._arp_running = False

    def stop_all_chord_playback(self):
        """Stop strum + arp + release all synth notes. Nuclear option."""
        self.stop_strum()
        self.stop_arpeggio()
        with self.lock:
            self.synth.release_all()
        self.active_midi_notes.clear()

    @property
    def pages(self):
        return VOICE_PAGES[self.voice_name]

    @property
    def abs_param_index(self):
        return self.pages[self.current_page]["params"][self.selected_param]

    def adjust_param(self, delta):
        with self.lock:
            if self.voice_idx == 3:  # FX tab
                if self.fx_selected_param < len(self.fx_reverb_params):
                    self.fx_reverb_params[self.fx_selected_param].nudge(delta)
                    self._sync_reverb_from_params()
            elif self.voice_idx < 3:
                self.voice.params[self.abs_param_index].nudge(delta)

    def reset_params(self):
        with self.lock:
            if self.voice_idx < len(self.voices):
                for p in self.voice.params:
                    p.value = p.default

    def update_preview(self):
        if self.voice_idx >= 3:
            return  # no preview for FX/CHORD
        vals = tuple(round(p.value, 4) for p in self.voice.params)
        h = hash((self.voice_idx, vals))
        if h == self._preview_param_hash and self.voice_idx == self._preview_voice_idx:
            return
        self._preview_param_hash = h
        self._preview_voice_idx = self.voice_idx
        if self.voice_idx == 0:
            # 808 preview: trigger → 150ms hold → release → 1.2s tail, so BODY
            # + RELEASE shape is fully visible in the waveform.
            tmp = Sub808Voice(SR)
            for i, p in enumerate(self.sub808.params):
                tmp.params[i].value = p.value
            tmp.trigger(36, 0.9)
            pre = int(0.15 * SR)
            post = int(1.2 * SR)
            buf_a = tmp.process(pre)
            tmp.release(36)
            buf_b = tmp.process(post)
            self.preview_buf = np.concatenate([buf_a, buf_b]).astype(np.float32)
        elif self.voice_idx == 1:
            tmp = KickVoice(SR)
            for i, p in enumerate(self.kick.params):
                tmp.params[i].value = p.value
            tmp.trigger(36, 0.9)
            full = tmp.process(3000)
            self.preview_buf = full
        else:
            from cypher.synth.mono import MonoSynthVoice
            tmp = MonoSynthVoice(SR)
            for i, p in enumerate(self.synth.params):
                tmp.params[i].value = p.value
            tmp.trigger(60, 0.9)
            tmp.process(128)
            full = tmp.process(1500)
            self.preview_buf = full
        peak = np.max(np.abs(self.preview_buf))
        if peak > 0.001:
            self.preview_buf = self.preview_buf / peak


# ── Rendering ─────────────────────────────────────────────────────────

def format_param_value(player, param_idx, p):
    mapped = p.mapped
    if p.unit == "ms":
        return f"{mapped:.0f}ms"
    elif p.unit == "Hz":
        return f"{mapped:.0f}Hz"
    elif p.unit == "st":
        return f"{mapped:.1f}st"
    elif p.unit == "ct":
        return f"{mapped:.0f}ct"
    elif p.unit == "%":
        return f"{p.value * 100:.0f}%"
    else:
        if player.voice_idx == 0 and param_idx == 10:
            names = ["OFF", "SOFT", "TAPE", "HARD", "CRUSH"]
            return names[min(int(mapped), 4)]
        elif player.voice_idx == 2 and param_idx in (0, 1):
            return WAVE_NAMES[max(0, min(3, int(round(mapped))))]
        elif player.voice_idx == 2 and param_idx == 6:
            return f"{mapped:+.2f}"
        elif player.voice_idx == 2 and param_idx == 7:
            from cypher.core.filters import FILTER_MODE_NAMES
            return FILTER_MODE_NAMES[max(0, min(2, int(round(mapped))))]
        elif player.voice_idx == 2 and param_idx == 14:
            return "FILTER" if int(round(mapped)) == 0 else "PITCH"
        return f"{p.value:.2f}"


def _draw_waveform(screen, player, x, y, w, h):
    panel_rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, (14, 14, 28), panel_rect, border_radius=8)
    pygame.draw.rect(screen, (40, 40, 65), panel_rect, 1, border_radius=8)
    center_y = y + h // 2
    for gx in range(x + 6, x + w - 6, 3):
        screen.set_at((gx, center_y), (40, 40, 85))
    player.update_preview()
    samples = player.preview_buf
    display_w = w - 12
    n_samples = len(samples)
    indices = np.linspace(0, n_samples - 1, display_w).astype(int)
    downsampled = samples[indices]
    margin_v = 8
    usable_h = (h - 2 * margin_v) / 2.0
    clamped = np.clip(downsampled, -1.0, 1.0)
    py_arr = center_y - (clamped * usable_h).astype(int)
    px_arr = np.arange(display_w) + x + 6
    if np.max(np.abs(downsampled)) < 0.001:
        pygame.draw.aaline(screen, (50, 50, 70), (x + 6, center_y), (x + w - 6, center_y))
        return
    fill_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pts_top = [(int(px_arr[i]) - x, int(py_arr[i]) - y) for i in range(display_w)]
    pts_base = [(int(px_arr[-1]) - x, h // 2), (int(px_arr[0]) - x, h // 2)]
    poly = pts_top + pts_base
    if len(poly) >= 3:
        pygame.draw.polygon(fill_surf, (220, 220, 240, 14), poly)
    screen.blit(fill_surf, (x, y))
    glow_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    glow_pts = [(int(px_arr[i]) - x, int(py_arr[i]) - y) for i in range(display_w)]
    if len(glow_pts) >= 2:
        pygame.draw.lines(glow_surf, (200, 200, 220, 30), False, glow_pts, 5)
        pygame.draw.lines(glow_surf, (220, 220, 240, 55), False, glow_pts, 3)
    screen.blit(glow_surf, (x, y))
    points = list(zip(px_arr.astype(int), py_arr.astype(int)))
    if len(points) >= 2:
        pygame.draw.aalines(screen, (240, 240, 250), False, points)


def _format_fx_param(p):
    val = p.mapped
    if p.unit == "ms":
        return f"{val:.0f}ms"
    elif p.unit == "s":
        if val >= 10:
            return f"{val:.0f}s"
        return f"{val:.1f}s"
    elif p.unit == "Hz":
        if val >= 1000:
            return f"{val/1000:.1f}kHz"
        return f"{val:.0f}Hz"
    elif p.unit == "%":
        return f"{p.value * 100:.0f}%"
    return f"{val:.2f}"


def _draw_metronome_strip(screen, fonts, player, hit):
    """Global metronome strip: play/pause + big BPM + nudge + beat flash.

    Positioned as a full-width band just above the help line.
    """
    font, font_bold, font_lg, font_title = fonts
    mouse_pos = pygame.mouse.get_pos()
    metro = player.metronome

    # Strip band
    strip_top = WIN_H - 18 - METRO_STRIP_H
    strip_rect = pygame.Rect(0, strip_top, WIN_W, METRO_STRIP_H)
    pygame.draw.rect(screen, (18, 18, 30), strip_rect)
    pygame.draw.rect(screen, DIM, strip_rect, 1)

    center_y = strip_top + METRO_STRIP_H // 2

    # Play/pause button (left)
    play_label = "[▶]" if not metro.running else "[⏸]"
    play_color = GREEN if metro.running else CYAN
    ps = font_title.render(play_label, True, play_color)
    pr = ps.get_rect(midleft=(14, center_y))
    pc = pr.inflate(16, 10)
    if pc.collidepoint(mouse_pos):
        pygame.draw.rect(screen, (35, 35, 60), pc, border_radius=6)
    screen.blit(ps, pr)
    hit["metro_play"] = pc

    # BPM label
    lbl = font.render("BPM", True, DIM)
    screen.blit(lbl, (pr.right + 22, center_y - lbl.get_height() // 2))

    # Big BPM value
    bpm_text = f"{player.project.bpm:.1f}"
    bs = font_title.render(bpm_text, True, TEXT)
    bx = pr.right + 62
    br = bs.get_rect(midleft=(bx, center_y))
    # Beat-flash halo
    import time as _t
    flash_age = _t.monotonic() - metro.last_tick_time
    if metro.running and flash_age < 0.12:
        alpha = int(255 * (1.0 - flash_age / 0.12))
        halo = pygame.Rect(br.x - 6, br.y - 2, br.width + 12, br.height + 4)
        pygame.draw.rect(screen, (60, 200, 120), halo, 2, border_radius=6)
    screen.blit(bs, br)
    hit["metro_bpm_value"] = br.inflate(12, 8)

    # − / + nudge buttons (1 BPM per click, hold-to-edit shift for 0.1)
    ns = font_lg.render("[−]", True, CYAN)
    nr = ns.get_rect(midleft=(br.right + 16, center_y))
    nc = nr.inflate(10, 8)
    if nc.collidepoint(mouse_pos):
        pygame.draw.rect(screen, (35, 35, 60), nc, border_radius=4)
    screen.blit(ns, nr)
    hit["metro_bpm_minus"] = nc

    ps2 = font_lg.render("[+]", True, CYAN)
    pr2 = ps2.get_rect(midleft=(nr.right + 8, center_y))
    pc2 = pr2.inflate(10, 8)
    if pc2.collidepoint(mouse_pos):
        pygame.draw.rect(screen, (35, 35, 60), pc2, border_radius=4)
    screen.blit(ps2, pr2)
    hit["metro_bpm_plus"] = pc2

    # Right-side hint
    hint = "shift+click = ±0.1"
    hs = font.render(hint, True, DIM)
    screen.blit(hs, (WIN_W - hs.get_width() - 14, center_y - hs.get_height() // 2))


def _draw_midi_keyboard(screen, fonts, player, hit, y):
    """Draw the global MIDI keyboard at the bottom. Returns new y."""
    font, font_bold, font_lg, font_title = fonts
    mouse_pos = pygame.mouse.get_pos()
    bx = 15
    bw = WIN_W - 30

    # Key/Scale selector row
    key_name = NOTE_NAMES[player.global_key]
    scale_name = SCALE_NAMES[player.global_scale_idx]

    # Key selector
    screen.blit(font_bold.render("KEY", True, DIM), (bx, y + 2))
    ks = font_bold.render(f"[< {key_name} >]", True, ORANGE)
    kr = ks.get_rect(topleft=(bx + 42, y + 2))
    # Left arrow
    left_r = pygame.Rect(bx + 42, y, 18, 20)
    right_r = pygame.Rect(kr.right - 18, y, 18, 20)
    hit["key_left"] = left_r
    hit["key_right"] = right_r
    screen.blit(ks, kr)

    # Scale selector
    sx_start = bx + 150
    screen.blit(font_bold.render("SCALE", True, DIM), (sx_start, y + 2))
    ss = font_bold.render(f"[< {scale_name} >]", True, ORANGE)
    sr = ss.get_rect(topleft=(sx_start + 60, y + 2))
    hit["scale_left"] = pygame.Rect(sx_start + 60, y, 18, 20)
    hit["scale_right"] = pygame.Rect(sr.right - 18, y, 18, 20)
    screen.blit(ss, sr)

    y += 22

    # Draw piano keys
    scale_notes = player.get_scale_notes()
    start_midi = player.octave * 12 + 24  # C of current octave

    # White key positions (C D E F G A B pattern)
    white_semitones = [0, 2, 4, 5, 7, 9, 11]
    black_semitones = [1, 3, 6, 8, 10]
    # Black key offsets relative to their white key
    black_offsets = {1: 0, 3: 1, 6: 3, 8: 4, 10: 5}

    # Calculate total white keys in our range
    total_whites = 0
    white_midi = []
    for i in range(MIDI_KB_KEYS):
        midi = start_midi + i
        if (i % 12) in white_semitones:
            white_midi.append(midi)
            total_whites += 1

    # Actually, let's draw 2 octaves of the piano keyboard properly
    # 15 white keys for 2 octaves + C
    n_white = 15
    kw = min(MIDI_KB_WHITE_W, (bw - 4) // n_white)
    kb_total_w = kw * n_white
    kb_x = bx + (bw - kb_total_w) // 2

    # Draw white keys first
    wi = 0
    white_key_rects = {}
    for oct_off in range(2):
        for ws_idx, ws in enumerate(white_semitones):
            midi = start_midi + oct_off * 12 + ws
            if wi >= n_white:
                break
            kx = kb_x + wi * kw
            ky = y
            in_scale = player.note_in_scale(midi)
            is_active = midi in player.active_midi_notes

            if is_active:
                color = KEY_ACTIVE
            elif in_scale:
                color = KEY_WHITE
            else:
                color = KEY_GREY

            rect = pygame.Rect(kx, ky, kw - 1, MIDI_KB_H)
            pygame.draw.rect(screen, color, rect, border_radius=2)
            pygame.draw.rect(screen, (50, 50, 70), rect, 1, border_radius=2)
            white_key_rects[midi] = rect
            hit[f"kb_{midi}"] = rect

            # Draw dot if active
            if is_active:
                pygame.draw.circle(screen, KEY_DOT, (kx + kw // 2, ky + MIDI_KB_H - 10), 4)

            # Note name on lowest octave keys
            if oct_off == 0 and ws_idx == 0:
                nn = NOTE_NAMES[midi % 12]
                ns = font.render(nn, True, (60, 60, 80) if not is_active else BG)
                screen.blit(ns, (kx + 2, ky + MIDI_KB_H - 16))

            wi += 1
    # Extra C at end
    if wi < n_white + 1:
        midi = start_midi + 24
        kx = kb_x + wi * kw
        in_scale = player.note_in_scale(midi)
        is_active = midi in player.active_midi_notes
        color = KEY_ACTIVE if is_active else (KEY_WHITE if in_scale else KEY_GREY)
        rect = pygame.Rect(kx, y, kw - 1, MIDI_KB_H)
        pygame.draw.rect(screen, color, rect, border_radius=2)
        pygame.draw.rect(screen, (50, 50, 70), rect, 1, border_radius=2)
        hit[f"kb_{midi}"] = rect
        if is_active:
            pygame.draw.circle(screen, KEY_DOT, (kx + kw // 2, y + MIDI_KB_H - 10), 4)

    # Draw black keys on top
    for oct_off in range(2):
        wi_base = oct_off * 7  # white key index for this octave
        for bs in black_semitones:
            midi = start_midi + oct_off * 12 + bs
            # Position relative to white keys
            bk_idx = black_offsets[bs]
            kx = kb_x + (wi_base + bk_idx) * kw + kw - MIDI_KB_BLACK_W // 2
            ky = y

            in_scale = player.note_in_scale(midi)
            is_active = midi in player.active_midi_notes

            if is_active:
                color = KEY_ACTIVE
            elif in_scale:
                color = KEY_BLACK
            else:
                color = KEY_BLACK_OUT

            rect = pygame.Rect(kx, ky, MIDI_KB_BLACK_W, MIDI_KB_BLACK_H)
            pygame.draw.rect(screen, color, rect, border_radius=2)
            if in_scale and not is_active:
                # Bright border so in-scale black keys are obvious
                pygame.draw.rect(screen, (100, 90, 140), rect, 1, border_radius=2)
                # Small dot at top
                pygame.draw.circle(screen, (140, 120, 200), (kx + MIDI_KB_BLACK_W // 2, ky + 6), 2)
            hit[f"kb_{midi}"] = rect

            if is_active:
                pygame.draw.circle(screen, KEY_DOT, (kx + MIDI_KB_BLACK_W // 2, ky + MIDI_KB_BLACK_H - 8), 3)

    y += MIDI_KB_H + 4
    return y


def _draw_send_controls(screen, fonts, player, hit, y):
    """Draw per-engine FX send toggle + amount on the current engine page."""
    font, font_bold, font_lg, font_title = fonts
    mouse_pos = pygame.mouse.get_pos()

    engine_idx = tab_voice_idx(player.voice_idx)
    if engine_idx is None or engine_idx >= len(player.fx_sends):
        return y

    send_on = player.fx_sends[engine_idx]
    send_amt = player.fx_send_amounts[engine_idx]

    # FX SEND row
    bx = 15
    label = "FX SEND"
    screen.blit(font_bold.render(label, True, DIM), (bx + 4, y + 2))

    # Toggle button
    tog_label = "ON" if send_on else "OFF"
    tog_color = GREEN if send_on else RED
    tog_s = font_bold.render(f"[{tog_label}]", True, tog_color)
    tog_r = tog_s.get_rect(topleft=(bx + 90, y + 2))
    tog_cr = tog_r.inflate(8, 6)
    tog_hover = tog_cr.collidepoint(mouse_pos)
    if tog_hover:
        tog_color = CYAN
        tog_s = font_bold.render(f"[{tog_label}]", True, tog_color)
        pygame.draw.rect(screen, (35, 35, 60), tog_cr, border_radius=4)
    screen.blit(tog_s, tog_r)
    hit["send_toggle"] = tog_cr

    # Send amount slider (only if on)
    if send_on:
        sx = bx + 160
        sw = 200
        sh = 10
        sy = y + 6
        pct = f"{int(send_amt * 100)}%"
        screen.blit(font.render(pct, True, GREEN), (sx - 40, y + 2))
        slider_rect = pygame.Rect(sx, sy - 6, sw, sh + 12)
        hit["send_slider"] = slider_rect
        pygame.draw.rect(screen, SLIDER_BG, (sx, sy, sw, sh), border_radius=4)
        fill = max(0, min(int(send_amt * sw), sw))
        if fill > 0:
            pygame.draw.rect(screen, (60, 130, 100), (sx, sy, fill, sh), border_radius=4)
        knob_r = 7 if slider_rect.collidepoint(mouse_pos) else 5
        pygame.draw.circle(screen, GREEN, (sx + fill, sy + sh // 2), knob_r)

    # Reverb on/off toggle (right side)
    rv_label = "REVERB ON" if player.reverb_on else "REVERB"
    rv_color = GREEN if player.reverb_on else DIM
    rv_surf = font_bold.render(f"[{rv_label}]", True, rv_color)
    rv_rect = rv_surf.get_rect(topright=(WIN_W - 20, y + 2))
    rv_click = rv_rect.inflate(10, 8)
    rv_hover = rv_click.collidepoint(mouse_pos)
    if rv_hover:
        rv_color = CYAN if not player.reverb_on else GREEN
        rv_surf = font_bold.render(f"[{rv_label}]", True, rv_color)
        pygame.draw.rect(screen, (35, 35, 60), rv_click, border_radius=4)
    screen.blit(rv_surf, rv_rect)
    hit["reverb"] = rv_click

    y += 24
    return y


def _draw_fx_panel(screen, fonts, player, mouse_pos, hit, y):
    """Draw the FX rack when FX tab is selected. Returns new y position."""
    font, font_bold, font_lg, font_title = fonts
    bx, bw = 15, WIN_W - 30

    # ── REVERB slot ──
    expanded = player.fx_reverb_expanded
    rv_on = player.reverb_on
    arrow = "\u25bc" if expanded else "\u25b6"
    slot_header_h = 28

    if expanded:
        slot_h = slot_header_h + len(player.fx_reverb_params) * 36 + 46
    else:
        slot_h = slot_header_h

    slot_rect = pygame.Rect(bx, y, bw, slot_h)
    pygame.draw.rect(screen, PANEL, slot_rect, border_radius=6)
    border_c = GREEN if rv_on else DIM
    pygame.draw.rect(screen, border_c, slot_rect, 1, border_radius=6)

    header_rect = pygame.Rect(bx, y, bw - 80, slot_header_h)
    hit["fx_reverb_header"] = header_rect
    hdr_hover = header_rect.collidepoint(mouse_pos)
    hdr_color = CYAN if hdr_hover else TEXT
    screen.blit(font_bold.render(f" {arrow} REVERB", True, hdr_color), (bx + 8, y + 5))

    on_label = "ON" if rv_on else "OFF"
    on_color = GREEN if rv_on else RED
    on_s = font_bold.render(f"[{on_label}]", True, on_color)
    on_r = on_s.get_rect(topright=(bx + bw - 10, y + 4))
    on_cr = on_r.inflate(8, 6)
    on_hover = on_cr.collidepoint(mouse_pos)
    if on_hover:
        on_color = CYAN
        on_s = font_bold.render(f"[{on_label}]", True, on_color)
        pygame.draw.rect(screen, (35, 35, 60), on_cr, border_radius=4)
    screen.blit(on_s, on_r)
    hit["fx_reverb_toggle"] = on_cr
    y += slot_header_h

    if expanded:
        for pi, p in enumerate(player.fx_reverb_params):
            selected = pi == player.fx_selected_param
            row_y = y + pi * 36
            row_rect = pygame.Rect(bx + 4, row_y, bw - 8, 32)
            hit[f"fx_param_row_{pi}"] = row_rect
            row_hover = row_rect.collidepoint(mouse_pos)
            if selected:
                pygame.draw.rect(screen, (40, 40, 70), row_rect, border_radius=4)
            elif row_hover:
                pygame.draw.rect(screen, (32, 32, 55), row_rect, border_radius=4)
            color = YELLOW if selected else TEXT
            prefix = "\u25b6 " if selected else "  "
            val_str = _format_fx_param(p)
            screen.blit(font_bold.render(f"{prefix}{p.label}", True, color), (bx + 12, row_y + 6))
            screen.blit(font.render(val_str, True, color), (bx + 150, row_y + 6))
            sx = bx + 260
            sw = bw - 280
            sh = 10
            sy = row_y + 10
            slider_rect = pygame.Rect(sx, sy - 6, sw, sh + 12)
            hit[f"fx_slider_{pi}"] = slider_rect
            pygame.draw.rect(screen, SLIDER_BG, (sx, sy, sw, sh), border_radius=4)
            fill = max(0, min(int(p.value * sw), sw))
            if fill > 0:
                bc = GREEN if selected else (60, 130, 100)
                pygame.draw.rect(screen, bc, (sx, sy, fill, sh), border_radius=4)
            knob_r = 8 if (selected and slider_rect.collidepoint(mouse_pos)) else 6
            pygame.draw.circle(screen, color, (sx + fill, sy + sh // 2), knob_r)

        y += len(player.fx_reverb_params) * 36 + 4

        # Mode radio buttons
        screen.blit(font_bold.render("  MODE", True, TEXT), (bx + 12, y + 4))
        mode_x = bx + 140
        for mi, mname in enumerate(DattorroPlateReverb.MODE_NAMES):
            active = mi == player.fx_reverb_mode
            mc = GREEN if active else DIM
            ms = font_bold.render(f"[{mname}]", True, mc)
            mr = ms.get_rect(topleft=(mode_x, y + 4))
            mcr = mr.inflate(6, 6)
            mhover = mcr.collidepoint(mouse_pos)
            if mhover:
                mc = CYAN if not active else GREEN
                ms = font_bold.render(f"[{mname}]", True, mc)
                pygame.draw.rect(screen, (35, 35, 60), mcr, border_radius=4)
            screen.blit(ms, mr)
            hit[f"fx_mode_{mi}"] = mcr
            mode_x += mr.width + 10
        y += 36

    y += 8

    # ── Placeholder slots (DELAY, FLANGER) ──
    for slot_name in ["DELAY", "FLANGER"]:
        ph_rect = pygame.Rect(bx, y, bw, 28)
        pygame.draw.rect(screen, (22, 22, 40), ph_rect, border_radius=6)
        pygame.draw.rect(screen, (50, 50, 65), ph_rect, 1, border_radius=6)
        screen.blit(font_bold.render(f" \u25b6 {slot_name}", True, (50, 50, 70)), (bx + 8, y + 5))
        screen.blit(font.render("coming soon", True, (40, 40, 55)), (bx + bw - 130, y + 6))
        y += 34

    return y


def _draw_sampler_bpm_row(screen, fonts, player, slot, mouse_pos, hit, y):
    """BPM status line + fine-tune row for the focused sampler slot.

    Layout:
        Sample: 92 · Project: 140 · [Free] [Match]
        (when Match on) Sample BPM: 92.0 [−][+] ×2 ÷2 [Tap]
    """
    font, font_bold, _, _ = fonts
    proj_bpm = player.project.bpm
    sample_bpm = slot.sample_bpm
    conf = slot.bpm_confidence
    has_bpm = sample_bpm > 0.0

    # ── Line 1: Sample/Project badges + Match/Free toggle ──────────────
    cx = 20
    screen.blit(font.render("Sample:", True, DIM), (cx, y + 2))
    cx += 60
    bpm_text = f"{sample_bpm:5.1f}" if has_bpm else "?"
    bpm_color = GREEN if has_bpm and conf >= 0.6 else (YELLOW if has_bpm else RED)
    screen.blit(font_bold.render(bpm_text, True, bpm_color), (cx, y + 2))
    cx += 60
    screen.blit(font.render("·  Project:", True, DIM), (cx, y + 2))
    cx += 78
    screen.blit(font_bold.render(f"{proj_bpm:5.1f}", True, CYAN), (cx, y + 2))
    cx += 70

    # Free / Match toggle (disabled if no BPM)
    free_color = CYAN if not slot.match_mode else DIM
    match_color = CYAN if slot.match_mode else DIM
    if not has_bpm:
        match_color = (80, 80, 80)
    fs = font_bold.render("[Free]", True, free_color)
    fr = fs.get_rect(topleft=(cx, y))
    fc = fr.inflate(10, 6)
    if fc.collidepoint(mouse_pos):
        pygame.draw.rect(screen, (35, 35, 60), fc, border_radius=4)
    screen.blit(fs, fr)
    hit["sampler_bpm_free"] = fc
    cx += fr.width + 14
    ms = font_bold.render("[Match]", True, match_color)
    mr = ms.get_rect(topleft=(cx, y))
    mc = mr.inflate(10, 6)
    if mc.collidepoint(mouse_pos) and has_bpm:
        pygame.draw.rect(screen, (35, 35, 60), mc, border_radius=4)
    screen.blit(ms, mr)
    hit["sampler_bpm_match"] = mc
    cx += mr.width + 14

    # Tap tempo fallback when no BPM or user wants to re-detect
    if not has_bpm:
        ts = font_bold.render("[Tap tempo]", True, YELLOW)
        tr = ts.get_rect(topleft=(cx, y))
        tc = tr.inflate(10, 6)
        if tc.collidepoint(mouse_pos):
            pygame.draw.rect(screen, (60, 60, 35), tc, border_radius=4)
        screen.blit(ts, tr)
        hit["sampler_bpm_tap"] = tc
    y += 24

    # ── Line 2: fine-tune (only when Match active) ─────────────────────
    if slot.match_mode and has_bpm:
        cx = 20
        screen.blit(font.render("Sample BPM:", True, DIM), (cx, y + 2))
        cx += 90
        val_txt = f"{sample_bpm:5.1f}"
        vcol = CYAN if slot.user_corrected else TEXT
        screen.blit(font_bold.render(val_txt, True, vcol), (cx, y + 2))
        cx += 50
        for label, key in (("[−]", "sampler_bpm_minus"),
                            ("[+]", "sampler_bpm_plus")):
            bs = font_bold.render(label, True, CYAN)
            br = bs.get_rect(topleft=(cx, y))
            bc = br.inflate(8, 6)
            if bc.collidepoint(mouse_pos):
                pygame.draw.rect(screen, (35, 35, 60), bc, border_radius=4)
            screen.blit(bs, br)
            hit[key] = bc
            cx += br.width + 6
        cx += 6
        for label, key in (("×2", "sampler_bpm_double"),
                            ("÷2", "sampler_bpm_half")):
            bs = font_bold.render(label, True, YELLOW)
            br = bs.get_rect(topleft=(cx, y))
            bc = br.inflate(10, 6)
            if bc.collidepoint(mouse_pos):
                pygame.draw.rect(screen, (60, 60, 35), bc, border_radius=4)
            screen.blit(bs, br)
            hit[key] = bc
            cx += br.width + 10
        cx += 4
        ts = font_bold.render("[Tap]", True, YELLOW)
        tr = ts.get_rect(topleft=(cx, y))
        tc = tr.inflate(10, 6)
        if tc.collidepoint(mouse_pos):
            pygame.draw.rect(screen, (60, 60, 35), tc, border_radius=4)
        screen.blit(ts, tr)
        hit["sampler_bpm_tap"] = tc
        y += 22
    return y


def _draw_sampler_panel(screen, fonts, player, mouse_pos, hit, y):
    """SAMPLER tab UI: folder picker, file list, 4x4 pad grid, load-to-pad."""
    font, font_bold, _, _ = fonts
    sampler = player.sampler

    # Folder row + LOAD FOLDER button
    folder_label = player.sampler_folder or "(no folder — click LOAD FOLDER)"
    screen.blit(font.render(f"Folder: {folder_label}", True, TEXT), (20, y))
    lf_surf = font_bold.render("[LOAD FOLDER]", True, CYAN)
    lf_rect = lf_surf.get_rect(topright=(WIN_W - 20, y))
    lf_click = lf_rect.inflate(10, 8)
    if lf_click.collidepoint(mouse_pos):
        pygame.draw.rect(screen, (35, 35, 60), lf_click, border_radius=4)
    screen.blit(lf_surf, lf_rect)
    hit["sampler_load_folder"] = lf_click
    y += 26

    # File list (left 2/3) + pad grid (right 1/3)
    list_x, list_w = 20, int(WIN_W * 0.60)
    list_h = 210
    pygame.draw.rect(screen, (20, 20, 35), (list_x, y, list_w, list_h), border_radius=6)
    pygame.draw.rect(screen, DIM, (list_x, y, list_w, list_h), 1, border_radius=6)

    files = player.sampler_files
    if not files:
        screen.blit(font.render("No audio files. Pick a folder with .wav files.",
                                True, DIM), (list_x + 12, y + 12))
    else:
        row_h = 18
        max_rows = (list_h - 12) // row_h
        start_idx = max(0, player.sampler_selected_file - max_rows // 2)
        end_idx = min(len(files), start_idx + max_rows)
        for i, path in enumerate(files[start_idx:end_idx]):
            ri = start_idx + i
            row_y = y + 6 + i * row_h
            row_rect = pygame.Rect(list_x + 4, row_y, list_w - 8, row_h)
            if ri == player.sampler_selected_file:
                pygame.draw.rect(screen, (40, 40, 70), row_rect, border_radius=3)
                color = YELLOW
            else:
                color = TEXT
            screen.blit(font.render(path.name, True, color), (list_x + 10, row_y + 1))
            hit[f"sampler_file_{ri}"] = row_rect

    # 4x4 pad grid (right)
    right_x = list_x + list_w + 10
    right_w = WIN_W - right_x - 20
    grid_h = 160
    pad_w = (right_w - 4 * 3) // 4
    pad_h = (grid_h - 4 * 3) // 4
    for r in range(4):
        for c in range(4):
            idx = r * 4 + c
            px = right_x + c * (pad_w + 4)
            py = y + r * (pad_h + 4)
            slot = sampler.slots[idx]
            focused = idx == sampler.focused_slot_idx
            loaded = slot.loaded
            bg = (60, 130, 100) if loaded else (35, 35, 55)
            if focused:
                bg = (100, 180, 140) if loaded else (80, 80, 110)
            pygame.draw.rect(screen, bg, (px, py, pad_w, pad_h), border_radius=4)
            pygame.draw.rect(screen, DIM, (px, py, pad_w, pad_h), 1, border_radius=4)
            lbl = font_bold.render(f"{idx+1}", True, TEXT if loaded else DIM)
            screen.blit(lbl, (px + 4, py + 2))
            if loaded:
                name = slot.name
                if len(name) > 10:
                    name = name[:9] + "…"
                screen.blit(font.render(name, True, (230, 230, 240)),
                            (px + 4, py + 20))
            hit[f"sampler_pad_{idx}"] = pygame.Rect(px, py, pad_w, pad_h)

    # LOAD → PAD + CLEAR buttons under the grid
    after_y = y + grid_h + 6
    ld_surf = font_bold.render(
        f"[LOAD → PAD {sampler.focused_slot_idx + 1}]", True, GREEN)
    ld_rect = ld_surf.get_rect(topleft=(right_x, after_y))
    ld_click = ld_rect.inflate(10, 8)
    if ld_click.collidepoint(mouse_pos):
        pygame.draw.rect(screen, (40, 60, 50), ld_click, border_radius=4)
    screen.blit(ld_surf, ld_rect)
    hit["sampler_load_to_pad"] = ld_click

    cl_surf = font_bold.render("[CLEAR PAD]", True, RED)
    cl_rect = cl_surf.get_rect(topleft=(right_x, after_y + 22))
    cl_click = cl_rect.inflate(10, 8)
    if cl_click.collidepoint(mouse_pos):
        pygame.draw.rect(screen, (60, 35, 40), cl_click, border_radius=4)
    screen.blit(cl_surf, cl_rect)
    hit["sampler_clear_pad"] = cl_click

    y += list_h + 12

    # Info line
    st = sampler.get_state()
    info = (f"Pad {st['focused_slot']+1}: {st['focused_name'] or '(empty)'}   "
            f"Active: {st['active_voices']}/{st['max_voices']}   "
            f"Loaded: {st['slots_loaded']}/16")
    screen.blit(font.render(info, True, DIM), (20, y))
    y += 20

    # BPM status line + fine-tune row
    slot_for_bpm = sampler.slots[sampler.focused_slot_idx]
    if slot_for_bpm.loaded:
        y = _draw_sampler_bpm_row(screen, fonts, player, slot_for_bpm,
                                  mouse_pos, hit, y)

    # MODE radios + REVERSE toggle + FREEZE arm (only when slot loaded)
    slot = sampler.slots[sampler.focused_slot_idx]
    if slot.loaded:
        from cypher.sampler.sampler import (
            MODE_NAMES as _SMN, P_MODE as _PM, P_REVERSE as _PR,
        )
        # MODE row
        cur_mode = int(round(slot.params[_PM].mapped))
        cx = 20
        screen.blit(font.render("MODE:", True, TEXT), (cx, y + 4))
        cx += 58
        for mi, name in enumerate(_SMN):
            sel = mi == cur_mode
            color = CYAN if sel else DIM
            label = f"[{name}]" if sel else f" {name} "
            ms = font_bold.render(label, True, color)
            mr = ms.get_rect(topleft=(cx, y))
            mc = mr.inflate(8, 6)
            if mc.collidepoint(mouse_pos) and not sel:
                pygame.draw.rect(screen, (35, 35, 60), mc, border_radius=4)
            screen.blit(ms, mr)
            hit[f"sampler_mode_{mi}"] = mc
            cx += mr.width + 14

        # REVERSE toggle (right side)
        rev_on = int(round(slot.params[_PR].mapped)) == 1
        rev_color = GREEN if rev_on else DIM
        rev_label = "[REV: ON]" if rev_on else "[REV]"
        rs = font_bold.render(rev_label, True, rev_color)
        rr = rs.get_rect(topright=(WIN_W - 20, y))
        rc = rr.inflate(10, 6)
        if rc.collidepoint(mouse_pos):
            pygame.draw.rect(screen, (35, 35, 60), rc, border_radius=4)
        screen.blit(rs, rr)
        hit["sampler_reverse"] = rc

        # FREEZE arm (next to REV)
        fz_on = slot.freeze_armed
        fz_color = MAGENTA if fz_on else DIM
        fz_label = "[FREEZE: ARMED]" if fz_on else "[FREEZE]"
        fs = font_bold.render(fz_label, True, fz_color)
        fr = fs.get_rect(topright=(rr.left - 10, y))
        fc = fr.inflate(10, 6)
        if fc.collidepoint(mouse_pos):
            pygame.draw.rect(screen, (50, 35, 50), fc, border_radius=4)
        screen.blit(fs, fr)
        hit["sampler_freeze_arm"] = fc
        y += 26

        # Gross Beat preset row
        from cypher.sampler.gross_beat import PRESET_NAMES as _GBN
        cur_gb = sampler.gross_beat.preset
        gx = 20
        screen.blit(font.render("GROSS:", True, TEXT), (gx, y + 4))
        gx += 66
        for gi, name in enumerate(_GBN):
            sel = gi == cur_gb
            color = GREEN if sel else DIM
            label = f"[{name}]" if sel else f" {name} "
            gs = font_bold.render(label, True, color)
            gr = gs.get_rect(topleft=(gx, y))
            gc = gr.inflate(6, 6)
            if gc.collidepoint(mouse_pos) and not sel:
                pygame.draw.rect(screen, (35, 35, 60), gc, border_radius=4)
            screen.blit(gs, gr)
            hit[f"sampler_gross_{gi}"] = gc
            gx += gr.width + 8
        y += 26

        # PITCH / GAIN / SLICES slider row
        from cypher.sampler.sampler import (
            P_PITCH as _PP, P_GAIN as _PG, P_SLICES as _PSL,
            P_FZ_POS as _PFP, P_FZ_GRAIN as _PFG,
            P_FZ_MOTION as _PFM, P_FZ_RATE as _PFR,
            DIVISION_NAMES as _DIV_NAMES, DIVISION_VALUES as _DIV_VALS,
            MODE_CHOP as _MC,
        )

        def _slider(sx, sy, sw, label, param, key, fmt):
            screen.blit(font.render(label, True, TEXT), (sx, sy))
            screen.blit(font.render(fmt(param), True, YELLOW), (sx + 70, sy))
            track_y = sy + 17
            pygame.draw.rect(screen, SLIDER_BG, (sx, track_y, sw, 10),
                             border_radius=4)
            fill = max(0, min(int(param.value * sw), sw))
            if fill > 0:
                pygame.draw.rect(screen, (60, 130, 100),
                                 (sx, track_y, fill, 10), border_radius=4)
            pygame.draw.circle(screen, TEXT, (sx + fill, track_y + 5), 6)
            hit[key] = pygame.Rect(sx, track_y - 6, sw, 22)

        # PITCH as +/- buttons (discrete semitones)
        slider_w = (WIN_W - 40 - 60) // 3
        pitch_p = slot.params[_PP]
        pitch_st = pitch_p.mapped
        px = 20
        screen.blit(font.render("PITCH", True, TEXT), (px, y))
        minus = font_bold.render("[−1]", True, CYAN)
        mr = minus.get_rect(topleft=(px + 60, y))
        mc = mr.inflate(8, 6)
        if mc.collidepoint(mouse_pos):
            pygame.draw.rect(screen, (35, 35, 60), mc, border_radius=4)
        screen.blit(minus, mr)
        hit["sampler_pitch_minus"] = mc
        val_txt = f"{pitch_st:+.0f} st"
        screen.blit(font_bold.render(val_txt, True, YELLOW),
                    (px + 110, y + 2))
        plus = font_bold.render("[+1]", True, CYAN)
        pr = plus.get_rect(topleft=(px + 170, y))
        pc = pr.inflate(8, 6)
        if pc.collidepoint(mouse_pos):
            pygame.draw.rect(screen, (35, 35, 60), pc, border_radius=4)
        screen.blit(plus, pr)
        hit["sampler_pitch_plus"] = pc

        _slider(20 + slider_w + 30, y, slider_w, "GAIN", slot.params[_PG],
                "sampler_gain_slider", lambda p: f"{p.mapped:.2f}")
        _slider(20 + 2 * (slider_w + 30), y, slider_w, "SLICES",
                slot.params[_PSL], "sampler_slices_slider",
                lambda p: f"{int(round(p.mapped))}")
        y += 32

        # Beat-division row (CHOP mode only). Sets SLICES to (beats × division).
        if slot.mode == _MC:
            bpm = player.project.bpm
            sec = slot.length / max(1, slot.source_rate)
            beats = max(1.0, sec * bpm / 60.0)
            dx = 20
            screen.blit(font.render("DIVISION:", True, TEXT), (dx, y + 4))
            dx += 82
            for di, name in enumerate(_DIV_NAMES):
                target = int(min(32, max(1, round(beats * _DIV_VALS[di]))))
                ds = font_bold.render(f"[{name}]", True, CYAN)
                dr = ds.get_rect(topleft=(dx, y))
                dc = dr.inflate(6, 6)
                if dc.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, (35, 35, 60), dc, border_radius=4)
                screen.blit(ds, dr)
                hit[f"sampler_div_{di}"] = dc
                dx += dr.width + 6
            y += 26

        # FREEZE sub-panel (only when armed)
        if slot.freeze_armed:
            mot_idx = max(0, min(3, int(round(slot.params[_PFM].mapped))))
            mx = 20
            screen.blit(font.render("MOTION:", True, TEXT), (mx, y + 4))
            mx += 72
            from cypher.sampler.freeze import MOTION_NAMES as _FZMN
            for mi, mname in enumerate(_FZMN):
                sel = mi == mot_idx
                color = MAGENTA if sel else DIM
                label = f"[{mname}]" if sel else f" {mname} "
                ms = font_bold.render(label, True, color)
                mr = ms.get_rect(topleft=(mx, y))
                mc = mr.inflate(6, 6)
                if mc.collidepoint(mouse_pos) and not sel:
                    pygame.draw.rect(screen, (50, 35, 50), mc, border_radius=4)
                screen.blit(ms, mr)
                hit[f"sampler_fz_motion_{mi}"] = mc
                mx += mr.width + 6
            y += 26

            rate_label = {0: "RATE (—)", 1: "DRIFT",
                          2: "SWING Hz", 3: "DECAY s"}.get(mot_idx, "RATE")
            _slider(20, y, slider_w, "FZ POS", slot.params[_PFP],
                    "sampler_fz_pos_slider",
                    lambda p: f"{int(p.mapped * 100)}%")
            _slider(20 + slider_w + 30, y, slider_w, "FZ GRAIN",
                    slot.params[_PFG], "sampler_fz_grain_slider",
                    lambda p: f"{int(p.mapped)}ms")
            _slider(20 + 2 * (slider_w + 30), y, slider_w, rate_label,
                    slot.params[_PFR], "sampler_fz_rate_slider",
                    lambda p: f"{p.value:.2f}")
            y += 32

    # Waveform view of focused slot (full width)
    _draw_sampler_waveform(screen, fonts, sampler, 20, y, WIN_W - 40, 130, hit=hit)
    y += 138

    # Key hint
    hint = ("Keys: A W S E D F T G Y H U J K → pads/slices 1–13 · "
            "Z/X = shift pad group (oct±) for 13–16")
    screen.blit(font.render(hint, True, DIM), (20, y))
    y += 18

    return y


def _draw_sampler_waveform(screen, fonts, sampler, x, y, w, h, hit=None):
    """Waveform view of focused slot with START/END handles + slice markers.

    When `hit` is provided, registers draggable hit rects for slice boundaries
    (`sampler_chop_<N>`) and a `sampler_chop_reset` button so the event loop
    can wire interactive slice nudging.
    """
    import numpy as np
    font = fonts[0]
    pygame.draw.rect(screen, (12, 12, 22), (x, y, w, h), border_radius=6)
    pygame.draw.rect(screen, (40, 40, 65), (x, y, w, h), 1, border_radius=6)

    slot = sampler.slots[sampler.focused_slot_idx]
    center_y = y + h // 2

    if not slot.loaded or slot.length < 2:
        msg = f"Pad {sampler.focused_slot_idx + 1} — no sample loaded"
        tx = x + (w - font.size(msg)[0]) // 2
        screen.blit(font.render(msg, True, DIM), (tx, center_y - 8))
        return

    # Center guide
    for gx in range(x + 4, x + w - 4, 4):
        screen.set_at((gx, center_y), (40, 40, 65))

    # Peak-envelope waveform (mirrored)
    data = slot.data
    n = len(data)
    display_w = w - 12
    bucket = max(1, n // display_w)
    usable = (n // bucket) * bucket
    if usable > 0:
        peaks = np.max(np.abs(data[:usable].reshape(-1, bucket)), axis=1)
    else:
        peaks = np.abs(data)
    if len(peaks) < display_w:
        peaks = np.pad(peaks, (0, display_w - len(peaks)))
    else:
        peaks = peaks[:display_w]
    pk_max = float(np.max(peaks)) if len(peaks) else 1.0
    if pk_max > 0.001:
        peaks = peaks / pk_max

    half_h = (h - 16) / 2.0
    base_x = x + 6
    col = (200, 215, 235)
    for i, p in enumerate(peaks):
        px = base_x + i
        amp = int(p * half_h)
        if amp < 1:
            screen.set_at((px, center_y), (60, 70, 90))
        else:
            pygame.draw.line(screen, col, (px, center_y - amp), (px, center_y + amp), 1)

    # START (green) / END (yellow) vertical lines
    from cypher.sampler.sampler import P_START, P_END, MODE_CHOP
    p_start = max(0.0, min(1.0, slot.params[P_START].mapped))
    p_end = max(0.0, min(1.0, slot.params[P_END].mapped))
    sx_pix = base_x + int(p_start * (display_w - 1))
    ex_pix = base_x + int(p_end * (display_w - 1))
    pygame.draw.line(screen, GREEN, (sx_pix, y + 4), (sx_pix, y + h - 4), 1)
    pygame.draw.line(screen, YELLOW, (ex_pix, y + 4), (ex_pix, y + h - 4), 1)

    # CHOP slice boundary lines (magenta) + slice numbers
    if slot.mode == MODE_CHOP:
        slot.refresh_slices()
        n_slices = len(slot._slice_points)
        line_col = CYAN if slot.slice_manual else MAGENTA
        for si, (s_start, _) in enumerate(slot._slice_points):
            frac = s_start / max(1, slot.length)
            sx = base_x + int(frac * (display_w - 1))
            if si > 0:
                pygame.draw.line(screen, line_col, (sx, y + 4), (sx, y + h - 4), 2)
                if hit is not None:
                    hit[f"sampler_chop_{si}"] = pygame.Rect(sx - 5, y + 4, 11, h - 8)
            if n_slices <= 16 and display_w // max(1, n_slices) > 20:
                screen.blit(font.render(str(si + 1), True, line_col), (sx + 2, y + 4))
        # Register strip geometry so the motion handler can map mouse x → frac
        if hit is not None:
            hit["sampler_chop_strip"] = pygame.Rect(base_x, y, display_w, h)
            if slot.slice_manual:
                btn_w, btn_h = 64, 18
                btn_rect = pygame.Rect(x + w - btn_w - 6, y + h - btn_h - 4,
                                       btn_w, btn_h)
                pygame.draw.rect(screen, (40, 80, 40), btn_rect, border_radius=3)
                pygame.draw.rect(screen, GREEN, btn_rect, 1, border_radius=3)
                lbl = font.render("RESET", True, GREEN)
                screen.blit(lbl, (btn_rect.x + (btn_w - lbl.get_width()) // 2,
                                  btn_rect.y + (btn_h - lbl.get_height()) // 2))
                hit["sampler_chop_reset"] = btn_rect

    # Live playhead for any voice playing this slot
    for v in sampler._voices:
        if v.is_active and v.slot is slot:
            pos_frac = max(0.0, min(1.0, v._position / max(1, slot.length)))
            ph_x = base_x + int(pos_frac * (display_w - 1))
            pygame.draw.line(screen, CYAN, (ph_x, y + 2), (ph_x, y + h - 2), 1)
            break

    # Duration label top-right
    dur_sec = slot.length / max(1, slot.source_rate)
    dur_str = f"{dur_sec*1000:.0f}ms" if dur_sec < 1 else f"{dur_sec:.2f}s"
    lbl = font.render(dur_str, True, DIM)
    screen.blit(lbl, (x + w - lbl.get_width() - 8, y + 4))


def _draw_chord_panel(screen, fonts, player, mouse_pos, hit, y):
    """Draw the CHORD engine UI. Returns new y."""
    font, font_bold, font_lg, font_title = fonts
    bx, bw = 15, WIN_W - 30

    # ── Progression list (OP-1 style scrollable) ──
    screen.blit(font_bold.render("PROGRESSION", True, TEXT), (bx + 4, y))
    y += 22

    # Show 5 items centered on current selection
    list_h = 5
    prog_idx = player.chord_prog_idx
    n_progs = len(PROGRESSION_LIST)
    start = prog_idx - list_h // 2
    list_rect = pygame.Rect(bx, y, bw, list_h * 24 + 4)
    pygame.draw.rect(screen, PANEL, list_rect, border_radius=6)
    pygame.draw.rect(screen, DIM, list_rect, 1, border_radius=6)

    for i in range(list_h):
        actual_idx = (start + i) % n_progs
        pname = PROGRESSION_LIST[actual_idx]
        row_y = y + 2 + i * 24
        is_selected = actual_idx == prog_idx

        row_rect = pygame.Rect(bx + 4, row_y, bw - 8, 22)
        hit[f"prog_{actual_idx}"] = row_rect

        if is_selected:
            pygame.draw.rect(screen, (45, 45, 80), row_rect, border_radius=3)
            prefix = "\u25b6 "
            color = CYAN
        else:
            color = DIM
            prefix = "  "

        s = font_bold.render(f"{prefix}{pname}", True, color)
        screen.blit(s, (bx + 12, row_y + 2))

        # Show chord names in the progression
        root = player.octave * 12 + 24
        plen = progression_length(pname)
        chord_labels = []
        for step in range(plen):
            _, lbl = build_progression_chord(root, pname, step)
            chord_labels.append(lbl)
        chord_str = " - ".join(chord_labels)
        cs = font.render(chord_str, True, YELLOW if is_selected else (60, 60, 80))
        screen.blit(cs, (bx + 180, row_y + 2))

    y += list_h * 24 + 8

    # ── Steps display ──
    pname = PROGRESSION_LIST[prog_idx]
    plen = progression_length(pname)
    root = player.octave * 12 + 24

    screen.blit(font_bold.render("STEPS", True, TEXT), (bx + 4, y))
    step_x = bx + 70
    for si in range(plen):
        _, lbl = build_progression_chord(root, pname, si)
        is_current = si == player.chord_step
        sc = CYAN if is_current else DIM
        s = font_bold.render(f"[{lbl}]", True, sc)
        sr = s.get_rect(topleft=(step_x, y))
        scr = sr.inflate(6, 6)
        shover = scr.collidepoint(mouse_pos)
        if shover:
            sc = GREEN
            s = font_bold.render(f"[{lbl}]", True, sc)
            pygame.draw.rect(screen, (35, 35, 60), scr, border_radius=4)
        if is_current:
            pygame.draw.rect(screen, (30, 50, 60), scr, border_radius=4)
        screen.blit(s, sr)
        hit[f"chord_step_{si}"] = scr
        step_x += sr.width + 10
    y += 28

    # ── Mode selector: STRUM DOWN / STRUM UP / ARPEGGIO ──
    screen.blit(font_bold.render("MODE", True, TEXT), (bx + 4, y + 2))
    mode_names = ["CHORD", "STRUM \u2193", "STRUM \u2191", "ARP"]
    mode_x = bx + 70
    for mi, mn in enumerate(mode_names):
        active = mi == player.chord_mode
        mc = GREEN if active else DIM
        ms = font_bold.render(f"[{mn}]", True, mc)
        mr = ms.get_rect(topleft=(mode_x, y + 2))
        mcr = mr.inflate(6, 6)
        mhover = mcr.collidepoint(mouse_pos)
        if mhover:
            mc = CYAN if not active else GREEN
            ms = font_bold.render(f"[{mn}]", True, mc)
            pygame.draw.rect(screen, (35, 35, 60), mcr, border_radius=4)
        screen.blit(ms, mr)
        hit[f"chord_mode_{mi}"] = mcr
        mode_x += mr.width + 10
    y += 28

    # ── Rhythm division ──
    screen.blit(font_bold.render("RATE", True, TEXT), (bx + 4, y + 2))
    rate_x = bx + 70
    for ri, rn in enumerate(RHYTHM_NAMES):
        active = ri == player.chord_rhythm_idx
        rc = GREEN if active else DIM
        rs = font_bold.render(f"[{rn}]", True, rc)
        rr = rs.get_rect(topleft=(rate_x, y + 2))
        rcr = rr.inflate(6, 6)
        rhover = rcr.collidepoint(mouse_pos)
        if rhover:
            rc = CYAN if not active else GREEN
            rs = font_bold.render(f"[{rn}]", True, rc)
            pygame.draw.rect(screen, (35, 35, 60), rcr, border_radius=4)
        screen.blit(rs, rr)
        hit[f"chord_rate_{ri}"] = rcr
        rate_x += rr.width + 10
    y += 28

    # ── BPM ──
    screen.blit(font_bold.render("BPM", True, TEXT), (bx + 4, y + 2))
    bpm_s = font_bold.render(f"[< {int(player.chord_bpm)} >]", True, ORANGE)
    bpm_r = bpm_s.get_rect(topleft=(bx + 70, y + 2))
    screen.blit(bpm_s, bpm_r)
    hit["bpm_left"] = pygame.Rect(bx + 70, y, 18, 20)
    hit["bpm_right"] = pygame.Rect(bpm_r.right - 18, y, 18, 20)

    # Swing slider
    swing_x = bx + 250
    screen.blit(font_bold.render("SWING", True, TEXT), (swing_x, y + 2))
    sw_sx = swing_x + 70
    sw_sw = 180
    sw_sh = 10
    sw_sy = y + 6
    sw_pct = f"{int(player.chord_swing * 100)}%"
    screen.blit(font.render(sw_pct, True, ORANGE), (sw_sx - 40, y + 2))
    swing_slider = pygame.Rect(sw_sx, sw_sy - 6, sw_sw, sw_sh + 12)
    hit["swing_slider"] = swing_slider
    pygame.draw.rect(screen, SLIDER_BG, (sw_sx, sw_sy, sw_sw, sw_sh), border_radius=4)
    fill = max(0, min(int(player.chord_swing * sw_sw), sw_sw))
    if fill > 0:
        pygame.draw.rect(screen, (180, 120, 40), (sw_sx, sw_sy, fill, sw_sh), border_radius=4)
    knob_r = 7 if swing_slider.collidepoint(mouse_pos) else 5
    pygame.draw.circle(screen, ORANGE, (sw_sx + fill, sw_sy + sw_sh // 2), knob_r)
    y += 28

    # ── Current chord info ──
    pname = PROGRESSION_LIST[player.chord_prog_idx]
    root = player.octave * 12 + 24
    notes, label = build_progression_chord(root, pname, player.chord_step)
    note_str = " ".join(NOTE_NAMES[n % 12] for n in notes)
    mode_str = ["CHORD", "STRUM \u2193", "STRUM \u2191", "ARP"][player.chord_mode]
    info = f"Current: {label} ({note_str})  Mode: {mode_str}"
    screen.blit(font_bold.render(info, True, MAGENTA), (bx + 4, y))
    y += 22

    return y


def draw(screen, fonts, player, held_notes, held_chord_key, held_chord_notes):
    """Render the full UI. Returns dict of clickable rects."""
    font, font_bold, font_lg, font_title = fonts
    hit: dict[str, pygame.Rect] = {}
    mouse_pos = pygame.mouse.get_pos()
    screen.fill(BG)
    y = 10

    # ── Title ──
    surf = font_title.render(f"CYPHER {player.voice_name}", True, CYAN)
    screen.blit(surf, (WIN_W // 2 - surf.get_width() // 2, y))
    y += 34

    # ── Voice tabs (clickable) ──
    tab_x = WIN_W // 2 - 200
    for i, vn in enumerate(VOICE_NAMES):
        active = i == player.voice_idx
        label = f"[{vn}]" if active else f" {vn} "
        surf = font_bold.render(label, True, TEXT)
        rect = surf.get_rect(topleft=(tab_x, y))
        click_rect = rect.inflate(8, 8)
        hover = click_rect.collidepoint(mouse_pos)
        color = CYAN if active else (GREEN if hover else DIM)
        surf = font_bold.render(label, True, color)
        if hover and not active:
            pygame.draw.rect(screen, (35, 35, 60), click_rect, border_radius=4)
        screen.blit(surf, rect)
        hit[f"engine_{i}"] = click_rect
        tab_x += 70
    y += 28

    # ── Level meter ──
    screen.blit(font.render("Level", True, TEXT), (20, y))
    lmx, lmw, lmh = 80, WIN_W - 100, 12
    pygame.draw.rect(screen, SLIDER_BG, (lmx, y + 2, lmw, lmh), border_radius=3)
    level = min(player.peak_level, 1.0)
    fw = int(level * lmw)
    if fw > 0:
        c = GREEN if level < 0.8 else RED
        pygame.draw.rect(screen, c, (lmx, y + 2, fw, lmh), border_radius=3)
    y += 22

    # ── FX tab: custom slot-based UI ──
    if player.voice_idx == 3:
        y = _draw_fx_panel(screen, fonts, player, mouse_pos, hit, y)
        y += 4
        # Per-engine send status
        for i, ename in enumerate(["808", "KICK", "SYNTH", "SAMPLER"]):
            send_on = player.fx_sends[i]
            send_amt = player.fx_send_amounts[i]
            sc = GREEN if send_on else DIM
            pct = f"{int(send_amt*100)}%" if send_on else "OFF"
            screen.blit(font.render(f"{ename}: {pct}", True, sc), (20 + i * 120, y))
        y += 20

    # ── CHORD tab: custom UI ──
    elif player.voice_idx == 4:
        y = _draw_chord_panel(screen, fonts, player, mouse_pos, hit, y)

    elif player.voice_idx == SAMPLER_IDX:
        y = _draw_sampler_panel(screen, fonts, player, mouse_pos, hit, y)

    # ── Normal engine UI (808 / KICK / SYNTH) ──
    elif player.voice_idx < 3:
        # Note info
        note_held = bool(held_notes) or held_chord_key
        if held_chord_notes and held_chord_key:
            names = [NOTE_NAMES[n % 12] for n in held_chord_notes]
            note_str = f"Chord: {' '.join(names)}"
        elif held_notes:
            last_note = list(held_notes.values())[-1]
            nn = NOTE_NAMES[last_note % 12]
            no = (last_note // 12) - 2
            note_str = f"Note: {nn}{no} (MIDI {last_note})"
        else:
            note_str = "Note: ---"

        screen.blit(font_bold.render(note_str, True, MAGENTA), (20, y))
        if note_held and player.voice.is_active:
            screen.blit(font_bold.render("HELD", True, GREEN), (360, y))
        oct_surf = font.render(f"Octave: C{player.octave}", True, TEXT)
        screen.blit(oct_surf, (WIN_W - oct_surf.get_width() - 20, y))
        y += 24

        # Page tabs
        tx = 20
        for i, page in enumerate(player.pages):
            active = i == player.current_page
            label = f"[{i+1}] {page['name']}"
            s = font_bold.render(label, True, TEXT)
            rect = s.get_rect(topleft=(tx, y))
            click_rect = rect.inflate(8, 8)
            hover = click_rect.collidepoint(mouse_pos)
            c = CYAN if active else (GREEN if hover else DIM)
            s = font_bold.render(label, True, c)
            if hover and not active:
                pygame.draw.rect(screen, (35, 35, 60), click_rect, border_radius=4)
            screen.blit(s, rect)
            hit[f"page_{i}"] = click_rect
            tx += rect.width + 24
        y += 24

        # Parameters
        page = player.pages[player.current_page]
        bx_p, bw_p = 15, WIN_W - 30
        bh = 4 * 36 + 12
        pygame.draw.rect(screen, PANEL, (bx_p, y, bw_p, bh), border_radius=6)
        pygame.draw.rect(screen, DIM, (bx_p, y, bw_p, bh), 1, border_radius=6)
        py = y + 6

        for pi, param_idx in enumerate(page["params"]):
            p = player.voice.params[param_idx]
            selected = pi == player.selected_param
            row_y = py + pi * 36
            row_rect = pygame.Rect(bx_p + 4, row_y - 2, bw_p - 8, 32)
            hit[f"param_row_{pi}"] = row_rect
            row_hover = row_rect.collidepoint(mouse_pos)
            if selected:
                pygame.draw.rect(screen, (40, 40, 70), row_rect, border_radius=4)
            elif row_hover:
                pygame.draw.rect(screen, (32, 32, 55), row_rect, border_radius=4)
            color = YELLOW if selected else TEXT
            prefix = "\u25b6 " if selected else "  "
            val_str = format_param_value(player, param_idx, p)
            screen.blit(font_bold.render(f"{prefix}{p.label}", True, color), (bx_p + 12, row_y + 4))
            screen.blit(font.render(val_str, True, color), (bx_p + 150, row_y + 4))
            sx = bx_p + 260
            sw = bw_p - 280
            sh = 10
            sy = row_y + 10
            slider_rect = pygame.Rect(sx, sy - 6, sw, sh + 12)
            hit[f"slider_{pi}"] = slider_rect
            pygame.draw.rect(screen, SLIDER_BG, (sx, sy, sw, sh), border_radius=4)
            fill = max(0, min(int(p.value * sw), sw))
            if fill > 0:
                bc = GREEN if selected else (60, 130, 100)
                pygame.draw.rect(screen, bc, (sx, sy, fill, sh), border_radius=4)
            knob_r = 8 if (selected and slider_rect.collidepoint(mouse_pos)) else 6
            pygame.draw.circle(screen, color, (sx + fill, sy + sh // 2), knob_r)

        y += bh + 8

        # Waveform
        _draw_waveform(screen, player, WIN_W - WAVE_W - 15, y, WAVE_W, WAVE_H)

        # Voice state (left of waveform)
        state = player.voice.get_state()
        active_str = "ACTIVE" if state["active"] else "idle"
        sc = GREEN if state["active"] else DIM
        if player.voice_idx == 0:
            stage = state.get("amp_env_stage", "?")
            freq = f"{state['current_pitch_hz']:.1f}Hz"
            mode = state.get("trigger_mode", "classic").upper()
            info = f"{active_str}  {stage}  {freq}  {mode}"
        elif player.voice_idx == 1:
            knock = f"Knock:{state.get('knock_env_stage', '?')}"
            body = f"Body:{state.get('body_env_stage', '?')}"
            paired = "PAIR" if state.get('paired', False) else "SOLO"
            info = f"{active_str}  {knock}  {body}  {paired}"
        else:
            wa = state.get('wave_a', 'SAW')
            wb = state.get('wave_b', 'SAW')
            fm = state.get('filter_mode', 'LP')
            av = state.get('active_voices', 0)
            mv = state.get('max_voices', 8)
            info = f"{active_str}  {wa}+{wb} {fm}  {av}/{mv}v"
        screen.blit(font.render(info, True, sc), (20, y))
        y += WAVE_H + 6

        # FX send controls (per-engine)
        y = _draw_send_controls(screen, fonts, player, hit, y)

        # Chord info (synth only)
        if player.voice_idx == 2:
            pname = PROGRESSION_LIST[player.chord_prog_idx]
            step = player.chord_step
            plen = progression_length(pname)
            root = player.octave * 12 + 24
            _, label = build_progression_chord(root, pname, step)
            ci = f"Chord: {label}   Prog: {pname} [{step+1}/{plen}]"
            screen.blit(font_bold.render(ci, True, MAGENTA), (20, y))
            btn_x = WIN_W - 240
            for btn_label, btn_key in [("<", "chord_prev"), (">", "chord_next"), ("PROG", "chord_cycle")]:
                bs = font_bold.render(f"[{btn_label}]", True, MAGENTA)
                br = bs.get_rect(topleft=(btn_x, y))
                bc = br.inflate(8, 8)
                bh_over = bc.collidepoint(mouse_pos)
                if bh_over:
                    bs = font_bold.render(f"[{btn_label}]", True, CYAN)
                    pygame.draw.rect(screen, (35, 35, 60), bc, border_radius=4)
                screen.blit(bs, br)
                hit[btn_key] = bc
                btn_x += br.width + 14
            y += 22

    # ── Global MIDI keyboard (always visible) ──
    y = max(y, WIN_H - MIDI_KB_H - METRO_STRIP_H - 50)
    y = _draw_midi_keyboard(screen, fonts, player, hit, y)

    # ── Global metronome strip (always visible) ──
    _draw_metronome_strip(screen, fonts, player, hit)

    # ── Help line ──
    help_str = "TAB voice  SPACE trigger  A-K chromatic  Z/X oct  C chord  Q quit"
    screen.blit(font.render(help_str, True, DIM), (20, WIN_H - 18))

    pygame.display.flip()
    return hit


# ── MIDI handling ─────────────────────────────────────────────────────

def make_midi_callback(player, monitor=False):
    def callback(msg):
        if monitor:
            print(f"  MIDI: {msg}")
        if isinstance(msg, NoteOn):
            if msg.channel == PAD_CHANNEL and msg.note in PAD_ENGINE_MAP:
                player.select_engine(PAD_ENGINE_MAP[msg.note])
            else:
                player._midi_note_voice[msg.note] = player.voice_idx
                player.trigger(msg.note, msg.velocity_float)
        elif isinstance(msg, NoteOff):
            if msg.channel == PAD_CHANNEL and msg.note in PAD_ENGINE_MAP:
                pass
            else:
                vidx = player._midi_note_voice.pop(msg.note, player.voice_idx)
                with player.lock:
                    player.voices[min(vidx, len(player.voices) - 1)].release(msg.note)
                player.active_midi_notes.discard(msg.note)
        elif isinstance(msg, ControlChange):
            if msg.cc in KNOB_CCS:
                ki = KNOB_CCS.index(msg.cc)
                if ki < 4 and player.voice_idx < 3:
                    page = player.pages[player.current_page]
                    if ki < len(page["params"]):
                        pidx = page["params"][ki]
                        with player.lock:
                            player.voice.params[pidx].value = msg.value_float
    return callback


def open_midi(player, device_name=None, monitor=False):
    if not midi_input.available():
        print("MIDI: mido not installed (pip install mido python-rtmidi)")
        return None
    try:
        mi = midi_input.MidiInput(make_midi_callback(player, monitor=monitor))
        name = mi.open(device_name)
        print(f"MIDI: connected to '{name}'")
        return mi
    except RuntimeError as e:
        print(f"MIDI: {e}")
        return None


# ── Headless mode ────────────────────────────────────────────────────

def run_headless(player, midi_in):
    print(f"\nCYPHER headless — {SR}Hz / {BLOCK_SIZE} frames")
    if midi_in:
        print(f"MIDI: {midi_in.device_name}")
    print("Ctrl+C to quit\n")
    try:
        while True:
            state = player.voice.get_state()
            active = "\u25cf ACTIVE" if state["active"] else "\u25cb idle"
            level = f"{player.peak_level:.2f}"
            print(f"\r  [{player.voice_name}] {active}  Level: {level}   ", end="", flush=True)
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nShutting down...")


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CYPHER interactive tweaker")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--midi", action="store_true")
    p.add_argument("--midi-device", type=str, default=None)
    p.add_argument("--list-midi", action="store_true")
    p.add_argument("--midi-monitor", action="store_true")
    return p.parse_args()


# ── Main loop ─────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.list_midi:
        devices = midi_input.list_inputs() if midi_input.available() else []
        if devices:
            print("MIDI input devices:")
            for i, name in enumerate(devices):
                print(f"  [{i}] {name}")
        else:
            print("No MIDI input devices found.")
        return

    player = Player()
    player.start()
    midi_in = None

    use_midi = args.midi or args.midi_device or args.headless or args.midi_monitor
    if use_midi:
        midi_in = open_midi(player, device_name=args.midi_device, monitor=args.midi_monitor)

    if args.headless:
        try:
            run_headless(player, midi_in)
        finally:
            if midi_in:
                midi_in.close()
            player.stop()
        return

    # ── Pygame UI mode ──
    pygame.display.init()
    pygame.font.init()
    pygame.key.set_repeat(400, 50)

    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.SCALED)
    pygame.display.set_caption("CYPHER")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("monospace", 15)
    font_bold = pygame.font.SysFont("monospace", 15, bold=True)
    font_lg = pygame.font.SysFont("monospace", 20, bold=True)
    font_title = pygame.font.SysFont("monospace", 24, bold=True)
    fonts = (font, font_bold, font_lg, font_title)

    held_notes: dict[int, int] = {}
    held_chord_key = False
    held_chord_notes: list[int] = []
    hit: dict[str, pygame.Rect] = {}
    dragging_slider: str | None = None  # "slider_N", "fx_slider_N", "send_slider", "swing_slider"

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    key = event.key

                    if key == pygame.K_q or key == pygame.K_ESCAPE:
                        running = False

                    elif key == pygame.K_TAB:
                        for k, note in list(held_notes.items()):
                            player.release_note(note)
                        held_notes.clear()
                        if held_chord_key:
                            player.release_chord(held_chord_notes)
                            held_chord_key = False
                            held_chord_notes = []
                        player.stop_arpeggio()
                        player.switch_voice()

                    # Chord trigger (synth or chord tab)
                    elif key == pygame.K_c and player.voice_idx in (2, 4) and not held_chord_key:
                        root = player.octave * 12 + 24
                        pname = PROGRESSION_LIST[player.chord_prog_idx]
                        notes, label = build_progression_chord(root, pname, player.chord_step)
                        player.chord_label = label

                        if player.chord_mode == 0:  # CHORD (humanized)
                            player.trigger_chord_humanized(notes)
                        elif player.chord_mode == 1:  # STRUM DOWN
                            player.trigger_strum(notes, direction="down")
                        elif player.chord_mode == 2:  # STRUM UP
                            player.trigger_strum(notes, direction="up")
                        elif player.chord_mode == 3:  # ARPEGGIO
                            player.start_arpeggio(notes)

                        held_chord_key = True
                        held_chord_notes = notes

                    elif player.voice_idx == SAMPLER_IDX and key in CHROMATIC_KEYS \
                         and key not in held_notes:
                        # A-K chromatic → pads 0..12 with octave shift (Z/X).
                        # Octave 2 is the base (A = pad 0); Octave 3 shifts
                        # by +12 to reach pads 12-15. Anything past 15 clamps.
                        semi = CHROMATIC_KEYS[key]
                        pad_idx = semi + (player.octave - 2) * 12
                        if 0 <= pad_idx < 16:
                            midi_note = PAD_MIDI_START + pad_idx
                            with player.lock:
                                player.sampler.trigger_pad(pad_idx, midi_note, 0.9)
                            if player.sampler.slots[pad_idx].loaded:
                                player.sampler.focus_slot(pad_idx)
                            player.active_midi_notes.add(midi_note)
                            held_notes[key] = midi_note

                    elif key == pygame.K_SPACE and key not in held_notes:
                        midi_note = player.octave * 12 + 24
                        if player.voice_idx < 3:
                            player.trigger(midi_note)
                            held_notes[key] = midi_note

                    elif key in CHROMATIC_KEYS and key not in held_notes:
                        semitone = CHROMATIC_KEYS[key]
                        midi_note = player.octave * 12 + 24 + semitone
                        if player.voice_idx < 3:
                            player.trigger(midi_note)
                            held_notes[key] = midi_note

                    elif key == pygame.K_UP:
                        if player.voice_idx == 3:
                            player.fx_selected_param = max(0, player.fx_selected_param - 1)
                        elif player.voice_idx == 4:
                            player.chord_prog_idx = (player.chord_prog_idx - 1) % len(PROGRESSION_LIST)
                            player.chord_step = 0
                        else:
                            player.selected_param = max(0, player.selected_param - 1)
                    elif key == pygame.K_DOWN:
                        if player.voice_idx == 3:
                            max_p = len(player.fx_reverb_params) - 1
                            player.fx_selected_param = min(max_p, player.fx_selected_param + 1)
                        elif player.voice_idx == 4:
                            player.chord_prog_idx = (player.chord_prog_idx + 1) % len(PROGRESSION_LIST)
                            player.chord_step = 0
                        else:
                            player.selected_param = min(3, player.selected_param + 1)
                    elif key == pygame.K_RIGHT:
                        if player.voice_idx == 4:
                            pname = PROGRESSION_LIST[player.chord_prog_idx]
                            player.chord_step = (player.chord_step + 1) % progression_length(pname)
                        else:
                            player.adjust_param(0.05)
                    elif key == pygame.K_LEFT:
                        if player.voice_idx == 4:
                            pname = PROGRESSION_LIST[player.chord_prog_idx]
                            player.chord_step = (player.chord_step - 1) % progression_length(pname)
                        else:
                            player.adjust_param(-0.05)
                    elif key == pygame.K_LEFTBRACKET:
                        player.adjust_param(-0.01)
                    elif key == pygame.K_RIGHTBRACKET:
                        player.adjust_param(0.01)

                    elif key == pygame.K_1 and player.voice_idx < 3:
                        player.current_page = 0
                        player.selected_param = min(player.selected_param, 3)
                    elif key == pygame.K_2 and player.voice_idx < 3 and len(player.pages) > 1:
                        player.current_page = 1
                        player.selected_param = min(player.selected_param, 3)
                    elif key == pygame.K_3 and player.voice_idx < 3 and len(player.pages) > 2:
                        player.current_page = 2
                        player.selected_param = min(player.selected_param, 3)
                    elif key == pygame.K_4 and player.voice_idx < 3 and len(player.pages) > 3:
                        player.current_page = 3
                        player.selected_param = min(player.selected_param, 3)

                    elif key == pygame.K_z:
                        player.octave = max(-1, player.octave - 1)
                    elif key == pygame.K_x:
                        player.octave = min(6, player.octave + 1)
                    elif key == pygame.K_r:
                        player.reset_params()
                    elif key == pygame.K_p:
                        if player.kick._paired:
                            player.kick.unpair_808()
                        else:
                            freq = (player.sub808._current_pitch_hz if player.sub808.is_active else 32.7)
                            player.kick.pair_808(freq)
                    elif key == pygame.K_m:
                        m = player.sub808._trigger_mode
                        player.sub808._trigger_mode = "oneshot" if m == "classic" else "classic"
                    elif key == pygame.K_v:
                        player.reverb_on = not player.reverb_on
                        if not player.reverb_on:
                            player.reverb.clear()

                    elif key == pygame.K_PERIOD:
                        pname = PROGRESSION_LIST[player.chord_prog_idx]
                        player.chord_step = (player.chord_step + 1) % progression_length(pname)
                    elif key == pygame.K_COMMA:
                        pname = PROGRESSION_LIST[player.chord_prog_idx]
                        player.chord_step = (player.chord_step - 1) % progression_length(pname)
                    elif key == pygame.K_n:
                        player.chord_prog_idx = (player.chord_prog_idx + 1) % len(PROGRESSION_LIST)
                        player.chord_step = 0

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos

                    # Metronome strip (always-on, any tab)
                    mods = pygame.key.get_mods()
                    nudge = 0.1 if (mods & pygame.KMOD_SHIFT) else 1.0
                    r = hit.get("metro_play")
                    if r and r.collidepoint(mx, my):
                        player.metronome.toggle()
                    r = hit.get("metro_bpm_minus")
                    if r and r.collidepoint(mx, my):
                        player.project.bpm = max(20.0, player.project.bpm - nudge)
                        player.metronome.reset_phase()
                    r = hit.get("metro_bpm_plus")
                    if r and r.collidepoint(mx, my):
                        player.project.bpm = min(300.0, player.project.bpm + nudge)
                        player.metronome.reset_phase()

                    # Engine tabs
                    for i in range(len(VOICE_NAMES)):
                        r = hit.get(f"engine_{i}")
                        if r and r.collidepoint(mx, my):
                            for k, note in list(held_notes.items()):
                                player.release_note(note)
                            held_notes.clear()
                            if held_chord_key:
                                player.release_chord(held_chord_notes)
                                held_chord_key = False
                                held_chord_notes = []
                            player.stop_arpeggio()
                            player.select_engine(i)
                            break

                    # Page tabs (for normal engines)
                    if player.voice_idx < 3:
                        for i in range(len(player.pages)):
                            r = hit.get(f"page_{i}")
                            if r and r.collidepoint(mx, my):
                                player.current_page = i
                                player.selected_param = min(player.selected_param, 3)
                                break

                    # Reverb toggle
                    r = hit.get("reverb")
                    if r and r.collidepoint(mx, my):
                        player.reverb_on = not player.reverb_on
                        if not player.reverb_on:
                            player.reverb.clear()

                    # SAMPLER tab clicks
                    if player.voice_idx == SAMPLER_IDX:
                        r = hit.get("sampler_load_folder")
                        if r and r.collidepoint(mx, my):
                            player.sampler_open_folder()

                        for fi in range(len(player.sampler_files)):
                            r = hit.get(f"sampler_file_{fi}")
                            if r and r.collidepoint(mx, my):
                                player.sampler_selected_file = fi
                                break

                        for pi in range(16):
                            r = hit.get(f"sampler_pad_{pi}")
                            if r and r.collidepoint(mx, my):
                                player.sampler.focus_slot(pi)
                                if player.sampler.slots[pi].loaded:
                                    with player.lock:
                                        player.sampler.trigger_pad(
                                            pi, PAD_MIDI_START + pi, 0.9)
                                break

                        r = hit.get("sampler_load_to_pad")
                        if r and r.collidepoint(mx, my):
                            player.sampler_load_selected_to(
                                player.sampler.focused_slot_idx)

                        r = hit.get("sampler_clear_pad")
                        if r and r.collidepoint(mx, my):
                            with player.lock:
                                player.sampler.clear_slot(
                                    player.sampler.focused_slot_idx)

                        # MODE radios, REVERSE, FREEZE, GROSS BEAT
                        slot_f = player.sampler.slots[player.sampler.focused_slot_idx]
                        from cypher.sampler.sampler import (
                            MODE_NAMES as _SMN, P_MODE as _PM, P_REVERSE as _PR,
                        )
                        from cypher.sampler.gross_beat import PRESET_NAMES as _GBN
                        for mi in range(len(_SMN)):
                            r = hit.get(f"sampler_mode_{mi}")
                            if r and r.collidepoint(mx, my):
                                slot_f.params[_PM].value = mi / max(1, len(_SMN) - 1)
                                break

                        r = hit.get("sampler_reverse")
                        if r and r.collidepoint(mx, my):
                            cur = int(round(slot_f.params[_PR].mapped))
                            slot_f.params[_PR].value = 0.0 if cur == 1 else 1.0

                        r = hit.get("sampler_freeze_arm")
                        if r and r.collidepoint(mx, my):
                            slot_f.freeze_armed = not slot_f.freeze_armed

                        # BPM match/free toggle
                        r = hit.get("sampler_bpm_free")
                        if r and r.collidepoint(mx, my):
                            slot_f.match_mode = False
                        r = hit.get("sampler_bpm_match")
                        if r and r.collidepoint(mx, my) and slot_f.sample_bpm > 0:
                            slot_f.match_mode = True

                        # BPM nudge −/+ (invalidate cache; persist after idle).
                        # Debounced persist happens in the main loop.
                        import time as _time
                        _fsi = player.sampler.focused_slot_idx
                        r = hit.get("sampler_bpm_minus")
                        if r and r.collidepoint(mx, my) and slot_f.sample_bpm > 0:
                            slot_f.sample_bpm = max(20.0, slot_f.sample_bpm - 0.1)
                            slot_f.user_corrected = True
                            slot_f.invalidate_bpm_cache()
                            player.sampler_bpm_dirty[_fsi] = _time.monotonic()
                        r = hit.get("sampler_bpm_plus")
                        if r and r.collidepoint(mx, my) and slot_f.sample_bpm > 0:
                            slot_f.sample_bpm = min(300.0, slot_f.sample_bpm + 0.1)
                            slot_f.user_corrected = True
                            slot_f.invalidate_bpm_cache()
                            player.sampler_bpm_dirty[_fsi] = _time.monotonic()

                        # ×2 / ÷2 (halftime/doubletime correction)
                        r = hit.get("sampler_bpm_double")
                        if r and r.collidepoint(mx, my) and slot_f.sample_bpm > 0:
                            slot_f.sample_bpm = min(300.0, slot_f.sample_bpm * 2.0)
                            slot_f.user_corrected = True
                            slot_f.invalidate_bpm_cache()
                            player.sampler_persist_slot_meta(player.sampler.focused_slot_idx)
                        r = hit.get("sampler_bpm_half")
                        if r and r.collidepoint(mx, my) and slot_f.sample_bpm > 0:
                            slot_f.sample_bpm = max(20.0, slot_f.sample_bpm / 2.0)
                            slot_f.user_corrected = True
                            slot_f.invalidate_bpm_cache()
                            player.sampler_persist_slot_meta(player.sampler.focused_slot_idx)

                        # Tap tempo
                        r = hit.get("sampler_bpm_tap")
                        if r and r.collidepoint(mx, my):
                            fs_idx = player.sampler.focused_slot_idx
                            now_t = _time.monotonic()
                            hist = player.sampler_tap_history.setdefault(fs_idx, [])
                            if hist and now_t - hist[-1] > 2.5:
                                hist.clear()
                            hist.append(now_t)
                            if len(hist) > 6:
                                del hist[:-6]
                            if len(hist) >= 2:
                                intervals = [hist[i] - hist[i-1] for i in range(1, len(hist))]
                                med = sorted(intervals)[len(intervals)//2]
                                if 0.2 < med < 2.0:
                                    slot_f.sample_bpm = 60.0 / med
                                    slot_f.user_corrected = True
                                    slot_f.invalidate_bpm_cache()
                                    if not slot_f.match_mode:
                                        slot_f.match_mode = True
                                    player.sampler_persist_slot_meta(fs_idx)

                        for gi in range(len(_GBN)):
                            r = hit.get(f"sampler_gross_{gi}")
                            if r and r.collidepoint(mx, my):
                                player.sampler.gross_beat.preset = gi
                                if gi == 0:
                                    player.sampler.gross_beat.reset()
                                break

                        # Slider clicks (+ start drag)
                        from cypher.sampler.sampler import (
                            P_PITCH as _PP, P_GAIN as _PG, P_SLICES as _PSL,
                            P_FZ_POS as _PFP, P_FZ_GRAIN as _PFG,
                            P_FZ_MOTION as _PFM, P_FZ_RATE as _PFR,
                            DIVISION_VALUES as _DIV_VALS,
                            MODE_CHOP as _MC,
                        )
                        # PITCH +/- buttons (1 semitone each)
                        r = hit.get("sampler_pitch_minus")
                        if r and r.collidepoint(mx, my):
                            cur = slot_f.params[_PP].mapped
                            new_st = max(-24.0, round(cur - 1.0))
                            slot_f.params[_PP].value = (new_st + 24.0) / 48.0
                        r = hit.get("sampler_pitch_plus")
                        if r and r.collidepoint(mx, my):
                            cur = slot_f.params[_PP].mapped
                            new_st = min(24.0, round(cur + 1.0))
                            slot_f.params[_PP].value = (new_st + 24.0) / 48.0

                        for key, pidx in [
                            ("sampler_gain_slider", _PG),
                            ("sampler_slices_slider", _PSL),
                            ("sampler_fz_pos_slider", _PFP),
                            ("sampler_fz_grain_slider", _PFG),
                            ("sampler_fz_rate_slider", _PFR),
                        ]:
                            r = hit.get(key)
                            if r and r.collidepoint(mx, my):
                                t = max(0.0, min(1.0, (mx - r.x) / max(1, r.width)))
                                slot_f.params[pidx].value = t
                                dragging_slider = key
                                break

                        # FREEZE motion radios
                        from cypher.sampler.freeze import MOTION_NAMES as _FZMN
                        for mi in range(len(_FZMN)):
                            r = hit.get(f"sampler_fz_motion_{mi}")
                            if r and r.collidepoint(mx, my):
                                slot_f.params[_PFM].value = (
                                    mi / max(1, len(_FZMN) - 1))
                                break

                        # Beat division → compute SLICES
                        for di in range(len(_DIV_VALS)):
                            r = hit.get(f"sampler_div_{di}")
                            if r and r.collidepoint(mx, my):
                                sec = slot_f.length / max(1, slot_f.source_rate)
                                beats = max(1.0, sec * player.project.bpm / 60.0)
                                target = int(min(32, max(1,
                                    round(beats * _DIV_VALS[di]))))
                                # snap=32 → value = (target-1)/31
                                slot_f.params[_PSL].value = (target - 1) / 31.0
                                slot_f.refresh_slices()
                                break

                        # CHOP slice boundary drag — pick up the boundary under
                        # the cursor and start a drag session.
                        if slot_f.mode == _MC:
                            r = hit.get("sampler_chop_reset")
                            if r and r.collidepoint(mx, my):
                                with player.lock:
                                    slot_f.reset_slices()
                            else:
                                for si in range(1, len(slot_f._slice_points)):
                                    r = hit.get(f"sampler_chop_{si}")
                                    if r and r.collidepoint(mx, my):
                                        dragging_slider = f"sampler_chop_{si}"
                                        strip = hit.get("sampler_chop_strip")
                                        if strip is not None:
                                            frac = max(0.0, min(1.0,
                                                (mx - strip.x) / max(1, strip.width - 1)))
                                            with player.lock:
                                                slot_f.nudge_slice_boundary(si, frac)
                                        break

                    # Send toggle (per-engine)
                    _send_idx = tab_voice_idx(player.voice_idx)
                    r = hit.get("send_toggle")
                    if r and r.collidepoint(mx, my) and _send_idx is not None:
                        player.fx_sends[_send_idx] = not player.fx_sends[_send_idx]

                    # Send slider drag
                    r = hit.get("send_slider")
                    if r and r.collidepoint(mx, my) and _send_idx is not None:
                        t = max(0.0, min(1.0, (mx - r.x) / r.width))
                        player.fx_send_amounts[_send_idx] = t
                        dragging_slider = "send_slider"

                    # Key selector
                    r = hit.get("key_left")
                    if r and r.collidepoint(mx, my):
                        player.global_key = (player.global_key - 1) % 12
                    r = hit.get("key_right")
                    if r and r.collidepoint(mx, my):
                        player.global_key = (player.global_key + 1) % 12

                    # Scale selector
                    r = hit.get("scale_left")
                    if r and r.collidepoint(mx, my):
                        player.global_scale_idx = (player.global_scale_idx - 1) % len(SCALE_NAMES)
                    r = hit.get("scale_right")
                    if r and r.collidepoint(mx, my):
                        player.global_scale_idx = (player.global_scale_idx + 1) % len(SCALE_NAMES)

                    # MIDI keyboard clicks
                    for midi_n in range(128):
                        r = hit.get(f"kb_{midi_n}")
                        if r and r.collidepoint(mx, my):
                            if player.voice_idx < 3:
                                player.trigger(midi_n)
                                # Will release on mouse up
                                held_notes[f"mouse_{midi_n}"] = midi_n
                            break

                    # ── FX-specific controls ──
                    if player.voice_idx == 3:
                        r = hit.get("fx_reverb_header")
                        if r and r.collidepoint(mx, my):
                            player.fx_reverb_expanded = not player.fx_reverb_expanded

                        r = hit.get("fx_reverb_toggle")
                        if r and r.collidepoint(mx, my):
                            player.reverb_on = not player.reverb_on
                            if not player.reverb_on:
                                player.reverb.clear()

                        for mi in range(4):
                            r = hit.get(f"fx_mode_{mi}")
                            if r and r.collidepoint(mx, my):
                                player.fx_reverb_mode = mi
                                player.reverb.set_mode(mi)

                        fx_slider_hit = False
                        for pi in range(len(player.fx_reverb_params)):
                            r = hit.get(f"fx_slider_{pi}")
                            if r and r.collidepoint(mx, my):
                                player.fx_selected_param = pi
                                t = max(0.0, min(1.0, (mx - r.x) / r.width))
                                player.fx_reverb_params[pi].value = t
                                player._sync_reverb_from_params()
                                dragging_slider = f"fx_slider_{pi}"
                                fx_slider_hit = True
                                break

                        if not fx_slider_hit:
                            for pi in range(len(player.fx_reverb_params)):
                                r = hit.get(f"fx_param_row_{pi}")
                                if r and r.collidepoint(mx, my):
                                    player.fx_selected_param = pi
                                    break

                    # ── CHORD-specific controls ──
                    elif player.voice_idx == 4:
                        # Progression selection
                        for pi in range(len(PROGRESSION_LIST)):
                            r = hit.get(f"prog_{pi}")
                            if r and r.collidepoint(mx, my):
                                player.chord_prog_idx = pi
                                player.chord_step = 0
                                break

                        # Step selection
                        pname = PROGRESSION_LIST[player.chord_prog_idx]
                        for si in range(progression_length(pname)):
                            r = hit.get(f"chord_step_{si}")
                            if r and r.collidepoint(mx, my):
                                player.chord_step = si
                                break

                        # Mode selection
                        for mi in range(4):
                            r = hit.get(f"chord_mode_{mi}")
                            if r and r.collidepoint(mx, my):
                                player.chord_mode = mi
                                if mi != 3:
                                    player.stop_arpeggio()
                                break

                        # Rate selection
                        for ri in range(len(RHYTHM_NAMES)):
                            r = hit.get(f"chord_rate_{ri}")
                            if r and r.collidepoint(mx, my):
                                player.chord_rhythm_idx = ri
                                break

                        # BPM
                        r = hit.get("bpm_left")
                        if r and r.collidepoint(mx, my):
                            player.chord_bpm = max(40, player.chord_bpm - 5)
                        r = hit.get("bpm_right")
                        if r and r.collidepoint(mx, my):
                            player.chord_bpm = min(300, player.chord_bpm + 5)

                        # Swing slider
                        r = hit.get("swing_slider")
                        if r and r.collidepoint(mx, my):
                            t = max(0.0, min(1.0, (mx - r.x) / r.width))
                            player.chord_swing = t
                            dragging_slider = "swing_slider"

                    # ── Normal engine slider/param handling ──
                    elif player.voice_idx < 3:
                        slider_hit = False
                        for pi in range(4):
                            r = hit.get(f"slider_{pi}")
                            if r and r.collidepoint(mx, my):
                                player.selected_param = pi
                                t = max(0.0, min(1.0, (mx - r.x) / r.width))
                                page = player.pages[player.current_page]
                                with player.lock:
                                    player.voice.params[page["params"][pi]].value = t
                                dragging_slider = f"slider_{pi}"
                                slider_hit = True
                                break

                        if not slider_hit:
                            for pi in range(4):
                                r = hit.get(f"param_row_{pi}")
                                if r and r.collidepoint(mx, my):
                                    player.selected_param = pi
                                    break

                    # Chord controls (synth tab)
                    if player.voice_idx == 2:
                        r = hit.get("chord_prev")
                        if r and r.collidepoint(mx, my):
                            pname = PROGRESSION_LIST[player.chord_prog_idx]
                            player.chord_step = (player.chord_step - 1) % progression_length(pname)
                        r = hit.get("chord_next")
                        if r and r.collidepoint(mx, my):
                            pname = PROGRESSION_LIST[player.chord_prog_idx]
                            player.chord_step = (player.chord_step + 1) % progression_length(pname)
                        r = hit.get("chord_cycle")
                        if r and r.collidepoint(mx, my):
                            player.chord_prog_idx = (player.chord_prog_idx + 1) % len(PROGRESSION_LIST)
                            player.chord_step = 0

                elif event.type == pygame.MOUSEMOTION and dragging_slider is not None:
                    mx_pos = event.pos[0]

                    if dragging_slider.startswith("fx_slider_"):
                        pi = int(dragging_slider.split("_")[-1])
                        r = hit.get(f"fx_slider_{pi}")
                        if r:
                            t = max(0.0, min(1.0, (mx_pos - r.x) / r.width))
                            player.fx_reverb_params[pi].value = t
                            player._sync_reverb_from_params()
                    elif dragging_slider.startswith("slider_"):
                        pi = int(dragging_slider.split("_")[-1])
                        r = hit.get(f"slider_{pi}")
                        if r:
                            t = max(0.0, min(1.0, (mx_pos - r.x) / r.width))
                            page = player.pages[player.current_page]
                            with player.lock:
                                player.voice.params[page["params"][pi]].value = t
                    elif dragging_slider == "send_slider":
                        r = hit.get("send_slider")
                        _si = tab_voice_idx(player.voice_idx)
                        if r and _si is not None:
                            t = max(0.0, min(1.0, (mx_pos - r.x) / r.width))
                            player.fx_send_amounts[_si] = t
                    elif dragging_slider == "swing_slider":
                        r = hit.get("swing_slider")
                        if r:
                            t = max(0.0, min(1.0, (mx_pos - r.x) / r.width))
                            player.chord_swing = t
                    elif dragging_slider.startswith("sampler_chop_") and \
                         dragging_slider.split("_")[-1].isdigit():
                        si = int(dragging_slider.split("_")[-1])
                        strip = hit.get("sampler_chop_strip")
                        slot_d = player.sampler.slots[
                            player.sampler.focused_slot_idx]
                        if strip is not None:
                            frac = max(0.0, min(1.0,
                                (mx_pos - strip.x) / max(1, strip.width - 1)))
                            with player.lock:
                                slot_d.nudge_slice_boundary(si, frac)
                    elif dragging_slider in (
                        "sampler_pitch_slider", "sampler_gain_slider",
                        "sampler_slices_slider", "sampler_fz_pos_slider",
                        "sampler_fz_grain_slider", "sampler_fz_rate_slider",
                    ):
                        r = hit.get(dragging_slider)
                        if r:
                            from cypher.sampler.sampler import (
                                P_PITCH, P_GAIN, P_SLICES,
                                P_FZ_POS, P_FZ_GRAIN, P_FZ_RATE,
                            )
                            idx_map = {
                                "sampler_pitch_slider": P_PITCH,
                                "sampler_gain_slider": P_GAIN,
                                "sampler_slices_slider": P_SLICES,
                                "sampler_fz_pos_slider": P_FZ_POS,
                                "sampler_fz_grain_slider": P_FZ_GRAIN,
                                "sampler_fz_rate_slider": P_FZ_RATE,
                            }
                            t = max(0.0, min(1.0, (mx_pos - r.x) / max(1, r.width)))
                            slot_d = player.sampler.slots[
                                player.sampler.focused_slot_idx]
                            slot_d.params[idx_map[dragging_slider]].value = t

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging_slider = None
                    # Release mouse-triggered notes
                    mouse_keys = [k for k in held_notes if isinstance(k, str) and k.startswith("mouse_")]
                    for mk in mouse_keys:
                        player.release_note(held_notes.pop(mk))

                elif event.type == pygame.KEYUP:
                    key = event.key
                    if key in held_notes:
                        player.release_note(held_notes.pop(key))
                    elif key == pygame.K_c and held_chord_key:
                        player.stop_all_chord_playback()
                        held_chord_key = False
                        held_chord_notes = []

            hit = draw(screen, fonts, player, held_notes, held_chord_key, held_chord_notes)
            player.peak_level *= 0.97

            # Debounced BPM sidecar persist (150ms idle)
            if player.sampler_bpm_dirty:
                import time as _tm
                _now = _tm.monotonic()
                _done = [si for si, ts in player.sampler_bpm_dirty.items()
                         if _now - ts >= 0.15]
                for _si in _done:
                    player.sampler_persist_slot_meta(_si)
                    del player.sampler_bpm_dirty[_si]

            clock.tick(FPS)

    finally:
        if midi_in:
            midi_in.close()
        player.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
