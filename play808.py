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
WIN_W, WIN_H = 800, 780
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
    "FX": [],     # FX uses custom slot-based UI
    "CHORD": [],  # CHORD uses custom UI
}

VOICE_NAMES = ["808", "KICK", "SYNTH", "FX", "CHORD"]

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
        self.sub808 = Sub808Voice(SR)
        self.kick = KickVoice(SR)
        self.synth = PolySynthVoice(SR)
        self.voices = [self.sub808, self.kick, self.synth]
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
        self.fx_sends = [False, False, True]  # 808, KICK, SYNTH send on/off
        self.fx_send_amounts = [0.5, 0.5, 0.5]  # send level 0-1 per engine
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
        if self.voice_idx >= len(self.voices):
            return self.voices[2]  # fallback to synth for FX/CHORD tabs
        return self.voices[self.voice_idx]

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
            with self.lock:
                self.synth.release_all()
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
            with self.lock:
                self.synth.release_all()
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
                if last_note >= 0:
                    with self.lock:
                        self.synth.release(last_note)
                    self.active_midi_notes.discard(last_note)
                with self.lock:
                    self.synth.trigger(note, velocity)
                self.active_midi_notes.add(note)
                last_note = note
                # Apply swing
                actual_dur = note_duration
                if idx % 2 == 1 and self.chord_swing > 0:
                    actual_dur *= (1.0 + self.chord_swing * 0.5)
                self._arp_stop.wait(actual_dur)
                idx += 1
            # Release last note
            if last_note >= 0:
                with self.lock:
                    self.synth.release(last_note)
                self.active_midi_notes.discard(last_note)
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
            tmp = Sub808Voice(SR)
            for i, p in enumerate(self.sub808.params):
                tmp.params[i].value = p.value
            tmp.trigger(36, 0.9)
            full = tmp.process(6000)
            self.preview_buf = full
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

    engine_idx = player.voice_idx
    if engine_idx >= len(player.fx_sends):
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
        for i, ename in enumerate(["808", "KICK", "SYNTH"]):
            send_on = player.fx_sends[i]
            send_amt = player.fx_send_amounts[i]
            sc = GREEN if send_on else DIM
            pct = f"{int(send_amt*100)}%" if send_on else "OFF"
            screen.blit(font.render(f"{ename}: {pct}", True, sc), (20 + i * 120, y))
        y += 20

    # ── CHORD tab: custom UI ──
    elif player.voice_idx == 4:
        y = _draw_chord_panel(screen, fonts, player, mouse_pos, hit, y)

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
    y = max(y, WIN_H - MIDI_KB_H - 50)
    y = _draw_midi_keyboard(screen, fonts, player, hit, y)

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

                    # Send toggle (per-engine)
                    r = hit.get("send_toggle")
                    if r and r.collidepoint(mx, my) and player.voice_idx < len(player.fx_sends):
                        player.fx_sends[player.voice_idx] = not player.fx_sends[player.voice_idx]

                    # Send slider drag
                    r = hit.get("send_slider")
                    if r and r.collidepoint(mx, my) and player.voice_idx < len(player.fx_send_amounts):
                        t = max(0.0, min(1.0, (mx - r.x) / r.width))
                        player.fx_send_amounts[player.voice_idx] = t
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
                        if r and player.voice_idx < len(player.fx_send_amounts):
                            t = max(0.0, min(1.0, (mx_pos - r.x) / r.width))
                            player.fx_send_amounts[player.voice_idx] = t
                    elif dragging_slider == "swing_slider":
                        r = hit.get("swing_slider")
                        if r:
                            t = max(0.0, min(1.0, (mx_pos - r.x) / r.width))
                            player.chord_swing = t

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
            clock.tick(FPS)

    finally:
        if midi_in:
            midi_in.close()
        player.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
