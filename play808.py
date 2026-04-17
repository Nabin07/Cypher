#!/usr/bin/env python3
"""Interactive CYPHER tweaker — pygame UI with proper key hold/release.

Controls:
  TAB          Switch voice (808 / KICK / SYNTH)
  SPACE        Trigger root note (hold to sustain)
  Z/X          Octave down/up
  A-K keys     Chromatic notes (hold to sustain, polyphonic on synth)
  UP/DOWN      Select parameter
  LEFT/RIGHT   Adjust parameter
  [/]          Fine-tune parameter
  1/2/3        Switch parameter page
  P            Pair kick with 808 / Unpair
  M            Toggle trigger mode (Classic / One-shot)
  R            Reset all params to defaults
  C            Trigger chord (synth only — hold to sustain)
  ,/.          Step through chord progression
  N            Cycle progression
  V            Toggle SEND to FX for current engine
  Q / ESC      Quit
"""

import argparse
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
    PROGRESSION_LIST,
)
from cypher.core.types import DEFAULT_SAMPLE_RATE
from cypher.fx.fx import FXEngine
from cypher.synth.chord_engine import ChordEngine, MODE_NAMES as CHORD_MODE_NAMES
from cypher.synth.chords import SCALE_LIST, is_note_in_scale
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
SLIDER_BG = (40, 40, 65)

# --- Window ---
WIN_W, WIN_H = 800, 640
WAVE_H = 130          # waveform display height
WAVE_W = 250          # waveform display width (~1/3 screen)
PREVIEW_SAMPLES = 4096  # samples to render for preview
FPS = 60

# Chromatic keyboard: A=C, W=C#, S=D, E=D#, D=E, F=F, T=F#, G=G, Y=G#, H=A, U=A#, J=B, K=C+1
CHROMATIC_KEYS = {
    pygame.K_a: 0,  pygame.K_w: 1,  pygame.K_s: 2,  pygame.K_e: 3,
    pygame.K_d: 4,  pygame.K_f: 5,  pygame.K_t: 6,  pygame.K_g: 7,
    pygame.K_y: 8,  pygame.K_h: 9,  pygame.K_u: 10, pygame.K_j: 11,
    pygame.K_k: 12,
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

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
    "FX": [
        {"name": "VERB", "params": [0, 1, 2, 3]},
        {"name": "TONE", "params": [4, 5, 6, 7]},
    ],
    "CHORD": [
        {"name": "KEY", "params": [0, 1, 2, 3]},
        {"name": "PLAY", "params": [4, 5, 6, 7]},
    ],
}

VOICE_NAMES = ["808", "KICK", "SYNTH", "FX", "CHORD"]
FX_IDX = 3    # index of FX in voices list
CHORD_IDX = 4 # index of CHORD engine in voices list

# ── MIDI pad mapping (KeyLab 61 Essential defaults) ──────────────────
# Pads typically send notes 36-43 on MIDI channel 10 (index 9).
# Run with --midi-monitor to verify your controller's actual values.
PAD_CHANNEL = 9       # MIDI Ch10
PAD_ENGINE_MAP = {    # pad note -> engine index
    36: 0,            # Pad 1 -> 808
    37: 1,            # Pad 2 -> KICK
    38: 2,            # Pad 3 -> SYNTH
}
# Pads 4-8 (notes 39-43): reserved for future use (chord trigger, etc.)

# Knob CC defaults (KeyLab 61 Essential "User" preset).
# First 4 knobs map to current page params. Verify with --midi-monitor.
KNOB_CCS = [10, 74, 71, 76, 77, 93, 73, 75, 18]


# ── Audio engine ──────────────────────────────────────────────────────

class Player:
    def __init__(self):
        self.sub808 = Sub808Voice(SR)
        self.kick = KickVoice(SR)
        self.synth = PolySynthVoice(SR)
        self.fx = FXEngine(SR)
        self.chord_engine = ChordEngine(SR)
        # ChordEngine routes note events to synth (the only melodic engine for now)
        self.chord_engine.target = self.synth
        self.chord_send_enabled = True  # CHORD → SYNTH routing (on by default)

        self.voices = [self.sub808, self.kick, self.synth, self.fx, self.chord_engine]
        # Which engines route to the FX send bus (per voice index)
        self.send_enabled: list[bool] = [False, False, False, False, False]
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

        # MIDI note tracking: note -> voice_idx it was triggered on
        # So Note Off reaches the right engine even after switching
        self._midi_note_voice: dict[int, int] = {}

        # Waveform preview (rendered from params, not live audio)
        self.preview_buf = np.zeros(PREVIEW_SAMPLES, dtype=np.float32)
        self._preview_param_hash: int = 0
        self._preview_voice_idx: int = -1

        # Chord engine auto-triggers a chord when chromatic keys are pressed
        # on the CHORD tab. Store per-key chord notes so release sends matching offs.
        self._chord_held_notes: set[int] = set()

    @property
    def voice(self):
        return self.voices[self.voice_idx]

    @property
    def voice_name(self):
        return VOICE_NAMES[self.voice_idx]

    def audio_callback(self, outdata, frames, time_info, status):
        with self.lock:
            # ChordEngine: always advance; target is synth only if SEND is on
            self.chord_engine.target = self.synth if self.chord_send_enabled else None
            self.chord_engine.advance(frames)

            dry = np.zeros(frames, dtype=np.float32)
            send = np.zeros(frames, dtype=np.float32)

            for i, v in enumerate(self.voices):
                if i in (FX_IDX, CHORD_IDX):
                    continue  # FX + ChordEngine aren't direct sound sources
                if v.is_active:
                    voice_out = v.process(frames)
                    dry += voice_out
                    if self.send_enabled[i]:
                        send += voice_out

            # Always process FX so the tail decays naturally even after sends go silent
            buf = self.fx.process(send, dry)

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
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def switch_voice(self):
        self.voice_idx = (self.voice_idx + 1) % len(self.voices)
        self.current_page = 0
        self.selected_param = 0

    def select_engine(self, idx):
        """Switch to a specific engine by index. Releases held MIDI notes."""
        if idx == self.voice_idx or idx < 0 or idx >= len(self.voices):
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

    def release_note(self, note):
        with self.lock:
            self.voice.release(note)

    def trigger_chord(self, notes, velocity=0.9):
        with self.lock:
            self.synth.release_all()
            for note in notes:
                self.synth.trigger(note, velocity)

    def release_chord(self, notes):
        with self.lock:
            for note in notes:
                self.synth.release(note)

    @property
    def pages(self):
        return VOICE_PAGES[self.voice_name]

    @property
    def abs_param_index(self):
        return self.pages[self.current_page]["params"][self.selected_param]

    def adjust_param(self, delta):
        with self.lock:
            self.voice.params[self.abs_param_index].nudge(delta)

    def reset_params(self):
        with self.lock:
            for p in self.voice.params:
                p.value = p.default

    def update_preview(self):
        """Re-render the waveform preview if params changed."""
        # Build a hash from current voice index + all param values
        vals = tuple(round(p.value, 4) for p in self.voice.params)
        h = hash((self.voice_idx, vals))
        if h == self._preview_param_hash and self.voice_idx == self._preview_voice_idx:
            return  # nothing changed

        self._preview_param_hash = h
        self._preview_voice_idx = self.voice_idx

        # Render a preview with a temporary voice copy.
        # Per-engine sample counts tuned to show the most interesting content.
        if self.voice_idx == 0:  # 808
            tmp = Sub808Voice(SR)
            for i, p in enumerate(self.sub808.params):
                tmp.params[i].value = p.value
            tmp.trigger(36, 0.9)
            full = tmp.process(6000)
            self.preview_buf = full
        elif self.voice_idx == 1:  # Kick
            tmp = KickVoice(SR)
            for i, p in enumerate(self.kick.params):
                tmp.params[i].value = p.value
            tmp.trigger(36, 0.9)
            full = tmp.process(3000)
            self.preview_buf = full
        elif self.voice_idx == 2:  # Synth
            from cypher.synth.mono import MonoSynthVoice
            tmp = MonoSynthVoice(SR)
            for i, p in enumerate(self.synth.params):
                tmp.params[i].value = p.value
            tmp.trigger(60, 0.9)
            tmp.process(128)
            full = tmp.process(1500)
            self.preview_buf = full
        else:  # FX — no preview waveform
            self.preview_buf = np.zeros(PREVIEW_SAMPLES, dtype=np.float32)
            return  # skip normalization

        # Normalize for display
        peak = np.max(np.abs(self.preview_buf))
        if peak > 0.001:
            self.preview_buf = self.preview_buf / peak


# ── Rendering ─────────────────────────────────────────────────────────

def format_param_value(player, param_idx, p):
    """Format a parameter value for display."""
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
        # Discrete display overrides
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
        elif player.voice_idx == FX_IDX and param_idx == 3:
            from cypher.fx.fx import MODE_NAMES
            return MODE_NAMES[max(0, min(4, int(round(mapped))))]
        elif player.voice_idx == CHORD_IDX:
            from cypher.synth.chords import PROGRESSION_LIST
            if param_idx == 0:  # KEY
                return NOTE_NAMES[max(0, min(11, int(round(mapped))))]
            elif param_idx == 1:  # SCALE
                return SCALE_LIST[max(0, min(len(SCALE_LIST) - 1, int(round(mapped))))]
            elif param_idx == 2:  # PROGRESSION
                return PROGRESSION_LIST[max(0, min(len(PROGRESSION_LIST) - 1, int(round(mapped))))]
            elif param_idx == 3:  # STEP
                return f"{int(round(mapped)) + 1}"
            elif param_idx == 4:  # MODE
                return CHORD_MODE_NAMES[max(0, min(len(CHORD_MODE_NAMES) - 1, int(round(mapped))))]
            elif param_idx == 6:  # OCTAVE
                return f"C{int(round(mapped))}"
        return f"{p.value:.2f}"


def _draw_waveform(screen, player, x, y, w, h):
    """Draw a parameter-driven waveform preview."""
    # Panel background
    panel_rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, (14, 14, 28), panel_rect, border_radius=8)
    pygame.draw.rect(screen, (40, 40, 65), panel_rect, 1, border_radius=8)

    center_y = y + h // 2

    # Dotted center line
    for gx in range(x + 6, x + w - 6, 3):
        screen.set_at((gx, center_y), (40, 40, 85))

    # Update preview if params changed
    player.update_preview()
    samples = player.preview_buf

    # Downsample to display width
    display_w = w - 12
    n_samples = len(samples)
    indices = np.linspace(0, n_samples - 1, display_w).astype(int)
    downsampled = samples[indices]

    # Scale to pixel coordinates
    margin_v = 8
    usable_h = (h - 2 * margin_v) / 2.0
    clamped = np.clip(downsampled, -1.0, 1.0)
    py_arr = center_y - (clamped * usable_h).astype(int)
    px_arr = np.arange(display_w) + x + 6

    if np.max(np.abs(downsampled)) < 0.001:
        pygame.draw.aaline(screen, (50, 50, 70),
                           (x + 6, center_y), (x + w - 6, center_y))
        return

    # Filled area (translucent white)
    fill_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pts_top = [(int(px_arr[i]) - x, int(py_arr[i]) - y) for i in range(display_w)]
    pts_base = [(int(px_arr[-1]) - x, h // 2), (int(px_arr[0]) - x, h // 2)]
    poly = pts_top + pts_base
    if len(poly) >= 3:
        pygame.draw.polygon(fill_surf, (220, 220, 240, 14), poly)
    screen.blit(fill_surf, (x, y))

    # Glow (white)
    glow_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    glow_pts = [(int(px_arr[i]) - x, int(py_arr[i]) - y) for i in range(display_w)]
    if len(glow_pts) >= 2:
        pygame.draw.lines(glow_surf, (200, 200, 220, 30), False, glow_pts, 5)
        pygame.draw.lines(glow_surf, (220, 220, 240, 55), False, glow_pts, 3)
    screen.blit(glow_surf, (x, y))

    # Crisp white line
    points = list(zip(px_arr.astype(int), py_arr.astype(int)))
    if len(points) >= 2:
        pygame.draw.aalines(screen, (240, 240, 250), False, points)


def _draw_piano_keyboard(screen, fonts, player, x, y, w, h, hit):
    """Draw a 2-octave piano keyboard. Grey/white only.

    Every key is labelled. Notes currently being played by the chord engine
    show a colored dot. Clicking a key sets the KEY param.
    """
    font, font_bold, _, _ = fonts
    ce = player.chord_engine
    root = ce.root_midi
    # Notes currently sounding (triggered via chord engine, held on the target voice)
    playing = set(ce._held_notes)

    # Background panel
    panel = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, (14, 14, 28), panel, border_radius=8)
    pygame.draw.rect(screen, (40, 40, 65), panel, 1, border_radius=8)

    # Layout: 2 octaves = 14 white keys, starting at root's octave
    start_midi = (root // 12) * 12
    white_count = 14
    margin = 10
    avail_w = w - margin * 2
    white_w = avail_w // white_count
    white_h = h - 20
    white_to_chromatic = [0, 2, 4, 5, 7, 9, 11]
    black_positions = [(0, 1), (1, 3), (3, 6), (4, 8), (5, 10)]
    black_w = int(white_w * 0.62)
    black_h = int(white_h * 0.62)

    # --- White keys ---
    for i in range(white_count):
        oct_idx = i // 7
        note_in_oct = i % 7
        chromatic = white_to_chromatic[note_in_oct]
        midi = start_midi + oct_idx * 12 + chromatic

        kx = x + margin + i * white_w
        ky = y + 10

        # Plain white key
        pygame.draw.rect(screen, (240, 240, 245), (kx, ky, white_w - 1, white_h),
                         border_radius=3)
        pygame.draw.rect(screen, (60, 60, 80), (kx, ky, white_w - 1, white_h), 1,
                         border_radius=3)

        hit[f"piano_key_{midi}"] = pygame.Rect(kx, ky, white_w - 1, white_h)

        # Label (letter name) bottom of key
        letter = "CDEFGAB"[note_in_oct]
        lbl = font.render(letter, True, (70, 70, 90))
        lbl_x = kx + (white_w - 1 - lbl.get_width()) // 2
        screen.blit(lbl, (lbl_x, ky + white_h - 18))

        # Dot when playing
        if midi in playing:
            cx = kx + (white_w - 1) // 2
            cy = ky + white_h // 2 - 6
            pygame.draw.circle(screen, MAGENTA, (cx, cy), 6)

    # --- Black keys (drawn on top of white keys) ---
    for oct_idx in range(2):
        for white_idx, chromatic in black_positions:
            i = oct_idx * 7 + white_idx
            midi = start_midi + oct_idx * 12 + chromatic

            kx = x + margin + i * white_w + white_w - black_w // 2
            ky = y + 10

            pygame.draw.rect(screen, (45, 45, 60), (kx, ky, black_w, black_h),
                             border_radius=2)
            pygame.draw.rect(screen, (10, 10, 20), (kx, ky, black_w, black_h), 1,
                             border_radius=2)

            hit[f"piano_key_{midi}"] = pygame.Rect(kx, ky, black_w, black_h)

            # Label (sharp name)
            label_names = {1: "C#", 3: "D#", 6: "F#", 8: "G#", 10: "A#"}
            lbl = font.render(label_names[chromatic], True, (180, 180, 200))
            lbl_x = kx + (black_w - lbl.get_width()) // 2
            screen.blit(lbl, (lbl_x, ky + black_h - 16))

            # Dot when playing
            if midi in playing:
                cx = kx + black_w // 2
                cy = ky + black_h // 2 - 4
                pygame.draw.circle(screen, MAGENTA, (cx, cy), 5)


def draw(screen, fonts, player, held_notes, held_chord_key, held_chord_notes):
    """Render the full UI. Returns dict of clickable rects."""
    font, font_bold, font_lg, font_title = fonts
    hit: dict[str, pygame.Rect] = {}
    mouse_pos = pygame.mouse.get_pos()
    screen.fill(BG)
    y = 12

    # ── Title ──
    surf = font_title.render(f"CYPHER {player.voice_name}", True, CYAN)
    screen.blit(surf, (WIN_W // 2 - surf.get_width() // 2, y))
    y += 38

    # ── Voice tabs (clickable) ──
    tab_x = WIN_W // 2 - 120
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
        tab_x += 90
    y += 32

    # ── Note info ──
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
    y += 28

    # ── Level meter ──
    screen.blit(font.render("Level", True, TEXT), (20, y))
    mx, mw, mh = 80, WIN_W - 100, 14
    pygame.draw.rect(screen, SLIDER_BG, (mx, y + 2, mw, mh), border_radius=3)
    level = min(player.peak_level, 1.0)
    fw = int(level * mw)
    if fw > 0:
        c = GREEN if level < 0.8 else RED
        pygame.draw.rect(screen, c, (mx, y + 2, fw, mh), border_radius=3)
    y += 28

    # ── Page tabs (clickable) ──
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
    y += 28

    # ── Parameters ──
    page = player.pages[player.current_page]
    bx, bw = 15, WIN_W - 30
    bh = 4 * 38 + 16
    pygame.draw.rect(screen, PANEL, (bx, y, bw, bh), border_radius=6)
    pygame.draw.rect(screen, DIM, (bx, y, bw, bh), 1, border_radius=6)
    py = y + 8

    for pi, param_idx in enumerate(page["params"]):
        p = player.voice.params[param_idx]
        selected = pi == player.selected_param
        row_y = py + pi * 38

        row_rect = pygame.Rect(bx + 4, row_y - 2, bw - 8, 34)
        hit[f"param_row_{pi}"] = row_rect
        row_hover = row_rect.collidepoint(mouse_pos)

        if selected:
            pygame.draw.rect(screen, (40, 40, 70), row_rect, border_radius=4)
        elif row_hover:
            pygame.draw.rect(screen, (32, 32, 55), row_rect, border_radius=4)

        color = YELLOW if selected else (TEXT if row_hover else TEXT)
        prefix = "\u25b6 " if selected else "  "
        val_str = format_param_value(player, param_idx, p)

        screen.blit(font_bold.render(f"{prefix}{p.label}", True, color), (bx + 12, row_y + 4))
        screen.blit(font.render(val_str, True, color), (bx + 150, row_y + 4))

        # Slider (clickable + draggable)
        sx = bx + 260
        sw = bw - 280
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

    y += bh + 12

    # ── Waveform display / Piano keyboard (per-tab) ──
    if player.voice_idx == CHORD_IDX:
        # Full-width piano keyboard
        _draw_piano_keyboard(screen, fonts, player, 15, y, WIN_W - 30, WAVE_H, hit)
    elif player.voice_idx != FX_IDX:
        _draw_waveform(screen, player, WIN_W - WAVE_W - 15, y, WAVE_W, WAVE_H)
    y += WAVE_H + 10

    # ── Voice state ──
    state = player.voice.get_state()
    active_str = "ACTIVE" if state["active"] else "idle"
    sc = GREEN if state["active"] else DIM

    if player.voice_idx == 0:
        stage = state.get("amp_env_stage", "?")
        freq = f"{state['current_pitch_hz']:.1f}Hz"
        mode = state.get("trigger_mode", "classic").upper()
        info = f"Voice: {active_str}  Stage: {stage}  Pitch: {freq}  Mode: {mode}"
    elif player.voice_idx == 1:
        knock = f"Knock: {state.get('knock_env_stage', '?')}"
        body = f"Body: {state.get('body_env_stage', '?')}"
        paired = "PAIRED" if state.get('paired', False) else "UNPAIRED"
        info = f"Voice: {active_str}  {knock}  {body}  {paired}"
    elif player.voice_idx == 2:
        wa = state.get('wave_a', 'SAW')
        wb = state.get('wave_b', 'SAW')
        fm = state.get('filter_mode', 'LP')
        av = state.get('active_voices', 0)
        mv = state.get('max_voices', 8)
        notes = state.get('active_notes', [])
        ns = " ".join(NOTE_NAMES[n % 12] for n in notes) if notes else "---"
        info = f"Voice: {active_str}  {wa}+{wb} {fm}  Poly: {av}/{mv}  Notes: {ns}"
    elif player.voice_idx == FX_IDX:
        mo = state.get('mode', 'PLATE')
        mx = state.get('mix', 0.0)
        dec = state.get('decay_sec', 0.0)
        info = f"FX: {mo}   MIX: {mx*100:.0f}%   DECAY: {dec:.1f}s"
    else:  # CHORD
        key = state.get('key', 'C')
        scale = state.get('scale', 'MINOR')
        prog = state.get('progression', '')
        step = state.get('step', 0) + 1
        plen = state.get('progression_length', 1)
        mode = state.get('mode', 'CHORD')
        label = state.get('chord_label', '')
        info = f"Key: {key} {scale}   Prog: {prog} [{step}/{plen}]   Mode: {mode}   Now: {label}"
        sc = MAGENTA if state.get('active') else CYAN

    screen.blit(font.render(info, True, sc), (20, y))
    y += 22

    # ── All voice status + SEND indicators ──
    sub = "ON" if player.sub808.is_active else "off"
    kck = "ON" if player.kick.is_active else "off"
    syn = "ON" if player.synth.is_active else "off"
    mode808 = player.sub808._trigger_mode.upper()
    sw = player.synth.wave_a_name
    sv = player.synth.active_voice_count
    # Small send markers
    s0 = " →FX" if player.send_enabled[0] else ""
    s1 = " →FX" if player.send_enabled[1] else ""
    s2 = " →FX" if player.send_enabled[2] else ""
    chord_recv = " ←CHORD" if player.chord_send_enabled else ""
    status = f"808: {sub} [{mode808}]{s0}   Kick: {kck}{s1}   Synth: {syn} [{sw}] {sv}v{s2}{chord_recv}"
    screen.blit(font.render(status, True, DIM), (20, y))

    # Context-sensitive SEND toggle button (right side)
    if player.voice_idx == CHORD_IDX:
        sending = player.chord_send_enabled
        s_label = "→SYNTH" if sending else "SEND→SYNTH"
        s_key = "chord_send"
    elif player.voice_idx != FX_IDX:
        sending = player.send_enabled[player.voice_idx]
        s_label = "→FX ON" if sending else "SEND→FX"
        s_key = "fx_send"
    else:
        s_key = None

    if s_key is not None:
        s_color = GREEN if sending else DIM
        s_surf = font_bold.render(f"[{s_label}]", True, s_color)
        s_rect = s_surf.get_rect(topright=(WIN_W - 20, y))
        s_click = s_rect.inflate(10, 8)
        s_hover = s_click.collidepoint(mouse_pos)
        if s_hover:
            s_color = CYAN if not sending else GREEN
            s_surf = font_bold.render(f"[{s_label}]", True, s_color)
            pygame.draw.rect(screen, (35, 35, 60), s_click, border_radius=4)
        screen.blit(s_surf, s_rect)
        hit[s_key] = s_click
    y += 22

    # ── Chord info + clickable controls (SYNTH legacy or CHORD tab) ──
    if player.voice_idx in (2, CHORD_IDX):
        if player.voice_idx == CHORD_IDX:
            ce = player.chord_engine
            cs = ce.get_state()
            cur_step = cs['step']
            plen = cs['progression_length']
            prog_name = cs['progression']
            label = cs['chord_label']
            mode = cs['mode']
            ci = f"Chord: {label}   Prog: {prog_name}   Mode: {mode}"
        else:
            prog_name = PROGRESSION_LIST[player.chord_prog_idx]
            cur_step = player.chord_step
            plen = progression_length(prog_name)
            root = player.octave * 12 + 24
            _, label = build_progression_chord(root, prog_name, cur_step)
            ci = f"Chord: {label}   Prog: {prog_name}"
        ci_surf = font_bold.render(ci, True, MAGENTA)
        screen.blit(ci_surf, (20, y))

        # Step position dots — filled for current step, hollow for others
        dots_x = 20 + ci_surf.get_width() + 14
        for i in range(plen):
            cx = dots_x + i * 14
            cy = y + 10
            if i == cur_step:
                pygame.draw.circle(screen, MAGENTA, (cx, cy), 5)
            else:
                pygame.draw.circle(screen, (60, 60, 80), (cx, cy), 4, 1)

        # Chord navigation buttons
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

    # ── Help ──
    y += 8
    help_lines = [
        "TAB voice   SPACE trigger   A-K chromatic   Z/X octave   UP/DOWN param",
        "LEFT/RIGHT adjust   [/] fine   1/2/3 page   P pair   M mode   R reset",
        "C chord   ,/. step chord   N cycle prog   V send to FX   Q quit",
    ]
    for line in help_lines:
        screen.blit(font.render(line, True, DIM), (20, y))
        y += 18

    pygame.display.flip()
    return hit


# ── MIDI handling ─────────────────────────────────────────────────────

def make_midi_callback(player, monitor=False):
    """Create a MIDI message handler bound to the player.

    Called from mido's background thread — Player methods are lock-protected.
    """
    def callback(msg):
        if monitor:
            print(f"  MIDI: {msg}")

        if isinstance(msg, NoteOn):
            if msg.channel == PAD_CHANNEL and msg.note in PAD_ENGINE_MAP:
                # Pad hit → engine select
                player.select_engine(PAD_ENGINE_MAP[msg.note])
            else:
                # Regular note → trigger on current engine
                player._midi_note_voice[msg.note] = player.voice_idx
                player.trigger(msg.note, msg.velocity_float)

        elif isinstance(msg, NoteOff):
            if msg.channel == PAD_CHANNEL and msg.note in PAD_ENGINE_MAP:
                pass  # Pad release — ignore
            else:
                # Release on the engine that originally got this note
                vidx = player._midi_note_voice.pop(msg.note, player.voice_idx)
                with player.lock:
                    player.voices[vidx].release(msg.note)

        elif isinstance(msg, ControlChange):
            # Map first 4 knob CCs to current page params
            if msg.cc in KNOB_CCS:
                ki = KNOB_CCS.index(msg.cc)
                if ki < 4:
                    page = player.pages[player.current_page]
                    if ki < len(page["params"]):
                        pidx = page["params"][ki]
                        with player.lock:
                            player.voice.params[pidx].value = msg.value_float

    return callback


def open_midi(player, device_name=None, monitor=False):
    """Try to open MIDI input. Returns MidiInput or None."""
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


# ── Headless mode (Pi over SSH) ────────────────────────────────────────

def run_headless(player, midi_in):
    """Audio + MIDI only, no display. Status printed to terminal."""
    print(f"\nCYPHER headless — {SR}Hz / {BLOCK_SIZE} frames")
    if midi_in:
        print(f"MIDI: {midi_in.device_name}")
    print("Ctrl+C to quit\n")
    print(f"Engine: {player.voice_name}")
    print("Pad 1=808  Pad 2=KICK  Pad 3=SYNTH\n")

    try:
        while True:
            state = player.voice.get_state()
            active = "● ACTIVE" if state["active"] else "○ idle"
            level = f"{player.peak_level:.2f}"
            print(
                f"\r  [{player.voice_name}] {active}  Level: {level}   ",
                end="", flush=True,
            )
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nShutting down...")


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CYPHER interactive tweaker")
    p.add_argument("--headless", action="store_true",
                   help="No display — audio + MIDI only (for Pi over SSH)")
    p.add_argument("--midi", action="store_true",
                   help="Enable MIDI input (auto-detect device)")
    p.add_argument("--midi-device", type=str, default=None,
                   help="MIDI device name (skip auto-detect)")
    p.add_argument("--list-midi", action="store_true",
                   help="List MIDI devices and exit")
    p.add_argument("--midi-monitor", action="store_true",
                   help="Print all incoming MIDI messages (for mapping discovery)")
    return p.parse_args()


# ── Main loop ─────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── List MIDI devices and exit ──
    if args.list_midi:
        devices = midi_input.list_inputs() if midi_input.available() else []
        if devices:
            print("MIDI input devices:")
            for i, name in enumerate(devices):
                print(f"  [{i}] {name}")
        else:
            print("No MIDI input devices found.")
            if not midi_input.available():
                print("  (mido not installed — pip install mido python-rtmidi)")
        return

    # ── Init player + audio ──
    player = Player()
    player.start()
    midi_in = None

    # ── Open MIDI if requested ──
    use_midi = args.midi or args.midi_device or args.headless or args.midi_monitor
    if use_midi:
        midi_in = open_midi(
            player,
            device_name=args.midi_device,
            monitor=args.midi_monitor,
        )

    # ── Headless mode (Pi) ──
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

    font = pygame.font.SysFont("monospace", 16)
    font_bold = pygame.font.SysFont("monospace", 16, bold=True)
    font_lg = pygame.font.SysFont("monospace", 22, bold=True)
    font_title = pygame.font.SysFont("monospace", 26, bold=True)
    fonts = (font, font_bold, font_lg, font_title)

    # Key state — proper KEYDOWN/KEYUP, no timing hacks
    held_notes: dict[int, int] = {}    # pygame key -> midi note
    held_chord_key = False
    held_chord_notes: list[int] = []
    hit: dict[str, pygame.Rect] = {}   # clickable regions from draw()
    dragging_slider: int | None = None  # param index being dragged

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    key = event.key

                    # ── Quit ──
                    if key == pygame.K_q or key == pygame.K_ESCAPE:
                        running = False

                    # ── Voice switch ──
                    elif key == pygame.K_TAB:
                        for k, note in list(held_notes.items()):
                            player.release_note(note)
                        held_notes.clear()
                        if held_chord_key:
                            player.release_chord(held_chord_notes)
                            held_chord_key = False
                            held_chord_notes = []
                        player.switch_voice()

                    # ── Chord trigger (synth, hold to sustain) ──
                    elif key == pygame.K_c and player.voice_idx == 2 and not held_chord_key:
                        root = player.octave * 12 + 24
                        pname = PROGRESSION_LIST[player.chord_prog_idx]
                        notes, label = build_progression_chord(
                            root, pname, player.chord_step)
                        player.chord_label = label
                        player.trigger_chord(notes)
                        held_chord_key = True
                        held_chord_notes = notes

                    # ── Space — root note ──
                    elif key == pygame.K_SPACE and key not in held_notes:
                        midi_note = player.octave * 12 + 24
                        player.trigger(midi_note)
                        held_notes[key] = midi_note

                    # ── Chromatic keys — polyphonic ──
                    elif key in CHROMATIC_KEYS and key not in held_notes:
                        semitone = CHROMATIC_KEYS[key]
                        midi_note = player.octave * 12 + 24 + semitone
                        player.trigger(midi_note)
                        held_notes[key] = midi_note

                    # ── Param navigation (repeats OK via set_repeat) ──
                    elif key == pygame.K_UP:
                        player.selected_param = max(0, player.selected_param - 1)
                    elif key == pygame.K_DOWN:
                        player.selected_param = min(3, player.selected_param + 1)
                    elif key == pygame.K_RIGHT:
                        player.adjust_param(0.05)
                    elif key == pygame.K_LEFT:
                        player.adjust_param(-0.05)
                    elif key == pygame.K_LEFTBRACKET:
                        player.adjust_param(-0.01)
                    elif key == pygame.K_RIGHTBRACKET:
                        player.adjust_param(0.01)

                    # ── Pages ──
                    elif key == pygame.K_1:
                        player.current_page = 0
                        player.selected_param = min(player.selected_param, 3)
                    elif key == pygame.K_2 and len(player.pages) > 1:
                        player.current_page = 1
                        player.selected_param = min(player.selected_param, 3)
                    elif key == pygame.K_3 and len(player.pages) > 2:
                        player.current_page = 2
                        player.selected_param = min(player.selected_param, 3)
                    elif key == pygame.K_4 and len(player.pages) > 3:
                        player.current_page = 3
                        player.selected_param = min(player.selected_param, 3)

                    # ── Octave ──
                    elif key == pygame.K_z:
                        player.octave = max(-1, player.octave - 1)
                    elif key == pygame.K_x:
                        player.octave = min(6, player.octave + 1)

                    # ── Reset ──
                    elif key == pygame.K_r:
                        player.reset_params()

                    # ── 808 pair/unpair ──
                    elif key == pygame.K_p:
                        if player.kick._paired:
                            player.kick.unpair_808()
                        else:
                            freq = (player.sub808._current_pitch_hz
                                    if player.sub808.is_active else 32.7)
                            player.kick.pair_808(freq)

                    # ── Trigger mode ──
                    elif key == pygame.K_m:
                        m = player.sub808._trigger_mode
                        player.sub808._trigger_mode = (
                            "oneshot" if m == "classic" else "classic")

                    # ── SEND toggle for current engine ──
                    elif key == pygame.K_v and player.voice_idx != FX_IDX:
                        player.send_enabled[player.voice_idx] = (
                            not player.send_enabled[player.voice_idx]
                        )

                    # ── Chord navigation (synth only) ──
                    elif key == pygame.K_PERIOD and player.voice_idx == 2:
                        pname = PROGRESSION_LIST[player.chord_prog_idx]
                        player.chord_step = (
                            (player.chord_step + 1) % progression_length(pname))
                    elif key == pygame.K_COMMA and player.voice_idx == 2:
                        pname = PROGRESSION_LIST[player.chord_prog_idx]
                        player.chord_step = (
                            (player.chord_step - 1) % progression_length(pname))
                    elif key == pygame.K_n and player.voice_idx == 2:
                        player.chord_prog_idx = (
                            (player.chord_prog_idx + 1) % len(PROGRESSION_LIST))
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
                            player.select_engine(i)
                            break

                    # Page tabs
                    for i in range(len(player.pages)):
                        r = hit.get(f"page_{i}")
                        if r and r.collidepoint(mx, my):
                            player.current_page = i
                            player.selected_param = min(player.selected_param, 3)
                            break

                    # SEND toggle — context-sensitive
                    r = hit.get("fx_send")
                    if r and r.collidepoint(mx, my):
                        player.send_enabled[player.voice_idx] = (
                            not player.send_enabled[player.voice_idx]
                        )
                    r = hit.get("chord_send")
                    if r and r.collidepoint(mx, my):
                        player.chord_send_enabled = not player.chord_send_enabled

                    # Piano keyboard click — set KEY on chord engine
                    for key_name, rect in list(hit.items()):
                        if key_name.startswith("piano_key_") and rect.collidepoint(mx, my):
                            midi = int(key_name[len("piano_key_"):])
                            player.chord_engine.set_key(midi % 12)
                            break

                    # Slider click (start drag) — check before param rows
                    slider_hit = False
                    for pi in range(4):
                        r = hit.get(f"slider_{pi}")
                        if r and r.collidepoint(mx, my):
                            player.selected_param = pi
                            t = max(0.0, min(1.0, (mx - r.x) / r.width))
                            page = player.pages[player.current_page]
                            with player.lock:
                                player.voice.params[page["params"][pi]].value = t
                            dragging_slider = pi
                            slider_hit = True
                            break

                    # Parameter row selection (only if no slider was hit)
                    if not slider_hit:
                        for pi in range(4):
                            r = hit.get(f"param_row_{pi}")
                            if r and r.collidepoint(mx, my):
                                player.selected_param = pi
                                break

                    # Chord controls — available on SYNTH (legacy) and CHORD tabs
                    if player.voice_idx in (2, CHORD_IDX):
                        r = hit.get("chord_prev")
                        if r and r.collidepoint(mx, my):
                            if player.voice_idx == CHORD_IDX:
                                player.chord_engine.step_progression(-1)
                            else:
                                pname = PROGRESSION_LIST[player.chord_prog_idx]
                                player.chord_step = (player.chord_step - 1) % progression_length(pname)
                        r = hit.get("chord_next")
                        if r and r.collidepoint(mx, my):
                            if player.voice_idx == CHORD_IDX:
                                player.chord_engine.step_progression(1)
                            else:
                                pname = PROGRESSION_LIST[player.chord_prog_idx]
                                player.chord_step = (player.chord_step + 1) % progression_length(pname)
                        r = hit.get("chord_cycle")
                        if r and r.collidepoint(mx, my):
                            if player.voice_idx == CHORD_IDX:
                                cur = int(round(player.chord_engine.params[2].value * (len(PROGRESSION_LIST) - 1)))
                                player.chord_engine.set_progression((cur + 1) % len(PROGRESSION_LIST))
                            else:
                                player.chord_prog_idx = (player.chord_prog_idx + 1) % len(PROGRESSION_LIST)
                                player.chord_step = 0

                elif event.type == pygame.MOUSEMOTION and dragging_slider is not None:
                    pi = dragging_slider
                    r = hit.get(f"slider_{pi}")
                    if r:
                        mx = event.pos[0]
                        t = max(0.0, min(1.0, (mx - r.x) / r.width))
                        page = player.pages[player.current_page]
                        with player.lock:
                            player.voice.params[page["params"][pi]].value = t

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging_slider = None

                elif event.type == pygame.KEYUP:
                    key = event.key
                    # Release individual notes
                    if key in held_notes:
                        player.release_note(held_notes.pop(key))
                    # Release chord
                    elif key == pygame.K_c and held_chord_key:
                        player.release_chord(held_chord_notes)
                        held_chord_key = False
                        held_chord_notes = []

            # ── Render ──
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
