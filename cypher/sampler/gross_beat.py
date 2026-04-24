"""Gross Beat — BPM-synced volume FX presets for the sampler.

6 preset patterns, each a periodic volume curve parameterised over "phase"
(0.0–1.0 within one beat of the project BPM). Applied as a multiplicative
gain on the sampler's output at process time.

Presets:
    OFF       — pass-through (gain = 1.0)
    STUTTER   — 1/16 fast chop (8 retriggered chops per beat)
    GATE      — 1/8 half-open gate
    PUMP      — sidechain-style pump (slow attack after each beat)
    SWELL     — slow exponential fade-in across the beat
    TREMOLO   — smooth sinusoidal modulation at 1/8

C++ portability notes:
    - All presets are plain math on phase; can be hand-coded as if-else.
    - One-block update: we compute gain values once per block (at block rate),
      then smooth across the block with a short linear ramp to avoid zipper.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.types import AudioBuffer


PRESET_OFF = 0
PRESET_STUTTER = 1
PRESET_GATE = 2
PRESET_PUMP = 3
PRESET_SWELL = 4
PRESET_TREMOLO = 5

PRESET_NAMES = ["OFF", "STUTTER", "GATE", "PUMP", "SWELL", "TREMOLO"]


def _gain_for_phase(preset: int, phase: float) -> float:
    """Instantaneous gain (0..1) for a given preset at phase ∈ [0, 1).

    `phase` advances 0→1 across one beat (adjust at higher level for 1/4
    etc.). Presets define their own sub-beat structure on top.
    """
    if preset == PRESET_OFF:
        return 1.0

    if preset == PRESET_STUTTER:
        # 8 retriggered chops per beat; each chop: quick attack → decay
        chop_phase = (phase * 8.0) % 1.0
        # Fast attack (first 5%), then exponential decay
        if chop_phase < 0.05:
            return chop_phase / 0.05
        return math.exp(-(chop_phase - 0.05) * 6.0)

    if preset == PRESET_GATE:
        # Half-beat hard gate (on for first half, off for second)
        sub_phase = (phase * 4.0) % 1.0
        return 1.0 if sub_phase < 0.5 else 0.0

    if preset == PRESET_PUMP:
        # Sidechain duck: silence briefly at beat start, ease back in
        # 0–0.1: drop to 0. 0.1–0.8: rise back to 1. 0.8–1.0: held at 1.
        if phase < 0.08:
            return max(0.0, 1.0 - phase / 0.08) * 0.05 + 0.0
        if phase < 0.8:
            t = (phase - 0.08) / 0.72
            # Smooth ease-out curve
            return 1.0 - (1.0 - t) ** 2
        return 1.0

    if preset == PRESET_SWELL:
        # Exponential fade-in over the whole beat, held briefly at the top
        if phase < 0.9:
            t = phase / 0.9
            return t * t  # quadratic attack
        return 1.0

    if preset == PRESET_TREMOLO:
        # Smooth sine at 1/8 rate (4 full cycles per beat)
        return 0.5 + 0.5 * math.sin(2.0 * math.pi * phase * 4.0)

    return 1.0


class GrossBeat:
    """Applies a Gross Beat-style volume preset to an audio buffer.

    One instance per sampler. Keeps its own phase accumulator so block
    boundaries don't re-trigger the pattern.
    """

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.preset: int = PRESET_OFF
        self.bpm: float = 120.0
        self._phase: float = 0.0          # 0.0–1.0, advances across one beat

    @property
    def preset_name(self) -> str:
        return PRESET_NAMES[self.preset]

    def reset(self) -> None:
        self._phase = 0.0

    def process(self, buf: AudioBuffer, bpm: float | None = None) -> AudioBuffer:
        """Apply preset gain to `buf` in-place-safe (returns new array)."""
        if self.preset == PRESET_OFF:
            return buf
        if bpm is not None:
            self.bpm = bpm

        n = len(buf)
        beat_samples = max(1.0, 60.0 / self.bpm * self.sample_rate)
        phase_inc = 1.0 / beat_samples  # phase per sample

        # Compute gain at block start + end, ramp across the block.
        # (Cheap + smooth; avoids zipper vs. per-sample recompute in Python.)
        g_start = _gain_for_phase(self.preset, self._phase)
        phase_end = (self._phase + phase_inc * n) % 1.0
        g_end = _gain_for_phase(self.preset, phase_end)

        # For presets with sharp transitions (STUTTER, GATE) we want the sharp
        # edges preserved; compute per-sample at modest cost for those.
        if self.preset in (PRESET_STUTTER, PRESET_GATE):
            phases = (self._phase + np.arange(n, dtype=np.float64) * phase_inc) % 1.0
            gains = np.empty(n, dtype=np.float32)
            if self.preset == PRESET_GATE:
                sub = (phases * 4.0) % 1.0
                gains[:] = (sub < 0.5).astype(np.float32)
            else:  # STUTTER
                cp = (phases * 8.0) % 1.0
                attack_mask = cp < 0.05
                gains[attack_mask] = (cp[attack_mask] / 0.05).astype(np.float32)
                gains[~attack_mask] = np.exp(-(cp[~attack_mask] - 0.05) * 6.0).astype(
                    np.float32
                )
            out = buf * gains
        else:
            ramp = np.linspace(g_start, g_end, n, dtype=np.float32)
            out = buf * ramp

        self._phase = phase_end
        return out
