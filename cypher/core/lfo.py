"""Low-frequency oscillator for modulation.

Generates per-sample modulation values in the -1.0 to +1.0 range.
Designed to modulate filter cutoff, pitch, or amplitude.

Waveforms: sine, triangle, square, sample-and-hold (random).
Phase-continuous — no clicks on parameter changes.

C++ portability notes:
  - All state is plain floats/ints, no dynamic allocation.
  - process() writes into a pre-sized buffer, maps to a simple loop.
  - No closures, no generators.
"""

from __future__ import annotations

import numpy as np

from .types import AudioBuffer, DEFAULT_SAMPLE_RATE

# LFO waveform indices
LFO_SINE = 0
LFO_TRI = 1
LFO_SQUARE = 2
LFO_SH = 3  # sample-and-hold (random steps)
LFO_NAMES = ["SIN", "TRI", "SQR", "S&H"]


class LFO:
    """Low-frequency oscillator for modulation routing.

    Output range: -1.0 to +1.0 (bipolar).
    The consumer scales this by depth and maps to the target parameter.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._phase: float = 0.0
        self._wave: int = LFO_SINE
        self._rate_hz: float = 1.0

        # Sample-and-hold state
        self._sh_value: float = 0.0
        self._sh_last_phase: float = 0.0

    @property
    def wave(self) -> int:
        return self._wave

    @wave.setter
    def wave(self, v: int) -> None:
        self._wave = max(0, min(3, v))

    @property
    def rate_hz(self) -> float:
        return self._rate_hz

    @rate_hz.setter
    def rate_hz(self, v: float) -> None:
        self._rate_hz = max(0.01, min(50.0, v))

    def reset(self) -> None:
        """Reset phase to 0. Call on note trigger if free-running isn't wanted."""
        self._phase = 0.0
        self._sh_value = 0.0
        self._sh_last_phase = 0.0

    def process(self, num_frames: int) -> AudioBuffer:
        """Generate modulation signal.

        Returns:
            Float32 array of length num_frames, values in [-1.0, +1.0].
        """
        # Phase ramp for this block
        phase_inc = self._rate_hz / self.sample_rate
        phases = (self._phase + np.arange(num_frames, dtype=np.float64) * phase_inc) % 1.0
        self._phase = (self._phase + num_frames * phase_inc) % 1.0

        wave = self._wave

        if wave == LFO_SINE:
            output = np.sin(2.0 * np.pi * phases)

        elif wave == LFO_TRI:
            # Triangle: 0→0.25 rises to +1, 0.25→0.75 falls to -1, 0.75→1.0 rises to 0
            output = 4.0 * np.abs(phases - 0.5) - 1.0

        elif wave == LFO_SQUARE:
            output = np.where(phases < 0.5, 1.0, -1.0)

        elif wave == LFO_SH:
            # Sample-and-hold: new random value each cycle
            output = np.empty(num_frames, dtype=np.float64)
            sh_val = self._sh_value
            last_p = self._sh_last_phase
            for i in range(num_frames):
                p = phases[i]
                # Detect phase wrap (new cycle)
                if p < last_p:
                    sh_val = np.random.uniform(-1.0, 1.0)
                last_p = p
                output[i] = sh_val
            self._sh_value = sh_val
            self._sh_last_phase = last_p

        else:
            output = np.zeros(num_frames, dtype=np.float64)

        return output.astype(np.float32)
