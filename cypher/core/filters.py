"""Biquad filter implementations.

Used by drum voices (noise shaping, metallic tone) and synth engine
(multimode filter). Direct-form II transposed biquad — efficient,
stable, sequential per-sample processing.

C++ portability notes:
  - All state is plain floats, no dynamic allocation.
  - process() is a tight per-sample loop — maps directly to C++.
  - set_*() methods compute coefficients from frequency/Q.
"""

from __future__ import annotations

import math

import numpy as np

from .types import AudioBuffer, DEFAULT_SAMPLE_RATE


class BiquadFilter:
    """Second-order IIR filter (biquad).

    Supports lowpass, highpass, bandpass, and notch modes.
    Processes audio buffer-at-a-time while maintaining state
    between calls for seamless real-time operation.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        # Coefficients
        self._b0: float = 1.0
        self._b1: float = 0.0
        self._b2: float = 0.0
        self._a1: float = 0.0
        self._a2: float = 0.0
        # State (direct form II transposed)
        self._z1: float = 0.0
        self._z2: float = 0.0

    def set_lowpass(self, freq: float, q: float = 0.707) -> None:
        """Configure as lowpass filter."""
        w0 = 2.0 * math.pi * freq / self.sample_rate
        alpha = math.sin(w0) / (2.0 * q)
        cos_w0 = math.cos(w0)

        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        self._set_coeffs(b0, b1, b2, a0, a1, a2)

    def set_highpass(self, freq: float, q: float = 0.707) -> None:
        """Configure as highpass filter."""
        w0 = 2.0 * math.pi * freq / self.sample_rate
        alpha = math.sin(w0) / (2.0 * q)
        cos_w0 = math.cos(w0)

        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        self._set_coeffs(b0, b1, b2, a0, a1, a2)

    def set_bandpass(self, freq: float, q: float = 1.0) -> None:
        """Configure as bandpass filter."""
        w0 = 2.0 * math.pi * freq / self.sample_rate
        alpha = math.sin(w0) / (2.0 * q)
        cos_w0 = math.cos(w0)

        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        self._set_coeffs(b0, b1, b2, a0, a1, a2)

    def _set_coeffs(self, b0: float, b1: float, b2: float,
                    a0: float, a1: float, a2: float) -> None:
        """Normalize and store coefficients."""
        self._b0 = b0 / a0
        self._b1 = b1 / a0
        self._b2 = b2 / a0
        self._a1 = a1 / a0
        self._a2 = a2 / a0

    def process(self, signal: AudioBuffer) -> AudioBuffer:
        """Filter the input buffer, maintaining state between calls."""
        output = np.empty(len(signal), dtype=np.float32)

        z1 = self._z1
        z2 = self._z2
        b0, b1, b2 = self._b0, self._b1, self._b2
        a1, a2 = self._a1, self._a2

        # Per-sample processing (biquad state requires sequential)
        for i in range(len(signal)):
            x = float(signal[i])
            y = b0 * x + z1
            z1 = b1 * x - a1 * y + z2
            z2 = b2 * x - a2 * y
            output[i] = y

        self._z1 = z1
        self._z2 = z2

        return output

    def set_mode(self, mode: int, freq: float, q: float = 0.707) -> None:
        """Set filter by mode index.

        Mode indices:
            0 = lowpass
            1 = highpass
            2 = bandpass

        This is the interface the synth voice uses — a single int
        from a Parameter snap value selects the filter type.
        """
        if mode == 0:
            self.set_lowpass(freq, q)
        elif mode == 1:
            self.set_highpass(freq, q)
        elif mode == 2:
            self.set_bandpass(freq, q)
        else:
            self.set_lowpass(freq, q)

    def reset(self) -> None:
        """Clear filter state."""
        self._z1 = 0.0
        self._z2 = 0.0


# Filter mode constants for parameter snap values
FILTER_LP = 0
FILTER_HP = 1
FILTER_BP = 2
FILTER_MODE_NAMES = ["LP", "HP", "BP"]
