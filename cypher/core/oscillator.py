"""Oscillator primitives for synthesis."""

from __future__ import annotations

import numpy as np

from .types import AudioBuffer, DEFAULT_SAMPLE_RATE


class SineOscillator:
    """Phase-continuous sine oscillator.

    Accepts per-sample frequency input (from pitch envelope) to support
    pitch slides and glides without discontinuities.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._phase: float = 0.0

    def process(self, freq: AudioBuffer) -> AudioBuffer:
        """Generate sine wave for per-sample frequency array.

        Args:
            freq: Array of frequency values (one per sample).

        Returns:
            Audio buffer of sine wave samples.
        """
        # Phase increment per sample: freq / sample_rate
        phase_inc = freq / self.sample_rate
        # Accumulate phase
        phases = self._phase + np.cumsum(phase_inc)
        # Wrap phase to avoid float precision loss over time
        phases = phases % 1.0
        self._phase = float(phases[-1]) if len(phases) > 0 else self._phase

        return np.sin(2.0 * np.pi * phases).astype(np.float32)

    def process_fixed(self, freq_hz: float, num_frames: int) -> AudioBuffer:
        """Generate sine wave at a fixed frequency."""
        freq_array = np.full(num_frames, freq_hz, dtype=np.float32)
        return self.process(freq_array)

    def reset(self) -> None:
        self._phase = 0.0


class NoiseGenerator:
    """White noise generator with optional seeding for reproducibility."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def process(self, num_frames: int) -> AudioBuffer:
        """Generate white noise samples in [-1, 1]."""
        return self._rng.uniform(-1.0, 1.0, num_frames).astype(np.float32)


class SquareOscillator:
    """Band-limited square wave oscillator.

    Used for metallic hi-hat synthesis (multiple detuned squares
    at inharmonic ratios = that classic analog hat sound).
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._phase: float = 0.0

    def process(self, freq: AudioBuffer) -> AudioBuffer:
        """Generate square wave for per-sample frequency array."""
        phase_inc = freq / self.sample_rate
        phases = self._phase + np.cumsum(phase_inc)
        phases = phases % 1.0
        self._phase = float(phases[-1]) if len(phases) > 0 else self._phase

        # Simple square: +1 for first half, -1 for second half
        return np.where(phases < 0.5,
                        np.float32(1.0),
                        np.float32(-1.0)).astype(np.float32)

    def process_fixed(self, freq_hz: float, num_frames: int) -> AudioBuffer:
        freq_array = np.full(num_frames, freq_hz, dtype=np.float32)
        return self.process(freq_array)

    def reset(self) -> None:
        self._phase = 0.0


class MetallicOscillator:
    """Six detuned square waves at inharmonic frequency ratios.

    Produces the metallic tone characteristic of analog hi-hats.
    Based on the TR-808 circuit which uses 6 square oscillators
    at carefully chosen non-integer frequency ratios.
    """

    # TR-808 inspired frequency ratios (inharmonic = metallic)
    RATIOS = [1.0, 1.4471, 1.6818, 1.9545, 2.2727, 2.6364]

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._oscillators = [SquareOscillator(sample_rate) for _ in self.RATIOS]

    def process(self, base_freq: AudioBuffer) -> AudioBuffer:
        """Generate metallic tone from 6 detuned squares."""
        output = np.zeros(len(base_freq), dtype=np.float32)
        for osc, ratio in zip(self._oscillators, self.RATIOS):
            output += osc.process(base_freq * ratio)
        # Normalize by number of oscillators
        output /= len(self.RATIOS)
        return output

    def process_fixed(self, base_freq_hz: float, num_frames: int) -> AudioBuffer:
        freq_array = np.full(num_frames, base_freq_hz, dtype=np.float32)
        return self.process(freq_array)

    def reset(self) -> None:
        for osc in self._oscillators:
            osc.reset()
