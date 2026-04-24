"""Tests for phase-vocoder pitch shift + time stretch."""

import numpy as np
import pytest

from cypher.sampler.pitch_shift import (
    pitch_shift, time_stretch,
)


def _sine(freq: float, dur: float, sr: int) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _dominant_freq(data: np.ndarray, sr: int) -> float:
    """Return the bin-center frequency with the highest FFT magnitude."""
    spec = np.abs(np.fft.rfft(data))
    peak_bin = int(np.argmax(spec))
    return peak_bin * sr / len(data)


class TestPitchShift:
    def test_zero_semitones_passes_through(self, sample_rate):
        data = _sine(440, 0.5, sample_rate)
        out = pitch_shift(data, 0.0)
        # Exact pass-through for 0 semitones
        np.testing.assert_array_equal(out, data)

    def test_length_preserved(self, sample_rate):
        data = _sine(440, 1.0, sample_rate)
        for st in [-12, -5, 3, 7, 12]:
            out = pitch_shift(data, st)
            assert abs(len(out) - len(data)) <= 8  # tolerance for rounding

    def test_shift_up_octave_doubles_freq(self, sample_rate):
        data = _sine(440, 1.0, sample_rate)
        out = pitch_shift(data, 12.0)  # +1 octave
        f_in = _dominant_freq(data, sample_rate)
        f_out = _dominant_freq(out, sample_rate)
        # Should be ~2x within 5%
        assert 1.9 < (f_out / f_in) < 2.1

    def test_shift_down_octave_halves_freq(self, sample_rate):
        data = _sine(880, 1.0, sample_rate)
        out = pitch_shift(data, -12.0)
        f_in = _dominant_freq(data, sample_rate)
        f_out = _dominant_freq(out, sample_rate)
        assert 0.45 < (f_out / f_in) < 0.55

    def test_shift_fifth(self, sample_rate):
        """+7 semitones = 3/2 frequency ratio (perfect fifth)."""
        data = _sine(440, 1.0, sample_rate)
        out = pitch_shift(data, 7.0)
        f_in = _dominant_freq(data, sample_rate)
        f_out = _dominant_freq(out, sample_rate)
        expected = 2 ** (7 / 12)  # ~1.498
        assert abs((f_out / f_in) - expected) < 0.05

    def test_short_sample_unchanged(self, sample_rate):
        """Buffers shorter than frame_size should pass through."""
        data = np.random.default_rng(0).standard_normal(512).astype(np.float32) * 0.1
        out = pitch_shift(data, 5.0)
        np.testing.assert_array_equal(out, data)


class TestTimeStretch:
    def test_ratio_1_passes_through(self, sample_rate):
        data = _sine(440, 0.5, sample_rate)
        out = time_stretch(data, 1.0)
        np.testing.assert_array_equal(out, data)

    def test_stretch_doubles_length(self, sample_rate):
        data = _sine(440, 1.0, sample_rate)
        out = time_stretch(data, 2.0)
        # Output length should be ~2x, within 5% (framing rounding)
        ratio = len(out) / len(data)
        assert 1.9 < ratio < 2.1

    def test_compress_halves_length(self, sample_rate):
        data = _sine(440, 1.0, sample_rate)
        out = time_stretch(data, 0.5)
        ratio = len(out) / len(data)
        assert 0.45 < ratio < 0.55

    def test_time_stretch_preserves_pitch(self, sample_rate):
        data = _sine(440, 1.0, sample_rate)
        out = time_stretch(data, 1.5)
        f_in = _dominant_freq(data, sample_rate)
        f_out = _dominant_freq(out, sample_rate)
        # Pitch shouldn't change when time-stretching (within 3%)
        assert abs(f_out - f_in) / f_in < 0.03
