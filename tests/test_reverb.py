"""Tests for Dattorro plate reverb."""

import numpy as np
import pytest

from cypher.core.reverb import (
    DattorroPlateReverb, DelayLine, AllpassFilter, OnePoleLP,
)


class TestDelayLine:
    def test_read_back_delayed(self):
        """Written sample should appear at the correct delay."""
        dl = DelayLine(100)
        dl.write(1.0)
        for _ in range(49):
            dl.write(0.0)
        assert dl.read(50) == 1.0

    def test_fractional_read(self):
        """Fractional delay should interpolate between samples."""
        dl = DelayLine(100)
        dl.write(0.0)
        dl.write(1.0)
        # delay=1.5 should interpolate between the two samples
        val = dl.read_frac(1.5)
        assert 0.4 < val < 0.6

    def test_clear_zeroes_buffer(self):
        dl = DelayLine(64)
        for i in range(64):
            dl.write(1.0)
        dl.clear()
        assert dl.read(1) == 0.0


class TestAllpassFilter:
    def test_energy_preservation(self):
        """Allpass should approximately preserve signal energy."""
        ap = AllpassFilter(37, gain=0.5)
        rng = np.random.default_rng(42)
        input_signal = rng.standard_normal(2048).astype(np.float32)

        output = np.zeros(2048, dtype=np.float32)
        for i in range(2048):
            output[i] = ap.process(float(input_signal[i]))

        # Skip transient, compare RMS
        in_rms = np.sqrt(np.mean(input_signal[200:] ** 2))
        out_rms = np.sqrt(np.mean(output[200:] ** 2))
        assert abs(out_rms / in_rms - 1.0) < 0.3  # within 30%


class TestOnePoleLP:
    def test_zero_damp_passes_through(self):
        """With damp=0, output should equal input."""
        lp = OnePoleLP()
        assert lp.process(0.5, 0.0) == pytest.approx(0.5)
        assert lp.process(1.0, 0.0) == pytest.approx(1.0)

    def test_high_damp_smooths(self):
        """With high damping, output should lag behind input."""
        lp = OnePoleLP()
        lp.process(0.0, 0.9)
        val = lp.process(1.0, 0.9)
        assert val < 0.2  # heavily smoothed


class TestDattorroPlateReverb:
    def test_dry_signal_passes_through(self, sample_rate):
        """With mix=0, output should equal input."""
        rev = DattorroPlateReverb(sample_rate)
        rev.mix = 0.0
        impulse = np.zeros(512, dtype=np.float32)
        impulse[0] = 1.0
        out = rev.process(impulse)
        np.testing.assert_allclose(out, impulse, atol=1e-6)

    def test_wet_adds_tail(self, sample_rate):
        """With mix>0, reverb should add energy after the impulse."""
        rev = DattorroPlateReverb(sample_rate)
        rev.mix = 0.5
        rev.decay = 0.8
        impulse = np.zeros(4096, dtype=np.float32)
        impulse[0] = 1.0
        out = rev.process(impulse)

        # Tail (last quarter) should have reverb energy
        tail = out[3072:]
        assert np.max(np.abs(tail)) > 0.001

    def test_silence_in_silence_out(self, sample_rate):
        """Silent input should produce silent output."""
        rev = DattorroPlateReverb(sample_rate)
        rev.mix = 0.5
        silence = np.zeros(1024, dtype=np.float32)
        out = rev.process(silence)
        assert np.max(np.abs(out)) < 1e-6

    def test_decay_controls_tail_length(self, sample_rate):
        """Higher decay should produce a longer reverb tail."""
        impulse = np.zeros(int(sample_rate * 2), dtype=np.float32)
        impulse[0] = 1.0

        rev_short = DattorroPlateReverb(sample_rate)
        rev_short.mix = 1.0
        rev_short.decay = 0.3
        out_short = rev_short.process(impulse)

        rev_long = DattorroPlateReverb(sample_rate)
        rev_long.mix = 1.0
        rev_long.decay = 0.9
        out_long = rev_long.process(impulse)

        # Compare energy in the last second
        tail_start = sample_rate
        energy_short = np.sum(out_short[tail_start:] ** 2)
        energy_long = np.sum(out_long[tail_start:] ** 2)
        assert energy_long > energy_short * 5

    def test_damping_reduces_brightness(self, sample_rate):
        """Higher damping should reduce high-frequency content in tail."""
        impulse = np.zeros(int(sample_rate * 1), dtype=np.float32)
        impulse[0] = 1.0

        rev_bright = DattorroPlateReverb(sample_rate)
        rev_bright.mix = 1.0
        rev_bright.decay = 0.85
        rev_bright.damping = 0.0
        out_bright = rev_bright.process(impulse)

        rev_dark = DattorroPlateReverb(sample_rate)
        rev_dark.mix = 1.0
        rev_dark.decay = 0.85
        rev_dark.damping = 0.9
        out_dark = rev_dark.process(impulse)

        # Compare HF energy in the tail
        tail = slice(sample_rate // 2, None)
        fft_bright = np.abs(np.fft.rfft(out_bright[tail]))
        fft_dark = np.abs(np.fft.rfft(out_dark[tail]))
        hf_band = len(fft_bright) // 2  # upper half of spectrum
        assert np.sum(fft_dark[hf_band:]) < np.sum(fft_bright[hf_band:])

    def test_output_bounded(self, sample_rate):
        """Output shouldn't explode even with high decay."""
        rev = DattorroPlateReverb(sample_rate)
        rev.mix = 0.5
        rev.decay = 0.95

        # Feed in a burst of noise
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(2048).astype(np.float32) * 0.5
        out = rev.process(noise)

        # Then let it ring
        silence = np.zeros(4096, dtype=np.float32)
        out2 = rev.process(silence)

        assert np.max(np.abs(out)) < 10.0
        assert np.max(np.abs(out2)) < 10.0

    def test_clear_resets_state(self, sample_rate):
        """After clear(), reverb should behave as freshly initialized."""
        rev = DattorroPlateReverb(sample_rate)
        rev.mix = 0.5
        rev.decay = 0.85

        # Feed some audio to build up state
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(2048).astype(np.float32) * 0.3
        rev.process(noise)

        # Clear and process silence — should be silent
        rev.clear()
        silence = np.zeros(1024, dtype=np.float32)
        out = rev.process(silence)
        assert np.max(np.abs(out)) < 1e-6

    def test_get_state(self, sample_rate):
        rev = DattorroPlateReverb(sample_rate)
        rev.mix = 0.4
        rev.decay = 0.75
        state = rev.get_state()
        assert state["mix"] == 0.4
        assert state["decay"] == 0.75
        assert "damping" in state
        assert "predelay_ms" in state

    def test_predelay_offsets_output(self, sample_rate):
        """Pre-delay should shift the wet signal in time."""
        impulse = np.zeros(4096, dtype=np.float32)
        impulse[0] = 1.0

        rev_no_pd = DattorroPlateReverb(sample_rate)
        rev_no_pd.mix = 1.0
        rev_no_pd.predelay_ms = 0.0
        rev_no_pd.decay = 0.7
        out_no_pd = rev_no_pd.process(impulse)

        rev_pd = DattorroPlateReverb(sample_rate)
        rev_pd.mix = 1.0
        rev_pd.predelay_ms = 50.0
        rev_pd.decay = 0.7
        out_pd = rev_pd.process(impulse)

        # First non-trivial output should appear later with predelay
        thresh = 0.001
        first_no_pd = np.argmax(np.abs(out_no_pd) > thresh)
        first_pd = np.argmax(np.abs(out_pd) > thresh)
        assert first_pd > first_no_pd + 100  # at least ~2ms later
