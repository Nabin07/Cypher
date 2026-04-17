"""Tests for LFO module."""

import numpy as np
import pytest

from cypher.core.lfo import LFO, LFO_SINE, LFO_TRI, LFO_SQUARE, LFO_SH


class TestLFO:
    def test_output_range(self, sample_rate):
        """LFO output should always be in [-1.0, +1.0]."""
        for wave in [LFO_SINE, LFO_TRI, LFO_SQUARE, LFO_SH]:
            lfo = LFO(sample_rate)
            lfo.wave = wave
            lfo.rate_hz = 5.0
            out = lfo.process(4096)
            assert np.max(out) <= 1.0 + 1e-6
            assert np.min(out) >= -1.0 - 1e-6

    def test_sine_is_smooth(self, sample_rate):
        """Sine LFO should have no discontinuities."""
        lfo = LFO(sample_rate)
        lfo.wave = LFO_SINE
        lfo.rate_hz = 2.0
        out = lfo.process(4096)

        # Check that adjacent samples don't jump too far
        diffs = np.abs(np.diff(out))
        assert np.max(diffs) < 0.01  # smooth transitions

    def test_square_is_bipolar(self, sample_rate):
        """Square LFO should only output +1 or -1."""
        lfo = LFO(sample_rate)
        lfo.wave = LFO_SQUARE
        lfo.rate_hz = 2.0
        out = lfo.process(4096)

        assert np.all((np.abs(out - 1.0) < 1e-6) | (np.abs(out + 1.0) < 1e-6))

    def test_triangle_reaches_extremes(self, sample_rate):
        """Triangle LFO should reach +1 and -1."""
        lfo = LFO(sample_rate)
        lfo.wave = LFO_TRI
        lfo.rate_hz = 2.0
        out = lfo.process(sample_rate)  # 1 second = 2 full cycles

        assert np.max(out) > 0.95
        assert np.min(out) < -0.95

    def test_rate_affects_frequency(self, sample_rate):
        """Higher rate should produce more zero crossings."""
        lfo_slow = LFO(sample_rate)
        lfo_slow.wave = LFO_SINE
        lfo_slow.rate_hz = 1.0
        out_slow = lfo_slow.process(sample_rate)

        lfo_fast = LFO(sample_rate)
        lfo_fast.wave = LFO_SINE
        lfo_fast.rate_hz = 5.0
        out_fast = lfo_fast.process(sample_rate)

        # Count zero crossings
        zc_slow = np.sum(np.diff(np.sign(out_slow)) != 0)
        zc_fast = np.sum(np.diff(np.sign(out_fast)) != 0)
        assert zc_fast > zc_slow * 3

    def test_sample_and_hold_steps(self, sample_rate):
        """S&H should hold values between phase wraps."""
        lfo = LFO(sample_rate)
        lfo.wave = LFO_SH
        lfo.rate_hz = 4.0  # 4 steps per second
        out = lfo.process(sample_rate)

        # Should have exactly ~4 distinct step levels per second
        # (allowing for float imprecision)
        diffs = np.abs(np.diff(out))
        step_changes = np.sum(diffs > 0.01)
        assert 2 <= step_changes <= 6  # roughly 4, with tolerance

    def test_reset_restarts_phase(self, sample_rate):
        """Reset should bring the phase back to 0."""
        lfo = LFO(sample_rate)
        lfo.wave = LFO_SINE
        lfo.rate_hz = 1.0

        out1_start = lfo.process(256)
        lfo.reset()
        out2_start = lfo.process(256)

        np.testing.assert_allclose(out1_start, out2_start, atol=1e-6)

    def test_phase_continuity_across_blocks(self, sample_rate):
        """Processing in blocks should produce the same result as one big block."""
        lfo_single = LFO(sample_rate)
        lfo_single.wave = LFO_SINE
        lfo_single.rate_hz = 3.0
        out_single = lfo_single.process(4096)

        lfo_multi = LFO(sample_rate)
        lfo_multi.wave = LFO_SINE
        lfo_multi.rate_hz = 3.0
        out_multi = np.concatenate([
            lfo_multi.process(1024),
            lfo_multi.process(1024),
            lfo_multi.process(1024),
            lfo_multi.process(1024),
        ])

        np.testing.assert_allclose(out_single, out_multi, atol=1e-5)
