"""Tests for Gross Beat volume preset library."""

import numpy as np
import pytest

from cypher.sampler.gross_beat import (
    GrossBeat, PRESET_NAMES,
    PRESET_OFF, PRESET_STUTTER, PRESET_GATE, PRESET_PUMP,
    PRESET_SWELL, PRESET_TREMOLO,
)


class TestGrossBeat:
    def test_off_is_passthrough(self, sample_rate):
        gb = GrossBeat(sample_rate)
        gb.preset = PRESET_OFF
        x = np.random.default_rng(0).standard_normal(1024).astype(np.float32)
        out = gb.process(x)
        np.testing.assert_array_equal(out, x)

    def test_stutter_creates_gaps(self, sample_rate):
        """STUTTER should zero most of the signal between chops."""
        gb = GrossBeat(sample_rate)
        gb.preset = PRESET_STUTTER
        gb.bpm = 120.0
        x = np.ones(sample_rate, dtype=np.float32)  # 1 second of 1.0
        out = gb.process(x)
        # Plenty of samples should be near-zero (between stutter chops)
        near_zero = np.sum(np.abs(out) < 0.1)
        assert near_zero > len(out) * 0.4

    def test_gate_is_binary(self, sample_rate):
        gb = GrossBeat(sample_rate)
        gb.preset = PRESET_GATE
        x = np.ones(sample_rate // 4, dtype=np.float32)
        out = gb.process(x)
        # GATE should be either 0 or 1 (binary) — allow a tiny tolerance
        for v in out:
            assert abs(v) < 1e-5 or abs(v - 1.0) < 1e-5

    def test_tremolo_is_smooth(self, sample_rate):
        gb = GrossBeat(sample_rate)
        gb.preset = PRESET_TREMOLO
        x = np.ones(sample_rate, dtype=np.float32)
        out = gb.process(x)
        # Tremolo should vary but smoothly — no huge jumps
        diffs = np.abs(np.diff(out))
        assert np.max(diffs) < 0.1

    def test_all_presets_run_without_error(self, sample_rate):
        x = np.ones(512, dtype=np.float32) * 0.5
        for p in range(len(PRESET_NAMES)):
            gb = GrossBeat(sample_rate)
            gb.preset = p
            out = gb.process(x)
            assert len(out) == len(x)
            # All presets should output within [-1, 1] approximately
            assert np.max(np.abs(out)) <= 1.5

    def test_preset_name_lookup(self, sample_rate):
        gb = GrossBeat(sample_rate)
        gb.preset = PRESET_SWELL
        assert gb.preset_name == "SWELL"

    def test_phase_continues_across_blocks(self, sample_rate):
        """Processing two back-to-back blocks should give the same result
        as processing one long block."""
        gb1 = GrossBeat(sample_rate)
        gb1.preset = PRESET_TREMOLO
        x = np.ones(2048, dtype=np.float32)
        out1 = gb1.process(x)

        gb2 = GrossBeat(sample_rate)
        gb2.preset = PRESET_TREMOLO
        a = gb2.process(x[:1024])
        b = gb2.process(x[1024:])
        out2 = np.concatenate([a, b])

        # Block-rate updates won't match sample-perfectly for ramp-smoothed
        # presets, but the overall shape must be close.
        err = np.mean(np.abs(out1 - out2))
        assert err < 0.2
