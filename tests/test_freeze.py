"""Tests for Granular Freeze."""

import numpy as np
import pytest

from cypher.sampler.freeze import (
    FreezeState, FreezeProcessor,
    MOTION_HOLD, MOTION_DRIFT, MOTION_OSCILLATE, MOTION_DECAY, MOTION_NAMES,
)


def _impulsey_source(length=48000) -> np.ndarray:
    """Source with visible features for verification."""
    t = np.arange(length) / 48000
    return (np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestFreeze:
    def test_hold_continues_indefinitely(self, sample_rate):
        src = _impulsey_source(length=sample_rate)
        st = FreezeState(
            source=src, position_frac=0.5, grain_size=2048,
            motion=MOTION_HOLD, rate=0.0, depth=0.0,
            sample_rate=sample_rate,
        )
        fp = FreezeProcessor(st, sample_rate)
        out = fp.process(sample_rate * 2)  # 2 seconds
        # HOLD should keep producing audio the whole time
        assert np.max(np.abs(out)) > 0.01
        # Energy should be consistent across the buffer (within ±50%)
        half = len(out) // 2
        e1 = np.sum(out[:half] ** 2)
        e2 = np.sum(out[half:] ** 2)
        assert 0.5 < e1 / e2 < 2.0

    def test_decay_fades_to_silence(self, sample_rate):
        src = _impulsey_source(length=sample_rate)
        st = FreezeState(
            source=src, position_frac=0.3, grain_size=2048,
            motion=MOTION_DECAY, rate=0.2, depth=0.0,
            sample_rate=sample_rate,
        )
        fp = FreezeProcessor(st, sample_rate)
        out = fp.process(int(sample_rate * 0.5))  # past decay time
        # Last few samples should be near silent
        tail = out[-256:]
        assert np.max(np.abs(tail)) < 0.05

    def test_oscillate_moves_position(self, sample_rate):
        src = _impulsey_source(length=sample_rate)
        st = FreezeState(
            source=src, position_frac=0.5, grain_size=2048,
            motion=MOTION_OSCILLATE, rate=1.0, depth=0.2,
            sample_rate=sample_rate,
        )
        fp = FreezeProcessor(st, sample_rate)
        # After some time, position should have moved away from position0
        fp.process(sample_rate // 4)  # 0.25s
        # Position oscillates around position0, won't equal position0 at arbitrary time
        assert abs(st.position - st.position0) > 0  # moved at all

    def test_drift_advances_forward(self, sample_rate):
        src = _impulsey_source(length=sample_rate)
        st = FreezeState(
            source=src, position_frac=0.2, grain_size=2048,
            motion=MOTION_DRIFT, rate=0.2, depth=0.0,
            sample_rate=sample_rate,
        )
        initial_pos = st.position
        fp = FreezeProcessor(st, sample_rate)
        fp.process(sample_rate // 2)
        assert st.position > initial_pos

    def test_short_source_does_not_crash(self, sample_rate):
        src = np.ones(100, dtype=np.float32) * 0.5
        st = FreezeState(
            source=src, position_frac=0.5, grain_size=8192,
            motion=MOTION_HOLD, rate=0.0, depth=0.0,
            sample_rate=sample_rate,
        )
        fp = FreezeProcessor(st, sample_rate)
        out = fp.process(512)  # should not crash
        assert len(out) == 512

    def test_crossfade_has_no_hard_seams(self, sample_rate):
        """Loop boundary should be smooth thanks to crossfade — no big jumps."""
        src = _impulsey_source(length=sample_rate)
        st = FreezeState(
            source=src, position_frac=0.5, grain_size=4096,
            motion=MOTION_HOLD, rate=0.0, depth=0.0,
            sample_rate=sample_rate,
        )
        fp = FreezeProcessor(st, sample_rate)
        out = fp.process(sample_rate)  # 1 second
        diffs = np.abs(np.diff(out))
        # No single-sample jump bigger than 0.2 (sine adjacent samples <0.06)
        assert np.max(diffs) < 0.2
