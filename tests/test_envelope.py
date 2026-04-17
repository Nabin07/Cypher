"""Tests for envelope generators."""

import numpy as np
import pytest

from cypher.core.envelope import ADEnvelope, PitchEnvelope


class TestADEnvelope:
    def test_idle_produces_silence(self, sample_rate):
        env = ADEnvelope(sample_rate)
        output = env.process(512)
        assert np.all(output == 0.0)
        assert not env.is_active

    def test_trigger_produces_sound(self, sample_rate):
        env = ADEnvelope(sample_rate)
        env.trigger()
        output = env.process(512)
        assert np.max(output) > 0.0
        assert env.is_active

    def test_decay_reaches_zero(self, sample_rate):
        env = ADEnvelope(sample_rate)
        env.decay_time = 0.05  # 50ms
        env.trigger()
        # Render enough to fully decay
        output = env.process(int(0.2 * sample_rate))
        # Last samples should be near zero
        assert output[-1] < 0.01

    def test_attack_reaches_peak(self, sample_rate):
        env = ADEnvelope(sample_rate)
        env.attack_time = 0.01  # 10ms
        env.trigger()
        # Render through attack
        output = env.process(int(0.02 * sample_rate))
        assert np.max(output) > 0.9

    def test_gate_mode_sustains(self, sample_rate):
        env = ADEnvelope(sample_rate)
        env.sustain_level = 0.8
        env.decay_time = 0.5
        env.trigger()
        # Render through attack
        output = env.process(int(0.1 * sample_rate))
        # Should be sustaining at 0.8
        assert output[-1] == pytest.approx(0.8, abs=0.05)

    def test_gate_release_decays(self, sample_rate):
        env = ADEnvelope(sample_rate)
        env.sustain_level = 0.8
        env.decay_time = 0.1
        env.trigger()
        env.process(int(0.05 * sample_rate))  # Sustain a bit
        env.release()
        output = env.process(int(0.5 * sample_rate))
        assert output[-1] < 0.01


class TestPitchEnvelope:
    def test_sweep_starts_high(self, sample_rate):
        pe = PitchEnvelope(sample_rate)
        pe.start_hz = 400.0
        pe.end_hz = 50.0
        pe.slide_time = 0.05
        pe.trigger(50.0)
        output = pe.process(10)
        # First sample should be near start frequency
        assert output[0] > 200.0

    def test_sweep_settles_at_target(self, sample_rate):
        pe = PitchEnvelope(sample_rate)
        pe.start_hz = 400.0
        pe.end_hz = 50.0
        pe.slide_time = 0.05
        pe.trigger(50.0)
        output = pe.process(int(0.2 * sample_rate))
        assert output[-1] == pytest.approx(50.0, rel=0.05)

    def test_glide_transitions(self, sample_rate):
        pe = PitchEnvelope(sample_rate)
        pe.start_hz = 100.0
        pe.end_hz = 50.0
        pe.glide_time = 0.1
        pe.trigger(50.0)
        pe.process(int(0.2 * sample_rate))  # Settle at 50Hz

        pe.glide_to(80.0)
        output = pe.process(int(0.3 * sample_rate))
        assert output[-1] == pytest.approx(80.0, rel=0.05)

    def test_glide_is_smooth(self, sample_rate):
        pe = PitchEnvelope(sample_rate)
        pe.glide_time = 0.1
        pe.end_hz = 50.0
        pe.trigger(50.0)
        pe.process(int(0.2 * sample_rate))

        pe.glide_to(100.0)
        output = pe.process(int(0.05 * sample_rate))
        # Should be monotonically increasing
        diffs = np.diff(output)
        assert np.all(diffs >= -0.01)  # Allow tiny float errors
