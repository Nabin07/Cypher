"""Tests for oscillator primitives."""

import numpy as np
import pytest

from cypher.core.oscillator import MetallicOscillator, NoiseGenerator, SineOscillator, SquareOscillator


class TestSineOscillator:
    def test_produces_output(self, sample_rate):
        osc = SineOscillator(sample_rate)
        freq = np.full(512, 440.0, dtype=np.float32)
        output = osc.process(freq)
        assert len(output) == 512
        assert np.max(np.abs(output)) > 0.9

    def test_correct_frequency(self, sample_rate):
        osc = SineOscillator(sample_rate)
        duration = 1.0
        n = int(duration * sample_rate)
        freq = np.full(n, 440.0, dtype=np.float32)
        output = osc.process(freq)
        # FFT to check dominant frequency
        fft = np.abs(np.fft.rfft(output))
        freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
        peak_freq = freqs[np.argmax(fft)]
        assert peak_freq == pytest.approx(440.0, abs=2.0)

    def test_range_bounded(self, sample_rate):
        osc = SineOscillator(sample_rate)
        freq = np.full(4096, 440.0, dtype=np.float32)
        output = osc.process(freq)
        assert np.max(output) <= 1.001
        assert np.min(output) >= -1.001

    def test_phase_continuity(self, sample_rate):
        osc = SineOscillator(sample_rate)
        freq = np.full(256, 440.0, dtype=np.float32)
        out1 = osc.process(freq)
        out2 = osc.process(freq)
        # Check no discontinuity at boundary
        diff = abs(float(out2[0]) - float(out1[-1]))
        expected_diff = abs(float(out1[-1]) - float(out1[-2]))
        assert diff < expected_diff * 3  # Allow some tolerance


class TestNoiseGenerator:
    def test_produces_output(self):
        noise = NoiseGenerator()
        output = noise.process(1024)
        assert len(output) == 1024
        assert np.max(np.abs(output)) > 0.5

    def test_range_bounded(self):
        noise = NoiseGenerator()
        output = noise.process(10000)
        assert np.max(output) <= 1.0
        assert np.min(output) >= -1.0

    def test_roughly_zero_mean(self):
        noise = NoiseGenerator(seed=42)
        output = noise.process(100000)
        assert abs(np.mean(output)) < 0.01


class TestSquareOscillator:
    def test_produces_output(self, sample_rate):
        osc = SquareOscillator(sample_rate)
        output = osc.process_fixed(440.0, 512)
        assert len(output) == 512

    def test_binary_values(self, sample_rate):
        osc = SquareOscillator(sample_rate)
        output = osc.process_fixed(440.0, 1024)
        unique = np.unique(output)
        assert len(unique) == 2
        assert -1.0 in unique
        assert 1.0 in unique


class TestMetallicOscillator:
    def test_produces_output(self, sample_rate):
        osc = MetallicOscillator(sample_rate)
        output = osc.process_fixed(8000.0, 512)
        assert len(output) == 512
        assert np.max(np.abs(output)) > 0.1

    def test_range_bounded(self, sample_rate):
        osc = MetallicOscillator(sample_rate)
        output = osc.process_fixed(8000.0, 4096)
        assert np.max(output) <= 1.001
        assert np.min(output) >= -1.001
