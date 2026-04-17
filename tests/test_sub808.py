"""Tests for the 808 Sub voice (Serum-style + Zay architecture)."""

import numpy as np
import pytest

from cypher.drum.sub808 import Sub808Voice


class TestSub808:
    def test_trigger_produces_sound(self, sample_rate):
        voice = Sub808Voice(sample_rate)
        voice.trigger(36, 0.9)
        output = voice.process(512)
        assert np.max(np.abs(output)) > 0.01

    def test_idle_is_silent(self, sample_rate):
        voice = Sub808Voice(sample_rate)
        output = voice.process(512)
        assert np.all(output == 0.0)
        assert not voice.is_active

    def test_velocity_scales_output(self, sample_rate):
        v1 = Sub808Voice(sample_rate)
        v1.trigger(36, 1.0)
        out_loud = v1.process(512)

        v2 = Sub808Voice(sample_rate)
        v2.trigger(36, 0.3)
        out_quiet = v2.process(512)

        assert np.max(np.abs(out_loud)) > np.max(np.abs(out_quiet))

    def test_sustains_while_held(self, sample_rate):
        """808 should sustain at full level while note is held (gated amp)."""
        voice = Sub808Voice(sample_rate)
        voice.params[1].value = 0.0  # No punch — flat tone
        voice.trigger(36, 0.9)

        # Render 500ms without releasing
        output = voice.process(int(0.5 * sample_rate))

        # Should still have energy at the end
        tail = output[-1024:]
        assert np.max(np.abs(tail)) > 0.5

    def test_release_triggers_tail(self, sample_rate):
        """After note-off, the 808 should decay to silence."""
        voice = Sub808Voice(sample_rate)
        voice.params[6].value = 0.2  # RELEASE (index 6)

        voice.trigger(36, 0.9)
        voice.process(int(0.1 * sample_rate))
        voice.release(36)

        output = voice.process(int(2.0 * sample_rate))
        assert output[-1] < 0.01

    def test_pitch_punch_creates_transient(self, sample_rate):
        """With PUNCH > 0, the initial samples should have higher pitch content."""
        # With punch
        v1 = Sub808Voice(sample_rate)
        v1.params[1].value = 0.6  # PUNCH: ~22 semitones
        v1.trigger(36, 0.9)
        out_punch = v1.process(int(0.05 * sample_rate))  # First 50ms

        # Without punch
        v2 = Sub808Voice(sample_rate)
        v2.params[1].value = 0.0  # No punch
        v2.trigger(36, 0.9)
        out_flat = v2.process(int(0.05 * sample_rate))

        # Punch sweeps from ~250Hz down to ~33Hz. Compare energy in 80-500Hz
        # band (above fundamental, where the sweep lives).
        fft_punch = np.abs(np.fft.rfft(out_punch))
        fft_flat = np.abs(np.fft.rfft(out_flat))
        freqs = np.fft.rfftfreq(len(out_punch), 1.0 / sample_rate)
        sweep_band = (freqs >= 80) & (freqs <= 500)

        assert np.sum(fft_punch[sweep_band]) > np.sum(fft_flat[sweep_band])

    def test_no_punch_is_pure_sub(self, sample_rate):
        """With PUNCH=0, TONE=0, DRIVE=0, should be a near-pure sine."""
        voice = Sub808Voice(sample_rate)
        voice.params[1].value = 0.0  # No punch
        voice.params[2].value = 0.0  # No tone
        voice.params[3].value = 0.0  # No drive
        voice.trigger(36, 0.9)
        output = voice.process(int(0.5 * sample_rate))

        fft = np.abs(np.fft.rfft(output))
        freqs = np.fft.rfftfreq(len(output), 1.0 / sample_rate)

        low_energy = np.sum(fft[freqs <= 200] ** 2)
        high_energy = np.sum(fft[freqs > 200] ** 2)
        ratio = high_energy / max(low_energy, 1e-10)

        assert ratio < 0.05

    def test_legato_glide(self, sample_rate):
        voice = Sub808Voice(sample_rate)
        voice.params[7].value = 0.5  # GLIDE (index 7)

        voice.trigger(36, 0.8)
        voice.process(int(0.3 * sample_rate))

        voice.trigger(31, 0.8)
        assert voice.is_active
        assert voice._pitch_glide.is_gliding

    def test_all_notes_off(self, sample_rate):
        voice = Sub808Voice(sample_rate)
        voice.trigger(36, 0.9)
        voice.process(256)
        voice.all_notes_off()

        output = voice.process(512)
        assert np.all(output == 0.0)
        assert not voice.is_active

    def test_get_state(self, sample_rate):
        voice = Sub808Voice(sample_rate)
        voice.trigger(36, 0.9)
        voice.process(256)

        state = voice.get_state()
        assert state["active"] is True
        assert state["amplitude"] > 0
        assert "current_pitch_hz" in state
        assert "pitch_mod_level" in state
        assert "amp_env_stage" in state
        assert "tone_amount" in state
        assert "noise_active" in state
        assert "sat_character" in state

    def test_params_layout(self, sample_rate):
        """12 params across 3 pages: simple (4) + advanced (4+4)."""
        voice = Sub808Voice(sample_rate)
        assert len(voice.params) == 12
        # Page 1 — Simple
        assert voice.params[0].label == "DECAY"
        assert voice.params[1].label == "PUNCH"
        assert voice.params[2].label == "TONE"
        assert voice.params[3].label == "DRIVE"
        # Page 2 — Advanced
        assert voice.params[4].label == "SHAPE"
        assert voice.params[5].label == "NOISE"
        assert voice.params[6].label == "RELEASE"
        assert voice.params[7].label == "GLIDE"
        # Page 3 — Advanced
        assert voice.params[8].label == "FILTER"
        assert voice.params[9].label == "RESO"
        assert voice.params[10].label == "SAT"
        assert voice.params[11].label == "P.SUST"

    def test_drive_adds_harmonics(self, sample_rate):
        clean = Sub808Voice(sample_rate)
        clean.params[1].value = 0.0  # No punch
        clean.params[2].value = 0.0  # No tone
        clean.params[3].value = 0.0  # No drive
        clean.trigger(36, 0.9)
        out_clean = clean.process(int(0.5 * sample_rate))

        driven = Sub808Voice(sample_rate)
        driven.params[1].value = 0.0
        driven.params[2].value = 0.0
        driven.params[3].value = 0.8  # DRIVE (index 3)
        driven.params[10].value = 0.25  # SAT: soft (must be on for drive to apply)
        driven.trigger(36, 0.9)
        out_driven = driven.process(int(0.5 * sample_rate))

        fft_clean = np.abs(np.fft.rfft(out_clean))
        fft_driven = np.abs(np.fft.rfft(out_driven))
        high_band = len(fft_clean) // 4

        assert np.sum(fft_driven[high_band:]) > np.sum(fft_clean[high_band:])

    def test_tone_adds_harmonics(self, sample_rate):
        """TONE knob should add harmonics via Chebyshev waveshaping."""
        clean = Sub808Voice(sample_rate)
        clean.params[1].value = 0.0
        clean.params[2].value = 0.0  # TONE: 0
        clean.params[3].value = 0.0
        clean.trigger(36, 0.9)
        out_clean = clean.process(int(0.5 * sample_rate))

        toned = Sub808Voice(sample_rate)
        toned.params[1].value = 0.0
        toned.params[2].value = 0.8  # TONE: high
        toned.params[3].value = 0.0
        toned.trigger(36, 0.9)
        out_toned = toned.process(int(0.5 * sample_rate))

        # TONE adds 2nd + 3rd harmonics. Compare harmonic-to-fundamental ratio.
        fft_clean = np.abs(np.fft.rfft(out_clean))
        fft_toned = np.abs(np.fft.rfft(out_toned))
        freqs = np.fft.rfftfreq(len(out_clean), 1.0 / sample_rate)

        fundamental = (freqs >= 20) & (freqs <= 50)
        harmonics = (freqs >= 50) & (freqs <= 200)  # 2nd + 3rd of ~33Hz

        ratio_clean = np.sum(fft_clean[harmonics]) / max(np.sum(fft_clean[fundamental]), 1e-10)
        ratio_toned = np.sum(fft_toned[harmonics]) / max(np.sum(fft_toned[fundamental]), 1e-10)

        assert ratio_toned > ratio_clean

    def test_noise_adds_brightness(self, sample_rate):
        """NOISE should add a burst of filtered noise on attack."""
        quiet = Sub808Voice(sample_rate)
        quiet.params[5].value = 0.0  # No noise
        quiet.trigger(36, 0.9)
        out_quiet = quiet.process(int(0.05 * sample_rate))

        noisy = Sub808Voice(sample_rate)
        noisy.params[5].value = 1.0  # Max noise (15%)
        noisy.trigger(36, 0.9)
        out_noisy = noisy.process(int(0.05 * sample_rate))

        fft_quiet = np.abs(np.fft.rfft(out_quiet))
        fft_noisy = np.abs(np.fft.rfft(out_noisy))
        high_band = len(fft_quiet) // 4

        assert np.sum(fft_noisy[high_band:]) > np.sum(fft_quiet[high_band:])

    def test_different_decay_times(self, sample_rate):
        """Faster decay = faster pitch drop = snappier transient."""
        fast = Sub808Voice(sample_rate)
        fast.params[0].value = 0.1   # Fast decay
        fast.params[1].value = 0.5   # Same punch
        fast.trigger(36, 0.9)
        out_fast = fast.process(int(0.1 * sample_rate))

        slow = Sub808Voice(sample_rate)
        slow.params[0].value = 0.8   # Slow decay
        slow.params[1].value = 0.5
        slow.trigger(36, 0.9)
        out_slow = slow.process(int(0.1 * sample_rate))

        # Both should produce sound
        assert np.max(np.abs(out_fast)) > 0.01
        assert np.max(np.abs(out_slow)) > 0.01

    def test_sat_type_characters(self, sample_rate):
        """Different saturation types should produce different timbres."""
        results = []
        for sat_val in [0.0, 0.4, 0.7, 1.0]:  # soft, tape, hard, crush
            v = Sub808Voice(sample_rate)
            v.params[1].value = 0.0
            v.params[3].value = 0.7   # DRIVE: high enough to hear sat
            v.params[10].value = sat_val  # SAT TYPE
            v.trigger(36, 0.9)
            out = v.process(int(0.3 * sample_rate))
            fft = np.abs(np.fft.rfft(out))
            results.append(np.sum(fft))

        # Different sat types should produce different spectral totals
        assert len(set(round(r, 2) for r in results)) > 1
