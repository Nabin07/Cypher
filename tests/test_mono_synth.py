"""Tests for MonoSynthVoice — dual-oscillator subtractive synth."""

import numpy as np
import pytest

from cypher.synth.mono import MonoSynthVoice, WAVE_SAW, WAVE_SINE, WAVE_SQUARE, WAVE_TRI


class TestMonoSynth:
    def test_trigger_produces_sound(self, sample_rate):
        v = MonoSynthVoice(sample_rate)
        v.trigger(60, 0.9)
        out = v.process(512)
        assert np.max(np.abs(out)) > 0.01

    def test_idle_is_silent(self, sample_rate):
        v = MonoSynthVoice(sample_rate)
        out = v.process(512)
        assert np.all(out == 0.0)

    def test_velocity_scales_output(self, sample_rate):
        v1 = MonoSynthVoice(sample_rate)
        v1.trigger(60, 1.0)
        v1.process(2048)
        out_loud = v1.process(512)

        v2 = MonoSynthVoice(sample_rate)
        v2.trigger(60, 0.3)
        v2.process(2048)
        out_quiet = v2.process(512)

        assert np.max(np.abs(out_loud)) > np.max(np.abs(out_quiet))

    # --- Oscillator A waveforms ---

    def test_saw_wave(self, sample_rate):
        v = MonoSynthVoice(sample_rate)
        v.params[0].value = 0.0  # WAVE A: SAW
        v.trigger(60, 0.9)
        out = v.process(512)
        assert np.max(np.abs(out)) > 0.01

    def test_sine_wave(self, sample_rate):
        v = MonoSynthVoice(sample_rate)
        v.params[0].value = 0.33  # WAVE A: SIN
        v.trigger(60, 0.9)
        out = v.process(512)
        assert np.max(np.abs(out)) > 0.01

    def test_square_wave(self, sample_rate):
        v = MonoSynthVoice(sample_rate)
        v.params[0].value = 0.67  # WAVE A: SQR
        v.trigger(60, 0.9)
        out = v.process(512)
        assert np.max(np.abs(out)) > 0.01

    def test_triangle_wave(self, sample_rate):
        v = MonoSynthVoice(sample_rate)
        v.params[0].value = 1.0  # WAVE A: TRI
        v.trigger(60, 0.9)
        out = v.process(512)
        assert np.max(np.abs(out)) > 0.01

    def test_different_waves_sound_different(self, sample_rate):
        """Each waveform should produce a different spectral signature."""
        spectra = []
        for wave_val in [0.0, 0.33, 0.67, 1.0]:  # SAW, SIN, SQR, TRI
            v = MonoSynthVoice(sample_rate)
            v.params[0].value = wave_val
            v.params[4].value = 1.0  # CUTOFF max
            v.trigger(60, 0.9)
            v.process(512)
            out = v.process(2048)
            spectra.append(np.abs(np.fft.rfft(out)))

        diffs = []
        for i in range(len(spectra)):
            for j in range(i + 1, len(spectra)):
                diffs.append(np.sum(np.abs(spectra[i] - spectra[j])))
        assert max(diffs) > 1.0

    # --- Dual oscillator ---

    def test_osc_b_adds_sound(self, sample_rate):
        """With MIX > 0, OSC B should contribute to the output."""
        v_a_only = MonoSynthVoice(sample_rate)
        v_a_only.params[2].value = 0.0  # MIX: all A
        v_a_only.params[4].value = 1.0  # CUTOFF max
        v_a_only.trigger(60, 0.9)
        v_a_only.process(512)
        out_a = v_a_only.process(2048)

        v_mixed = MonoSynthVoice(sample_rate)
        v_mixed.params[0].value = 0.0   # WAVE A: SAW
        v_mixed.params[1].value = 0.33  # WAVE B: SIN
        v_mixed.params[2].value = 0.5   # MIX: 50/50
        v_mixed.params[4].value = 1.0
        v_mixed.trigger(60, 0.9)
        v_mixed.process(512)
        out_mixed = v_mixed.process(2048)

        # Mixed output should differ from A-only
        correlation = np.corrcoef(out_a, out_mixed)[0, 1]
        assert correlation < 0.99

    def test_detune_between_oscillators(self, sample_rate):
        """Detune should create beating between OSC A and OSC B."""
        v = MonoSynthVoice(sample_rate)
        v.params[0].value = 0.0   # WAVE A: SAW
        v.params[1].value = 0.0   # WAVE B: SAW
        v.params[2].value = 0.5   # MIX: 50/50
        v.params[3].value = 0.8   # DETUNE: ~40 cents
        v.params[4].value = 1.0   # CUTOFF max
        v.trigger(60, 0.9)
        v.process(512)
        out = v.process(4096)

        # Detuned dual saws should show amplitude modulation (beating)
        envelope = np.abs(out)
        env_range = np.max(envelope) - np.min(envelope)
        assert env_range > 0.05

    def test_mix_zero_is_osc_a_only(self, sample_rate):
        """MIX at 0% should output only OSC A."""
        v = MonoSynthVoice(sample_rate)
        v.params[0].value = 0.33  # WAVE A: SIN
        v.params[1].value = 0.0   # WAVE B: SAW (different)
        v.params[2].value = 0.0   # MIX: 0% (all A)
        v.params[4].value = 1.0   # CUTOFF max
        v.trigger(60, 0.9)
        v.process(512)
        out = v.process(2048)

        # Should sound like pure sine — check low harmonic content
        fft = np.abs(np.fft.rfft(out))
        fundamental_bin = int(round(261.6 * 2048 / sample_rate))
        fundamental_energy = np.sum(fft[max(0, fundamental_bin-2):fundamental_bin+3])
        total_energy = np.sum(fft)
        assert fundamental_energy / total_energy > 0.3

    # --- Multimode filter ---

    def test_lowpass_cuts_highs(self, sample_rate):
        v_bright = MonoSynthVoice(sample_rate)
        v_bright.params[4].value = 1.0  # CUTOFF: max
        v_bright.params[7].value = 0.0  # MODE: LP
        v_bright.trigger(60, 0.9)
        v_bright.process(512)
        out_bright = v_bright.process(2048)

        v_dark = MonoSynthVoice(sample_rate)
        v_dark.params[4].value = 0.15  # CUTOFF: low
        v_dark.params[7].value = 0.0   # MODE: LP
        v_dark.trigger(60, 0.9)
        v_dark.process(512)
        out_dark = v_dark.process(2048)

        fft_bright = np.abs(np.fft.rfft(out_bright))
        fft_dark = np.abs(np.fft.rfft(out_dark))
        high_band = len(fft_bright) // 4
        assert np.sum(fft_bright[high_band:]) > np.sum(fft_dark[high_band:])

    def test_highpass_cuts_lows(self, sample_rate):
        """HP mode should remove low-frequency content."""
        v_lp = MonoSynthVoice(sample_rate)
        v_lp.params[4].value = 0.5  # CUTOFF: mid
        v_lp.params[7].value = 0.0  # MODE: LP
        v_lp.trigger(48, 0.9)
        v_lp.process(512)
        out_lp = v_lp.process(2048)

        v_hp = MonoSynthVoice(sample_rate)
        v_hp.params[4].value = 0.5  # CUTOFF: mid
        v_hp.params[7].value = 0.5  # MODE: HP
        v_hp.trigger(48, 0.9)
        v_hp.process(512)
        out_hp = v_hp.process(2048)

        fft_lp = np.abs(np.fft.rfft(out_lp))
        fft_hp = np.abs(np.fft.rfft(out_hp))
        low_band = len(fft_lp) // 8
        assert np.sum(fft_hp[:low_band]) < np.sum(fft_lp[:low_band])

    def test_bandpass_narrows_spectrum(self, sample_rate):
        """BP mode should attenuate both low and high extremes vs LP."""
        v_lp = MonoSynthVoice(sample_rate)
        v_lp.params[4].value = 0.5
        v_lp.params[7].value = 0.0  # LP
        v_lp.trigger(60, 0.9)
        v_lp.process(512)
        out_lp = v_lp.process(2048)

        v_bp = MonoSynthVoice(sample_rate)
        v_bp.params[4].value = 0.5
        v_bp.params[7].value = 1.0  # BP
        v_bp.trigger(60, 0.9)
        v_bp.process(512)
        out_bp = v_bp.process(2048)

        fft_lp = np.abs(np.fft.rfft(out_lp))
        fft_bp = np.abs(np.fft.rfft(out_bp))
        # BP should attenuate the very lowest frequencies that LP passes
        # Use first 8 bins (well below cutoff) to avoid BP resonance peak
        assert np.sum(fft_bp[:8]) < np.sum(fft_lp[:8])

    # --- Filter envelope ---

    def test_filter_env_creates_pluck(self, sample_rate):
        """Positive F.ENV should make the early sound brighter than later."""
        v = MonoSynthVoice(sample_rate)
        v.params[4].value = 0.3   # CUTOFF: lowish
        v.params[6].value = 1.0   # F.ENV: max positive
        v.params[8].value = 0.01  # ATTACK: fast
        v.params[9].value = 0.4   # DECAY
        v.params[10].value = 0.8  # SUSTAIN
        v.trigger(60, 0.9)

        v.process(128)
        out_early = v.process(1024)
        v.process(int(0.5 * sample_rate))
        out_late = v.process(1024)

        fft_early = np.abs(np.fft.rfft(out_early))
        fft_late = np.abs(np.fft.rfft(out_late))
        high_band = len(fft_early) // 4
        assert np.sum(fft_early[high_band:]) > np.sum(fft_late[high_band:])

    # --- Noise ---

    def test_noise_adds_content(self, sample_rate):
        v_clean = MonoSynthVoice(sample_rate)
        v_clean.params[15].value = 0.0  # NOISE: 0
        v_clean.params[4].value = 1.0
        v_clean.trigger(60, 0.9)
        out_clean = v_clean.process(2048)

        v_noisy = MonoSynthVoice(sample_rate)
        v_noisy.params[15].value = 0.8  # NOISE: 80%
        v_noisy.params[4].value = 1.0
        v_noisy.trigger(60, 0.9)
        out_noisy = v_noisy.process(2048)

        fft_clean = np.abs(np.fft.rfft(out_clean))
        fft_noisy = np.abs(np.fft.rfft(out_noisy))
        high_band = len(fft_clean) // 2
        assert np.sum(fft_noisy[high_band:]) > np.sum(fft_clean[high_band:]) * 0.8

    # --- LFO ---

    def test_lfo_to_filter_modulates_brightness(self, sample_rate):
        """LFO routed to filter should create cyclic brightness changes."""
        v = MonoSynthVoice(sample_rate)
        v.params[4].value = 0.4   # CUTOFF: mid-low
        v.params[12].value = 0.5  # LFO RATE: ~2Hz
        v.params[13].value = 0.8  # LFO DEPTH: heavy
        v.params[14].value = 0.0  # LFO DEST: filter
        v.params[10].value = 1.0  # SUSTAIN: full
        v.trigger(60, 0.9)
        v.process(512)

        out = v.process(int(1.0 * sample_rate))

        quarter = len(out) // 4
        fft_q1 = np.abs(np.fft.rfft(out[:quarter]))
        fft_q2 = np.abs(np.fft.rfft(out[quarter:2*quarter]))
        high_band = len(fft_q1) // 4
        diff = abs(np.sum(fft_q1[high_band:]) - np.sum(fft_q2[high_band:]))
        assert diff > 0.1

    def test_lfo_to_pitch_modulates_frequency(self, sample_rate):
        """LFO routed to pitch should create vibrato."""
        v = MonoSynthVoice(sample_rate)
        v.params[4].value = 1.0   # CUTOFF: max
        v.params[12].value = 0.6  # LFO RATE: ~3Hz
        v.params[13].value = 0.6  # LFO DEPTH: moderate
        v.params[14].value = 1.0  # LFO DEST: pitch
        v.params[10].value = 1.0  # SUSTAIN
        v.trigger(69, 0.9)  # A4 = 440Hz
        v.process(512)

        out = v.process(int(1.0 * sample_rate))

        fft = np.abs(np.fft.rfft(out))
        fundamental_bin = int(round(440.0 * len(out) / sample_rate))
        narrow = np.sum(fft[max(0, fundamental_bin-2):fundamental_bin+3])
        wide = np.sum(fft[max(0, fundamental_bin-15):fundamental_bin+16])
        assert wide > narrow * 1.1

    def test_lfo_zero_depth_no_effect(self, sample_rate):
        """LFO at depth 0 should have no audible effect."""
        v1 = MonoSynthVoice(sample_rate)
        v1.params[13].value = 0.0  # LFO DEPTH: 0
        v1.params[4].value = 1.0
        v1.params[10].value = 1.0
        v1.trigger(60, 0.9)
        v1.process(2048)
        out1 = v1.process(2048)

        v2 = MonoSynthVoice(sample_rate)
        v2.params[13].value = 0.0
        v2.params[4].value = 1.0
        v2.params[10].value = 1.0
        v2.trigger(60, 0.9)
        v2.process(2048)
        out2 = v2.process(2048)

        np.testing.assert_allclose(out1, out2, atol=1e-6)

    # --- Envelope / lifecycle ---

    def test_release_decays(self, sample_rate):
        v = MonoSynthVoice(sample_rate)
        v.trigger(60, 0.9)
        v.process(1024)

        v.release(60)
        out = v.process(int(2.0 * sample_rate))
        tail = out[-512:]
        assert np.max(np.abs(tail)) < 0.01

    def test_all_notes_off(self, sample_rate):
        v = MonoSynthVoice(sample_rate)
        v.trigger(60, 0.9)
        v.process(512)
        assert v.is_active

        v.all_notes_off()
        out = v.process(512)
        assert not v.is_active
        assert np.all(out == 0.0)

    # --- Params layout ---

    def test_params_layout(self, sample_rate):
        v = MonoSynthVoice(sample_rate)
        assert len(v.params) == 16
        # Page 1: OSC
        assert v.params[0].label == "WAVE A"
        assert v.params[1].label == "WAVE B"
        assert v.params[2].label == "MIX"
        assert v.params[3].label == "DETUNE"
        # Page 2: FILTER
        assert v.params[4].label == "CUTOFF"
        assert v.params[5].label == "RESO"
        assert v.params[6].label == "F.ENV"
        assert v.params[7].label == "MODE"
        # Page 3: AMP
        assert v.params[8].label == "ATTACK"
        assert v.params[9].label == "DECAY"
        assert v.params[10].label == "SUSTAIN"
        assert v.params[11].label == "RELEASE"
        # Page 4: MOD
        assert v.params[12].label == "LFO RATE"
        assert v.params[13].label == "LFO DEPTH"
        assert v.params[14].label == "LFO DEST"
        assert v.params[15].label == "NOISE"

    def test_get_state(self, sample_rate):
        v = MonoSynthVoice(sample_rate)
        v.trigger(60, 0.9)
        v.process(256)

        state = v.get_state()
        assert state["active"] is True
        assert state["wave_a"] == "SAW"
        assert state["wave_b"] == "SAW"
        assert state["note"] == 60
        assert "freq_hz" in state
        assert "amp_env_stage" in state
        assert "filter_env_stage" in state
        assert "filter_mode" in state
        assert "lfo_dest" in state

    def test_wave_name_properties(self, sample_rate):
        v = MonoSynthVoice(sample_rate)
        assert v.wave_a_name == "SAW"
        assert v.wave_b_name == "SAW"
        v.params[0].value = 0.33  # SIN
        v.params[1].value = 1.0   # TRI
        v._update_params()
        assert v.wave_a_name == "SIN"
        assert v.wave_b_name == "TRI"
