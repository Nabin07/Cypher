"""Tests for the Kick voice — sine + pitch envelope with link mode."""

import numpy as np
import pytest

from cypher.drum.kick import KickVoice


class TestKick:
    def test_trigger_produces_sound(self, sample_rate):
        voice = KickVoice(sample_rate)
        voice.trigger(36, 0.9)
        output = voice.process(512)
        assert np.max(np.abs(output)) > 0.01

    def test_idle_is_silent(self, sample_rate):
        voice = KickVoice(sample_rate)
        output = voice.process(512)
        assert np.all(output == 0.0)
        assert not voice.is_active

    def test_one_shot_decays(self, sample_rate):
        """Kick should decay to silence on its own."""
        voice = KickVoice(sample_rate)
        voice.trigger(36, 0.9)
        output = voice.process(int(3.0 * sample_rate))
        assert not voice.is_active

    def test_velocity_scales_output(self, sample_rate):
        v1 = KickVoice(sample_rate)
        v1.trigger(36, 1.0)
        out_loud = v1.process(512)

        v2 = KickVoice(sample_rate)
        v2.trigger(36, 0.3)
        out_quiet = v2.process(512)

        assert np.max(np.abs(out_loud)) > np.max(np.abs(out_quiet))

    def test_punch_adds_pitch_sweep(self, sample_rate):
        """More PUNCH = more pitch sweep = more high-freq energy in transient."""
        punchy = KickVoice(sample_rate)
        punchy.params[0].value = 0.8  # PUNCH: big sweep
        punchy.trigger(36, 0.9)
        out_punchy = punchy.process(int(0.02 * sample_rate))

        soft = KickVoice(sample_rate)
        soft.params[0].value = 0.0  # PUNCH: minimal sweep
        soft.trigger(36, 0.9)
        out_soft = soft.process(int(0.02 * sample_rate))

        fft_punchy = np.abs(np.fft.rfft(out_punchy))
        fft_soft = np.abs(np.fft.rfft(out_soft))
        freqs = np.fft.rfftfreq(len(out_punchy), 1.0 / sample_rate)
        sweep_band = (freqs >= 80) & (freqs <= 500)

        assert np.sum(fft_punchy[sweep_band]) > np.sum(fft_soft[sweep_band])

    def test_body_controls_decay(self, sample_rate):
        """Longer BODY = more sustained energy."""
        long = KickVoice(sample_rate)
        long.params[1].value = 0.9  # BODY: long decay
        long.trigger(36, 0.9)
        out_long = long.process(int(0.3 * sample_rate))

        short = KickVoice(sample_rate)
        short.params[1].value = 0.1  # BODY: short decay
        short.trigger(36, 0.9)
        out_short = short.process(int(0.3 * sample_rate))

        # Long body should have more energy in the tail
        tail = slice(int(0.1 * sample_rate), None)
        assert np.sum(np.abs(out_long[tail])) > np.sum(np.abs(out_short[tail]))

    def test_tone_filters_highs(self, sample_rate):
        """Low TONE = less high-frequency content. Use click for HF source."""
        dark = KickVoice(sample_rate)
        dark.params[2].value = 0.0  # TONE: dark (200Hz LPF)
        dark.params[4].value = 0.8  # CLICK: high — gives HF for filter to cut
        dark.trigger(36, 0.9)
        out_dark = dark.process(int(0.02 * sample_rate))

        bright = KickVoice(sample_rate)
        bright.params[2].value = 1.0  # TONE: bright (12kHz LPF)
        bright.params[4].value = 0.8  # CLICK: same
        bright.trigger(36, 0.9)
        out_bright = bright.process(int(0.02 * sample_rate))

        fft_dark = np.abs(np.fft.rfft(out_dark))
        fft_bright = np.abs(np.fft.rfft(out_bright))
        freqs = np.fft.rfftfreq(len(out_dark), 1.0 / sample_rate)
        high_band = freqs >= 1000

        assert np.sum(fft_bright[high_band]) > np.sum(fft_dark[high_band])

    def test_click_adds_noise_transient(self, sample_rate):
        """CLICK adds a noise burst for acoustic character."""
        no_click = KickVoice(sample_rate)
        no_click.params[4].value = 0.0  # CLICK: off
        no_click.trigger(36, 0.9)
        out_clean = no_click.process(int(0.01 * sample_rate))

        with_click = KickVoice(sample_rate)
        with_click.params[4].value = 1.0  # CLICK: max
        with_click.trigger(36, 0.9)
        out_click = with_click.process(int(0.01 * sample_rate))

        fft_clean = np.abs(np.fft.rfft(out_clean))
        fft_click = np.abs(np.fft.rfft(out_click))
        high_band = len(fft_clean) // 4

        assert np.sum(fft_click[high_band:]) > np.sum(fft_clean[high_band:])

    def test_pair_808_highpasses(self, sample_rate):
        """Paired kick has less sub content."""
        standalone = KickVoice(sample_rate)
        standalone.trigger(36, 0.9)
        out_standalone = standalone.process(int(0.1 * sample_rate))

        paired = KickVoice(sample_rate)
        paired.pair_808(32.7)  # Pair with 808 at C1
        paired.trigger(36, 0.9)
        out_paired = paired.process(int(0.1 * sample_rate))

        fft_standalone = np.abs(np.fft.rfft(out_standalone))
        fft_paired = np.abs(np.fft.rfft(out_paired))
        freqs = np.fft.rfftfreq(len(out_standalone), 1.0 / sample_rate)
        sub_band = freqs <= 80

        assert np.sum(fft_paired[sub_band]) < np.sum(fft_standalone[sub_band])

    def test_unpair_808_restores_sound(self, sample_rate):
        """Unpairing should restore the original kick sound."""
        voice = KickVoice(sample_rate)
        voice.pair_808(32.7)
        assert voice._paired is True
        voice.unpair_808()
        assert voice._paired is False
        assert voice._get_hpf_freq() == 0.0

    def test_pair_808_returns_info(self, sample_rate):
        """pair_808() returns a dict with HPF details."""
        voice = KickVoice(sample_rate)
        result = voice.pair_808(32.7)
        assert result["paired"] is True
        assert result["808_freq"] == 32.7
        assert result["hpf_freq"] > 32.7  # HPF sits above fundamental
        assert result["hpf_freq"] <= 150.0  # Capped

    def test_all_notes_off(self, sample_rate):
        voice = KickVoice(sample_rate)
        voice.trigger(36, 0.9)
        voice.process(256)
        voice.all_notes_off()

        output = voice.process(512)
        assert np.all(output == 0.0)
        assert not voice.is_active

    def test_params_layout(self, sample_rate):
        """8 params across 2 pages."""
        voice = KickVoice(sample_rate)
        assert len(voice.params) == 8
        # Page 1
        assert voice.params[0].label == "PUNCH"
        assert voice.params[1].label == "BODY"
        assert voice.params[2].label == "TONE"
        assert voice.params[3].label == "DRIVE"
        # Page 2
        assert voice.params[4].label == "CLICK"
        assert voice.params[5].label == "HOLD"
        assert voice.params[6].label == "ATTACK"
        assert voice.params[7].label == "CRUSH"

    def test_get_state(self, sample_rate):
        voice = KickVoice(sample_rate)
        voice.trigger(36, 0.9)
        voice.process(256)

        state = voice.get_state()
        assert state["active"] is True
        assert state["amplitude"] > 0
        assert "body_pitch_hz" in state
        assert "paired" in state
        assert "hpf_freq" in state

    def test_attack_param_controls_transient(self, sample_rate):
        """Shorter ATTACK = faster transient = more energy in first few ms."""
        fast = KickVoice(sample_rate)
        fast.params[6].value = 0.0  # ATTACK: minimum (0.5ms)
        fast.trigger(36, 0.9)
        out_fast = fast.process(int(0.003 * sample_rate))  # first 3ms

        slow = KickVoice(sample_rate)
        slow.params[6].value = 1.0  # ATTACK: maximum (20ms)
        slow.trigger(36, 0.9)
        out_slow = slow.process(int(0.003 * sample_rate))  # first 3ms

        # Fast attack should have more energy in the first 3ms
        assert np.sum(np.abs(out_fast)) > np.sum(np.abs(out_slow))

    def test_drive_adds_harmonics(self, sample_rate):
        clean = KickVoice(sample_rate)
        clean.params[3].value = 0.0
        clean.trigger(36, 0.9)
        out_clean = clean.process(int(0.1 * sample_rate))

        driven = KickVoice(sample_rate)
        driven.params[3].value = 0.9
        driven.trigger(36, 0.9)
        out_driven = driven.process(int(0.1 * sample_rate))

        fft_clean = np.abs(np.fft.rfft(out_clean))
        fft_driven = np.abs(np.fft.rfft(out_driven))
        high_band = len(fft_clean) // 4

        assert np.sum(fft_driven[high_band:]) > np.sum(fft_clean[high_band:])

    def test_release_is_noop(self, sample_rate):
        voice = KickVoice(sample_rate)
        voice.trigger(36, 0.9)
        voice.process(256)
        voice.release(36)
        output = voice.process(256)
        assert np.max(np.abs(output)) > 0.01
