"""Tests for FXEngine (global send-bus FX)."""

import numpy as np
import pytest

from cypher.fx.fx import (
    FXEngine, MODE_NAMES,
    MODE_PLATE, MODE_CHAMBER, MODE_HALL, MODE_ROOM, MODE_AMBIENCE,
)


class TestFXEngine:
    def test_params_layout(self, sample_rate):
        fx = FXEngine(sample_rate)
        assert len(fx.params) == 8
        assert fx.params[0].label == "MIX"
        assert fx.params[1].label == "PREDELAY"
        assert fx.params[2].label == "DECAY"
        assert fx.params[3].label == "MODE"
        assert fx.params[4].label == "HIGHCUT"
        assert fx.params[5].label == "LOWCUT"
        assert fx.params[6].label == "DAMPING"
        assert fx.params[7].label == "SIZE"

    def test_dry_is_preserved_at_mix_zero(self, sample_rate):
        """MIX=0 should return dry signal unchanged."""
        fx = FXEngine(sample_rate)
        fx.params[0].value = 0.0  # MIX = 0
        dry = np.random.default_rng(0).standard_normal(512).astype(np.float32) * 0.5
        send = dry.copy()
        out = fx.process(send, dry)
        np.testing.assert_allclose(out, dry, atol=1e-5)

    def test_wet_adds_energy(self, sample_rate):
        """With MIX>0 and a send signal, output has more energy than dry alone."""
        fx = FXEngine(sample_rate)
        fx.params[0].value = 0.8  # high MIX
        fx.params[2].value = 0.8  # DECAY

        impulse = np.zeros(4096, dtype=np.float32)
        impulse[0] = 1.0
        out = fx.process(impulse, impulse)

        # Tail of output should contain reverb energy
        tail = out[2048:]
        assert np.max(np.abs(tail)) > 0.001

    def test_silent_send_passes_dry_through(self, sample_rate):
        """Silent send + dry signal = dry signal at the output (plus tail decay)."""
        fx = FXEngine(sample_rate)
        fx.params[0].value = 0.5
        dry = np.random.default_rng(7).standard_normal(512).astype(np.float32) * 0.3
        send = np.zeros(512, dtype=np.float32)
        out = fx.process(send, dry)
        # Output should be very close to dry since send is zero
        np.testing.assert_allclose(out, dry, atol=1e-5)

    def test_modes_differ(self, sample_rate):
        """Different modes should produce different reverb tails."""
        impulse = np.zeros(4096, dtype=np.float32)
        impulse[0] = 1.0
        dry = np.zeros(4096, dtype=np.float32)

        outs = []
        for mode_idx in range(5):
            fx = FXEngine(sample_rate)
            fx.params[0].value = 1.0  # pure wet view
            fx.params[2].value = 0.7
            fx.params[3].value = mode_idx / 4.0  # select mode
            outs.append(fx.process(impulse, dry))

        # Each mode produces a unique tail
        for i in range(len(outs)):
            for j in range(i + 1, len(outs)):
                diff = np.sum(np.abs(outs[i] - outs[j]))
                assert diff > 0.01, f"Modes {i} and {j} produced identical output"

    def test_mode_name_lookup(self, sample_rate):
        fx = FXEngine(sample_rate)
        fx.params[3].value = 0.0
        fx.process(np.zeros(64, dtype=np.float32), np.zeros(64, dtype=np.float32))
        assert fx.mode_name == "PLATE"

        fx.params[3].value = 1.0
        fx.process(np.zeros(64, dtype=np.float32), np.zeros(64, dtype=np.float32))
        assert fx.mode_name == "AMBIENCE"

    def test_highcut_attenuates_highs(self, sample_rate):
        """Lowering HIGHCUT should reduce high-frequency content in wet tail."""
        impulse = np.zeros(4096, dtype=np.float32)
        impulse[0] = 1.0
        dry = np.zeros(4096, dtype=np.float32)

        fx_bright = FXEngine(sample_rate)
        fx_bright.params[0].value = 1.0  # full wet
        fx_bright.params[2].value = 0.85
        fx_bright.params[4].value = 1.0  # HIGHCUT max (~20k)
        out_bright = fx_bright.process(impulse, dry)

        fx_dark = FXEngine(sample_rate)
        fx_dark.params[0].value = 1.0
        fx_dark.params[2].value = 0.85
        fx_dark.params[4].value = 0.1  # HIGHCUT low
        out_dark = fx_dark.process(impulse, dry)

        tail = slice(1024, None)
        fft_bright = np.abs(np.fft.rfft(out_bright[tail]))
        fft_dark = np.abs(np.fft.rfft(out_dark[tail]))
        hf_band = len(fft_bright) // 2
        assert np.sum(fft_dark[hf_band:]) < np.sum(fft_bright[hf_band:])

    def test_lowcut_attenuates_lows(self, sample_rate):
        """Raising LOWCUT should reduce low-frequency content in wet tail."""
        impulse = np.zeros(4096, dtype=np.float32)
        impulse[0] = 1.0
        dry = np.zeros(4096, dtype=np.float32)

        fx_full = FXEngine(sample_rate)
        fx_full.params[0].value = 1.0
        fx_full.params[2].value = 0.85
        fx_full.params[5].value = 0.0  # LOWCUT off (~20Hz)
        out_full = fx_full.process(impulse, dry)

        fx_cut = FXEngine(sample_rate)
        fx_cut.params[0].value = 1.0
        fx_cut.params[2].value = 0.85
        fx_cut.params[5].value = 0.9  # LOWCUT high
        out_cut = fx_cut.process(impulse, dry)

        tail = slice(1024, None)
        fft_full = np.abs(np.fft.rfft(out_full[tail]))
        fft_cut = np.abs(np.fft.rfft(out_cut[tail]))
        lf_band = len(fft_full) // 16  # bottom bins
        assert np.sum(fft_cut[:lf_band]) < np.sum(fft_full[:lf_band])

    def test_clear_resets_state(self, sample_rate):
        fx = FXEngine(sample_rate)
        fx.params[0].value = 0.8
        fx.params[2].value = 0.9
        noise = np.random.default_rng(0).standard_normal(2048).astype(np.float32) * 0.3
        fx.process(noise, noise)

        fx.clear()
        silence = np.zeros(1024, dtype=np.float32)
        out = fx.process(silence, silence)
        assert np.max(np.abs(out)) < 1e-5

    def test_get_state(self, sample_rate):
        fx = FXEngine(sample_rate)
        fx.params[0].value = 0.5
        fx.process(np.zeros(64, dtype=np.float32), np.zeros(64, dtype=np.float32))
        s = fx.get_state()
        assert "mix" in s
        assert "mode" in s
        assert s["mode"] == "PLATE"

    def test_output_bounded(self, sample_rate):
        """Output shouldn't explode with high decay and long input."""
        fx = FXEngine(sample_rate)
        fx.params[0].value = 0.7
        fx.params[2].value = 0.95  # max-ish decay

        rng = np.random.default_rng(42)
        noise = rng.standard_normal(4096).astype(np.float32) * 0.4
        out = fx.process(noise, noise)
        assert np.max(np.abs(out)) < 5.0

    def test_voice_like_noops(self, sample_rate):
        """FX should accept voice-interface calls as no-ops (for UI use)."""
        fx = FXEngine(sample_rate)
        assert fx.is_active is False
        fx.trigger(60, 0.9)  # should not raise
        fx.release(60)
        fx.all_notes_off()
        assert fx.is_active is False
