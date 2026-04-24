"""Tests for SamplerEngine — 16 slots, 8 voices, resampling pitch."""

import numpy as np
import pytest

from cypher.sampler.sampler import (
    SamplerEngine, SampleSlot,
    PAD_COUNT, VOICE_COUNT, PAD_MIDI_START, PAD_MIDI_END,
    MODE_PAD, MODE_CLASSIC, MODE_CHOP,
    P_MODE, P_PITCH, P_REVERSE, P_GAIN, P_START, P_END,
    P_ATTACK, P_DECAY, P_SLICES, P_FILTER,
)
from cypher.sampler.loader import scan_folder, load_sample


def _fake_wav(freq_hz: float = 440.0, duration_sec: float = 0.5,
              sample_rate: int = 48000) -> tuple[np.ndarray, int]:
    """Build a sine-wave sample for testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec),
                    endpoint=False, dtype=np.float32)
    data = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    return data, sample_rate


class TestSampleSlot:
    def test_starts_empty(self):
        s = SampleSlot()
        assert s.loaded is False
        assert s.length == 0
        assert len(s.params) == 14

    def test_load_sets_state(self):
        s = SampleSlot()
        data, sr = _fake_wav()
        s.load("kick.wav", "/path/kick.wav", data, sr)
        assert s.loaded is True
        assert s.name == "kick.wav"
        assert s.source_rate == sr
        assert s.length == len(data)

    def test_clear_resets(self):
        s = SampleSlot()
        data, sr = _fake_wav()
        s.load("x", "/x", data, sr)
        s.clear()
        assert not s.loaded
        assert s.length == 0


class TestSamplerEngineBasics:
    def test_params_shape(self, sample_rate):
        se = SamplerEngine(sample_rate)
        assert len(se.params) == 14
        labels = [p.label for p in se.params]
        for required in ["MODE", "PITCH", "FILTER", "START", "GAIN", "SLICES"]:
            assert required in labels

    def test_idle_is_silent(self, sample_rate):
        se = SamplerEngine(sample_rate)
        out = se.process(512)
        assert np.all(out == 0.0)
        assert se.is_active is False

    def test_unloaded_pad_does_nothing(self, sample_rate):
        se = SamplerEngine(sample_rate)
        se.trigger(PAD_MIDI_START, 0.9)  # pad 0, but empty
        out = se.process(512)
        assert np.all(out == 0.0)

    def test_trigger_plays_sample(self, sample_rate):
        se = SamplerEngine(sample_rate)
        data, sr = _fake_wav(sample_rate=sample_rate)
        se.load_into_slot(0, "sine.wav", "/sine.wav", data, sr)
        se.trigger(PAD_MIDI_START, 0.9)
        out = se.process(1024)
        assert np.max(np.abs(out)) > 0.01
        assert se.is_active is True

    def test_pad_midi_range(self):
        # Notes outside C2–D#3 shouldn't map to any pad
        for n in [35, 52, 0, 127]:
            assert SamplerEngine._note_to_pad(n) == -1
        # Inclusive range
        assert SamplerEngine._note_to_pad(PAD_MIDI_START) == 0
        assert SamplerEngine._note_to_pad(PAD_MIDI_END) == PAD_COUNT - 1


class TestSamplerPolyphony:
    def test_voice_count(self, sample_rate):
        se = SamplerEngine(sample_rate)
        assert len([v for v in se._voices]) == VOICE_COUNT

    def test_multiple_pads_play_together(self, sample_rate):
        se = SamplerEngine(sample_rate)
        data, sr = _fake_wav(sample_rate=sample_rate)
        for i in range(4):
            se.load_into_slot(i, f"s{i}.wav", f"/s{i}.wav", data, sr)
            se.trigger(PAD_MIDI_START + i, 0.9)
        out = se.process(512)
        assert np.max(np.abs(out)) > 0.01
        active = sum(1 for v in se._voices if v.is_active)
        assert active == 4

    def test_voice_stealing(self, sample_rate):
        """Triggering more pads than voices should steal the oldest."""
        se = SamplerEngine(sample_rate)
        data, sr = _fake_wav(sample_rate=sample_rate, duration_sec=2.0)
        # Load 10 slots so we can trigger 10 different notes
        for i in range(10):
            se.load_into_slot(i, f"s{i}.wav", f"/s{i}.wav", data, sr)

        for i in range(10):
            se.trigger(PAD_MIDI_START + i, 0.9)
            se.process(64)  # advance a little

        active = sum(1 for v in se._voices if v.is_active)
        assert active == VOICE_COUNT  # capped at pool size

    def test_release_stops_note(self, sample_rate):
        se = SamplerEngine(sample_rate)
        data, sr = _fake_wav(sample_rate=sample_rate, duration_sec=2.0)
        se.load_into_slot(0, "x.wav", "/x.wav", data, sr)

        se.trigger(PAD_MIDI_START, 0.9)
        se.process(256)
        se.release(PAD_MIDI_START)
        # Release → envelope enters release stage; play enough to let it decay
        se.process(sample_rate)
        out_tail = se.process(256)
        assert np.max(np.abs(out_tail)) < 0.05


class TestSamplerPitchAndReverse:
    def test_pitch_preserves_duration(self, sample_rate):
        """Phase-vocoder pitch shift: +12 semi should NOT speed up playback."""
        # Long sine so phase vocoder has frames to work with
        t = np.arange(int(sample_rate * 0.5)) / sample_rate
        data = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        se = SamplerEngine(sample_rate)
        se.load_into_slot(0, "x", "/x", data, sample_rate)
        # Put slot into PAD mode so only pad 0 triggers it, and apply PITCH
        se.slots[0].params[P_MODE].value = 0.0       # MODE_PAD
        se.slots[0].params[P_PITCH].value = 0.75     # PITCH = +12 semi
        se.trigger(PAD_MIDI_START, 1.0)

        # Play back 1.2x the sample duration and check energy doesn't end early
        total_out = se.process(int(sample_rate * 0.45))
        # Energy should extend through most of the output (not bunched at start)
        mid_energy = np.sum(total_out[sample_rate // 4:] ** 2)
        assert mid_energy > 0.01  # still playing well into the buffer

    def test_classic_mode_pitches_with_pad(self, sample_rate):
        """CLASSIC mode: pad N plays slot at +N semitones."""
        t = np.arange(int(sample_rate * 0.5)) / sample_rate
        data = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        se = SamplerEngine(sample_rate)
        se.load_into_slot(0, "x", "/x", data, sample_rate)
        se.slots[0].params[P_MODE].value = 0.5  # CLASSIC (snap=3, 0.5 ≈ index 1)

        # Trigger pad 7 (=7 semi above slot 0) and pad 0 (=0 semi)
        se.trigger(PAD_MIDI_START, 1.0)
        out_0 = se.process(int(sample_rate * 0.2))

        se2 = SamplerEngine(sample_rate)
        se2.load_into_slot(0, "x", "/x", data, sample_rate)
        se2.slots[0].params[P_MODE].value = 0.5  # CLASSIC
        se2.trigger(PAD_MIDI_START + 7, 1.0)
        out_7 = se2.process(int(sample_rate * 0.2))

        # Dominant frequency of out_7 should be ~3/2 the one from out_0
        f0 = _dominant_freq(out_0, sample_rate)
        f7 = _dominant_freq(out_7, sample_rate)
        ratio = f7 / max(1.0, f0)
        assert 1.4 < ratio < 1.6

    def test_reverse_plays_backwards(self, sample_rate):
        data = np.zeros(int(sample_rate * 0.1), dtype=np.float32)
        impulse_from_end = 800
        data[-impulse_from_end] = 1.0
        se = SamplerEngine(sample_rate)
        se.load_into_slot(0, "x", "/x", data, sample_rate)
        se.slots[0].params[P_REVERSE].value = 1.0  # REVERSE on
        se.trigger(PAD_MIDI_START, 1.0)
        out = se.process(2048)
        peak = int(np.argmax(np.abs(out)))
        assert abs(peak - impulse_from_end) < 50


def _dominant_freq(data, sr):
    spec = np.abs(np.fft.rfft(data))
    peak_bin = int(np.argmax(spec))
    return peak_bin * sr / max(1, len(data))


class TestSamplerChop:
    def test_chop_mode_slices_trigger_from_pads(self, sample_rate):
        """CHOP mode: pads N, N+1, ... play slices 0, 1, ..."""
        # Sample where each quarter has a distinct impulse location
        data = np.zeros(sample_rate, dtype=np.float32)
        data[100] = 1.0                          # slice 0 starts
        data[sample_rate // 4 + 100] = 1.0       # slice 1 starts
        data[sample_rate // 2 + 100] = 1.0       # slice 2 starts
        data[3 * sample_rate // 4 + 100] = 1.0   # slice 3 starts

        se = SamplerEngine(sample_rate)
        se.load_into_slot(0, "x", "/x", data, sample_rate)
        slot = se.slots[0]
        slot.params[P_MODE].value = 1.0         # MODE_CHOP (snap=3, value=1.0 → idx 2)
        slot.params[P_SLICES].value = 3.0 / 31.0  # SLICES = 4 (snap=32, normalized)

        # Trigger pad 2 → slice 2 → first 200 samples of output should be silent
        se.trigger(PAD_MIDI_START + 2, 1.0)
        out = se.process(400)
        # Output starts from where slice 2 begins; impulse at slice_start+100
        assert int(np.argmax(np.abs(out))) < 200


class TestSamplerFocus:
    def test_default_focus(self, sample_rate):
        se = SamplerEngine(sample_rate)
        assert se.focused_slot_idx == 0
        # params comes from the focused slot
        assert se.params is se.slots[0].params

    def test_focus_changes_params(self, sample_rate):
        se = SamplerEngine(sample_rate)
        se.focus_slot(5)
        assert se.focused_slot_idx == 5
        assert se.params is se.slots[5].params

    def test_focus_clamped(self, sample_rate):
        se = SamplerEngine(sample_rate)
        se.focus_slot(999)
        assert se.focused_slot_idx == PAD_COUNT - 1


class TestSamplerSlotManagement:
    def test_find_free_slot(self, sample_rate):
        se = SamplerEngine(sample_rate)
        data, sr = _fake_wav(sample_rate=sample_rate)
        assert se.find_free_slot() == 0
        se.load_into_slot(0, "x", "/x", data, sr)
        assert se.find_free_slot() == 1
        # Fill everything
        for i in range(1, PAD_COUNT):
            se.load_into_slot(i, f"s{i}", "/s", data, sr)
        assert se.find_free_slot() == -1

    def test_clear_slot(self, sample_rate):
        se = SamplerEngine(sample_rate)
        data, sr = _fake_wav(sample_rate=sample_rate)
        se.load_into_slot(0, "x", "/x", data, sr)
        se.clear_slot(0)
        assert not se.slots[0].loaded


class TestGetState:
    def test_state_shape(self, sample_rate):
        se = SamplerEngine(sample_rate)
        s = se.get_state()
        assert "active" in s
        assert "focused_slot" in s
        assert "slots_loaded" in s
        assert s["max_voices"] == VOICE_COUNT


class TestLoader:
    def test_scan_empty_folder(self, tmp_path):
        assert scan_folder(tmp_path) == []

    def test_scan_finds_wavs(self, tmp_path):
        import soundfile as sf
        data = np.random.default_rng(0).standard_normal(1000).astype(np.float32) * 0.1
        for name in ["a.wav", "b.wav", "c.mp3", "d.WAV"]:
            if name.lower().endswith(".wav"):
                sf.write(tmp_path / name, data, 48000)
            else:
                (tmp_path / name).write_text("not audio")
        found = scan_folder(tmp_path)
        # Only WAVs returned, case-insensitive
        names = [p.name for p in found]
        assert "a.wav" in names
        assert "b.wav" in names
        assert "d.WAV" in names
        assert "c.mp3" not in names

    def test_load_sample_mono(self, tmp_path):
        import soundfile as sf
        # Stereo input — loader should sum to mono
        data = np.random.default_rng(0).standard_normal((1024, 2)).astype(np.float32)
        p = tmp_path / "stereo.wav"
        sf.write(p, data, 48000)
        buf, sr = load_sample(p)
        assert buf.ndim == 1
        assert sr == 48000
        assert buf.dtype == np.float32
