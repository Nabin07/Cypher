"""Tests for PolySynthVoice and chord system."""

import numpy as np
import pytest

from cypher.synth.poly import PolySynthVoice
from cypher.synth.chords import (
    build_chord, build_progression_chord, progression_length,
    CHORD_TYPES, CHORD_TYPE_LIST, PROGRESSIONS, PROGRESSION_LIST,
)


class TestPolySynth:
    def test_single_note_produces_sound(self, sample_rate):
        v = PolySynthVoice(sample_rate)
        v.trigger(60, 0.9)
        out = v.process(512)
        assert np.max(np.abs(out)) > 0.01

    def test_idle_is_silent(self, sample_rate):
        v = PolySynthVoice(sample_rate)
        out = v.process(512)
        assert np.all(out == 0.0)

    def test_multiple_simultaneous_notes(self, sample_rate):
        v = PolySynthVoice(sample_rate)
        v.trigger(60, 0.9)
        v.trigger(64, 0.9)
        v.trigger(67, 0.9)
        out = v.process(512)
        assert np.max(np.abs(out)) > 0.01
        assert v.active_voice_count == 3

    def test_chord_trigger(self, sample_rate):
        v = PolySynthVoice(sample_rate)
        v.trigger_chord([60, 64, 67], 0.9)
        out = v.process(512)
        assert np.max(np.abs(out)) > 0.01
        assert v.active_voice_count == 3

    def test_release_individual_note(self, sample_rate):
        v = PolySynthVoice(sample_rate)
        v.trigger(60, 0.9)
        v.trigger(64, 0.9)
        v.process(512)

        v.release(60)
        state = v.get_state()
        assert 60 not in state["active_notes"]
        assert 64 in state["active_notes"]

    def test_release_all(self, sample_rate):
        v = PolySynthVoice(sample_rate)
        v.trigger_chord([60, 64, 67], 0.9)
        v.process(512)

        v.release_all()
        state = v.get_state()
        assert len(state["active_notes"]) == 0

    def test_all_notes_off(self, sample_rate):
        v = PolySynthVoice(sample_rate)
        v.trigger_chord([60, 64, 67, 72], 0.9)
        v.process(512)

        v.all_notes_off()
        out = v.process(512)
        assert np.all(out == 0.0)
        assert not v.is_active

    def test_voice_stealing(self, sample_rate):
        """When all voices are used, oldest should be stolen."""
        v = PolySynthVoice(sample_rate, max_voices=3)
        v.trigger(60, 0.9)
        v.trigger(64, 0.9)
        v.trigger(67, 0.9)
        v.process(256)

        v.trigger(72, 0.9)
        state = v.get_state()
        assert 72 in state["active_notes"]
        assert 60 not in state["active_notes"]

    def test_retrigger_same_note(self, sample_rate):
        """Retriggering the same note reuses its voice."""
        v = PolySynthVoice(sample_rate)
        v.trigger(60, 0.9)
        v.process(256)
        assert v.active_voice_count == 1

        v.trigger(60, 0.9)
        v.process(256)
        assert v.active_voice_count == 1

    def test_shared_params(self, sample_rate):
        """All voices in the pool should share the same params."""
        v = PolySynthVoice(sample_rate)
        v.params[0].value = 0.33  # WAVE A: SIN
        v.trigger(60, 0.9)
        v.trigger(67, 0.9)
        out = v.process(512)
        assert np.max(np.abs(out)) > 0.01
        state = v.get_state()
        assert state["wave_a"] == "SIN"

    def test_params_layout(self, sample_rate):
        v = PolySynthVoice(sample_rate)
        assert len(v.params) == 16
        assert v.params[0].label == "WAVE A"
        assert v.params[4].label == "CUTOFF"
        assert v.params[7].label == "MODE"
        assert v.params[8].label == "ATTACK"
        assert v.params[12].label == "LFO RATE"

    def test_wave_names(self, sample_rate):
        v = PolySynthVoice(sample_rate)
        assert v.wave_a_name == "SAW"
        assert v.wave_b_name == "SAW"
        v.params[0].value = 1.0  # TRI
        v.params[1].value = 0.33  # SIN
        assert v.wave_a_name == "TRI"
        assert v.wave_b_name == "SIN"

    def test_filter_mode_name(self, sample_rate):
        v = PolySynthVoice(sample_rate)
        assert v.filter_mode_name == "LP"
        v.params[7].value = 0.5  # HP
        assert v.filter_mode_name == "HP"

    def test_get_state(self, sample_rate):
        v = PolySynthVoice(sample_rate)
        v.trigger_chord([60, 64, 67], 0.9)
        v.process(256)

        state = v.get_state()
        assert state["active"] is True
        assert state["active_voices"] == 3
        assert state["max_voices"] == 8
        assert set(state["active_notes"]) == {60, 64, 67}
        assert "wave_a" in state
        assert "wave_b" in state
        assert "filter_mode" in state

    def test_chord_has_richer_spectrum(self, sample_rate):
        """A chord should have more spectral content than a single note."""
        v_single = PolySynthVoice(sample_rate)
        v_single.params[4].value = 1.0
        v_single.trigger(60, 0.9)
        v_single.process(512)
        out_single = v_single.process(2048)

        v_chord = PolySynthVoice(sample_rate)
        v_chord.params[4].value = 1.0
        v_chord.trigger_chord([60, 64, 67], 0.9)
        v_chord.process(512)
        out_chord = v_chord.process(2048)

        fft_single = np.abs(np.fft.rfft(out_single))
        fft_chord = np.abs(np.fft.rfft(out_chord))
        thresh = np.max(fft_single) * 0.1
        peaks_single = np.sum(fft_single > thresh)
        peaks_chord = np.sum(fft_chord > thresh)
        assert peaks_chord > peaks_single

    def test_release_decays_to_silence(self, sample_rate):
        """After release, all voices should eventually go silent."""
        v = PolySynthVoice(sample_rate)
        v.trigger_chord([60, 64, 67], 0.9)
        v.process(1024)

        v.release_all()
        out = v.process(int(2.0 * sample_rate))
        tail = out[-512:]
        assert np.max(np.abs(tail)) < 0.01


class TestChords:
    def test_build_major_chord(self):
        notes = build_chord(60, "MAJ")
        assert notes == [60, 64, 67]

    def test_build_minor_chord(self):
        notes = build_chord(60, "MIN")
        assert notes == [60, 63, 67]

    def test_build_seventh_chord(self):
        notes = build_chord(60, "MIN7")
        assert notes == [60, 63, 67, 70]

    def test_build_power_chord(self):
        notes = build_chord(60, "PWR")
        assert notes == [60, 67]

    def test_unknown_chord_defaults_to_major(self):
        notes = build_chord(60, "DOESNT_EXIST")
        assert notes == [60, 64, 67]

    def test_progression_chord(self):
        notes, label = build_progression_chord(60, "TRAP I", 0)
        assert notes == [60, 63, 67]  # Cm
        assert "C" in label

        notes, label = build_progression_chord(60, "TRAP I", 1)
        assert notes[0] == 68

    def test_progression_wraps(self):
        plen = progression_length("TRAP I")
        assert plen == 4
        notes_0, _ = build_progression_chord(60, "TRAP I", 0)
        notes_4, _ = build_progression_chord(60, "TRAP I", 4)
        assert notes_0 == notes_4

    def test_all_progressions_defined(self):
        for name in PROGRESSION_LIST:
            assert name in PROGRESSIONS
            assert len(PROGRESSIONS[name]) >= 2

    def test_all_chord_types_defined(self):
        for name in CHORD_TYPE_LIST:
            assert name in CHORD_TYPES
            intervals = CHORD_TYPES[name]
            assert intervals[0] == 0
            assert len(intervals) >= 2

    def test_progression_labels(self):
        _, label = build_progression_chord(60, "TRAP I", 0)
        assert label == "Cm"

        _, label = build_progression_chord(60, "TRAP I", 1)
        assert "G#" in label or "Ab" in label

    def test_transposition(self):
        notes_c, _ = build_progression_chord(60, "TRAP I", 0)
        notes_d, _ = build_progression_chord(62, "TRAP I", 0)
        for nc, nd in zip(notes_c, notes_d):
            assert nd - nc == 2
