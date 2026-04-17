"""Tests for the DrumEngine."""

import numpy as np
import pytest

from cypher.engine import DrumEngine
from cypher.midi import AllNotesOff, ControlChange, NoteOff, NoteOn


class TestDrumEngine:
    def test_808_trigger(self, sample_rate):
        engine = DrumEngine(sample_rate)
        engine.handle_midi(NoteOn(36, 100))
        output = engine.process(512)
        assert np.max(np.abs(output)) > 0.01

    def test_kick_trigger(self, sample_rate):
        engine = DrumEngine(sample_rate)
        engine.handle_midi(NoteOn(37, 100))
        output = engine.process(512)
        assert np.max(np.abs(output)) > 0.01

    def test_multiple_voices_mix(self, sample_rate):
        engine = DrumEngine(sample_rate)
        engine.handle_midi(NoteOn(36, 100))  # 808
        engine.handle_midi(NoteOn(37, 80))   # Kick
        output = engine.process(512)
        assert np.max(np.abs(output)) > 0.01

    def test_all_notes_off(self, sample_rate):
        engine = DrumEngine(sample_rate)
        engine.handle_midi(NoteOn(36, 100))
        engine.process(256)
        engine.handle_midi(AllNotesOff())
        output = engine.process(512)
        assert np.max(np.abs(output)) < 0.001

    def test_unmapped_note_ignored(self, sample_rate):
        engine = DrumEngine(sample_rate)
        engine.handle_midi(NoteOn(60, 100))  # C3 — not mapped
        output = engine.process(512)
        assert np.all(output == 0.0)

    def test_cc_parameter_control(self, sample_rate):
        engine = DrumEngine(sample_rate)
        # Map CC 20 to 808 drive (param index 2)
        engine.cc_map[20] = ("sub808", 2)
        engine.handle_midi(ControlChange(20, 127))
        assert engine.sub808.params[2].value == pytest.approx(1.0)

    def test_master_level(self, sample_rate):
        engine = DrumEngine(sample_rate)
        engine.master_level = 0.5
        engine.handle_midi(NoteOn(36, 100))
        output_half = engine.process(512)

        engine2 = DrumEngine(sample_rate)
        engine2.master_level = 1.0
        engine2.handle_midi(NoteOn(36, 100))
        output_full = engine2.process(512)

        assert np.max(np.abs(output_half)) < np.max(np.abs(output_full))

    def test_focused_voice_params(self, sample_rate):
        engine = DrumEngine(sample_rate)
        engine.focused_voice = "sub808"
        assert engine.focused_params[0].label == "DECAY"

        engine.focused_voice = "kick"
        assert engine.focused_params[0].label == "PUNCH"

    def test_get_state(self, sample_rate):
        engine = DrumEngine(sample_rate)
        engine.handle_midi(NoteOn(36, 100))
        engine.process(256)

        state = engine.get_state()
        assert "sub808" in state
        assert "kick" in state
        assert "master_amplitude" in state
        assert state["sub808"]["active"] is True

    def test_silence_when_idle(self, sample_rate):
        engine = DrumEngine(sample_rate)
        output = engine.process(512)
        assert np.all(output == 0.0)
