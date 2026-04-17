"""Tests for ChordEngine — chord sequencer/controller."""

import pytest

from cypher.synth.chord_engine import (
    ChordEngine, MODE_NAMES,
    MODE_CHORD, MODE_STRUM_UP, MODE_STRUM_DOWN,
    MODE_ARP_UP, MODE_ARP_DOWN, MODE_ARP_UP_DOWN, MODE_ARP_RANDOM,
)
from cypher.synth.chords import PROGRESSION_LIST, SCALE_LIST


class _FakeTarget:
    """Captures trigger/release events for assertions."""
    def __init__(self):
        self.events: list[tuple[str, int, float]] = []

    def trigger(self, note: int, velocity: float) -> None:
        self.events.append(("on", note, velocity))

    def release(self, note: int) -> None:
        self.events.append(("off", note, 0.0))


class TestChordEngineBasics:
    def test_params_layout(self, sample_rate):
        ce = ChordEngine(sample_rate)
        assert len(ce.params) == 8
        assert ce.params[0].label == "KEY"
        assert ce.params[1].label == "SCALE"
        assert ce.params[2].label == "PROGRESSION"
        assert ce.params[4].label == "MODE"

    def test_idle_is_not_active(self, sample_rate):
        ce = ChordEngine(sample_rate)
        assert ce.is_active is False

    def test_no_target_is_safe(self, sample_rate):
        """advance() must not crash when target isn't set."""
        ce = ChordEngine(sample_rate)
        ce.trigger_chord_at_step(0.9)
        ce.advance(512)  # no target — should no-op


class TestChordMode:
    def test_all_notes_fire_together(self, sample_rate):
        ce = ChordEngine(sample_rate)
        target = _FakeTarget()
        ce.target = target
        ce.params[4].value = 0.0  # MODE_CHORD

        ce.trigger_chord_at_step(0.9)
        ce.advance(512)

        # Expect 3+ "on" events, all at the same (block) time
        ons = [e for e in target.events if e[0] == "on"]
        assert len(ons) >= 3

    def test_release_triggers_offs(self, sample_rate):
        ce = ChordEngine(sample_rate)
        target = _FakeTarget()
        ce.target = target

        ce.trigger_chord_at_step(0.9)
        ce.advance(512)
        ce.release_all()

        offs = [e for e in target.events if e[0] == "off"]
        assert len(offs) >= 3


class TestStrumMode:
    def test_strum_up_is_ordered_low_to_high(self, sample_rate):
        ce = ChordEngine(sample_rate)
        target = _FakeTarget()
        ce.target = target
        ce.params[4].value = 1 / (len(MODE_NAMES) - 1)  # MODE_STRUM_UP
        ce.params[5].value = 0.0  # fastest rate

        ce.trigger_chord_at_step(0.9)

        # Advance in small chunks so strum events land at distinct times
        for _ in range(20):
            ce.advance(256)

        ons = [e[1] for e in target.events if e[0] == "on"]
        assert ons == sorted(ons)  # ascending

    def test_strum_down_is_high_to_low(self, sample_rate):
        ce = ChordEngine(sample_rate)
        target = _FakeTarget()
        ce.target = target
        ce.params[4].value = 2 / (len(MODE_NAMES) - 1)  # MODE_STRUM_DOWN
        ce.params[5].value = 0.0

        ce.trigger_chord_at_step(0.9)
        for _ in range(20):
            ce.advance(256)

        ons = [e[1] for e in target.events if e[0] == "on"]
        assert ons == sorted(ons, reverse=True)


class TestArpMode:
    def test_arp_up_cycles_through_notes(self, sample_rate):
        ce = ChordEngine(sample_rate)
        target = _FakeTarget()
        ce.target = target
        ce.params[4].value = 3 / (len(MODE_NAMES) - 1)  # MODE_ARP_UP
        ce.params[5].value = 0.0  # fastest rate

        ce.trigger_chord_at_step(0.9)

        # Run for enough blocks to get multiple arp cycles
        for _ in range(40):
            ce.advance(256)

        ons = [e[1] for e in target.events if e[0] == "on"]
        assert len(ons) > 3  # should have stepped through several times
        # Arp should cycle — not all the same note
        assert len(set(ons)) > 1

    def test_arp_stops_on_release(self, sample_rate):
        ce = ChordEngine(sample_rate)
        target = _FakeTarget()
        ce.target = target
        ce.params[4].value = 3 / (len(MODE_NAMES) - 1)  # MODE_ARP_UP
        ce.params[5].value = 0.0

        ce.trigger_chord_at_step(0.9)
        for _ in range(10):
            ce.advance(256)
        ce.release_all()

        ons_before = sum(1 for e in target.events if e[0] == "on")
        for _ in range(20):
            ce.advance(256)
        ons_after = sum(1 for e in target.events if e[0] == "on")
        assert ons_after == ons_before  # no new notes after release


class TestKeyAndProgression:
    def test_set_key_changes_root(self, sample_rate):
        ce = ChordEngine(sample_rate)
        ce.set_key(0)
        root_c = ce.root_midi
        ce.set_key(5)
        root_f = ce.root_midi
        assert root_f - root_c == 5

    def test_progression_step_advances(self, sample_rate):
        ce = ChordEngine(sample_rate)
        chord_0, _ = ce.current_chord()
        ce.step_progression(1)
        chord_1, _ = ce.current_chord()
        assert chord_0 != chord_1

    def test_all_progressions_reachable(self, sample_rate):
        ce = ChordEngine(sample_rate)
        for i in range(len(PROGRESSION_LIST)):
            ce.set_progression(i)
            assert ce.progression_name == PROGRESSION_LIST[i]


class TestState:
    def test_get_state_shape(self, sample_rate):
        ce = ChordEngine(sample_rate)
        s = ce.get_state()
        assert "active" in s
        assert "key" in s
        assert "scale" in s
        assert "progression" in s
        assert "mode" in s
        assert "chord_label" in s

    def test_mode_name_lookup(self, sample_rate):
        ce = ChordEngine(sample_rate)
        ce.params[4].value = 0.0
        ce._update_params()
        assert ce.mode_name == "CHORD"
        ce.params[4].value = 1.0
        ce._update_params()
        assert ce.mode_name == MODE_NAMES[-1]
