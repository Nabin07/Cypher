"""Tests for Project — global state."""

import pytest

from cypher.core.project import Project


class TestProject:
    def test_defaults(self):
        p = Project()
        assert p.key == 0
        assert p.scale_idx == 1
        assert p.bpm == 120.0
        assert p.key_name == "C"

    def test_key_wraps(self):
        p = Project()
        p.key = 14
        assert p.key == 2
        assert p.key_name == "D"

    def test_bpm_clamped(self):
        p = Project()
        p.bpm = 1000.0
        assert p.bpm == 300.0
        p.bpm = 1.0
        assert p.bpm == 20.0

    def test_scale_idx_clamped_nonnegative(self):
        p = Project()
        p.scale_idx = -5
        assert p.scale_idx == 0

    def test_get_state(self):
        p = Project(key=4, scale_idx=2, bpm=140.0)
        s = p.get_state()
        assert s["key"] == 4
        assert s["key_name"] == "E"
        assert s["bpm"] == 140.0
