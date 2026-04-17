"""Tests for the Parameter system."""

import pytest
from cypher.core.parameter import Curve, Parameter


class TestParameter:
    def test_linear_mapping(self):
        p = Parameter("test", "TEST", min_val=0.0, max_val=100.0, default=0.5)
        assert p.mapped == pytest.approx(50.0)

    def test_linear_min(self):
        p = Parameter("test", "TEST", min_val=0.0, max_val=100.0, default=0.0)
        assert p.mapped == pytest.approx(0.0)

    def test_linear_max(self):
        p = Parameter("test", "TEST", min_val=0.0, max_val=100.0, default=1.0)
        assert p.mapped == pytest.approx(100.0)

    def test_exponential_mapping(self):
        p = Parameter("test", "TEST", min_val=20.0, max_val=20000.0, default=0.5,
                       curve=Curve.EXPONENTIAL)
        # At 0.5, exponential should give geometric mean
        expected = 20.0 * ((20000.0 / 20.0) ** 0.5)
        assert p.mapped == pytest.approx(expected, rel=0.01)

    def test_exponential_endpoints(self):
        p = Parameter("test", "TEST", min_val=20.0, max_val=20000.0, default=0.0,
                       curve=Curve.EXPONENTIAL)
        assert p.mapped == pytest.approx(20.0)
        p.value = 1.0
        assert p.mapped == pytest.approx(20000.0)

    def test_clamping(self):
        p = Parameter("test", "TEST", min_val=0.0, max_val=1.0, default=0.5)
        p.value = -0.5
        assert p.value == 0.0
        p.value = 1.5
        assert p.value == 1.0

    def test_nudge(self):
        p = Parameter("test", "TEST", min_val=0.0, max_val=100.0, default=0.5)
        p.nudge(0.1)
        assert p.value == pytest.approx(0.6)
        p.nudge(-0.3)
        assert p.value == pytest.approx(0.3)

    def test_nudge_clamps(self):
        p = Parameter("test", "TEST", min_val=0.0, max_val=100.0, default=0.9)
        p.nudge(0.5)
        assert p.value == 1.0

    def test_reset(self):
        p = Parameter("test", "TEST", min_val=0.0, max_val=100.0, default=0.5)
        p.value = 0.9
        p.reset()
        assert p.value == 0.5

    def test_to_dict(self):
        p = Parameter("decay", "DECAY", min_val=0.0, max_val=1000.0,
                       default=0.5, unit="ms")
        d = p.to_dict()
        assert d["name"] == "decay"
        assert d["label"] == "DECAY"
        assert d["unit"] == "ms"
        assert "mapped" in d
        assert "display" in d
