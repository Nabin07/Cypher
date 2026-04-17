"""Shared test fixtures."""

import pytest

from cypher.core.types import DEFAULT_SAMPLE_RATE


@pytest.fixture
def sample_rate():
    return DEFAULT_SAMPLE_RATE


@pytest.fixture
def buffer_size():
    return 512
