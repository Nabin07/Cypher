"""Shared types and constants for the CYPHER DSP engine."""

import numpy as np
from numpy.typing import NDArray

AudioBuffer = NDArray[np.float32]

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_BUFFER_SIZE = 256  # frames per process() call in real-time mode

# MIDI note to frequency conversion reference
A4_FREQ = 440.0
A4_NOTE = 69


def note_to_freq(note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return A4_FREQ * (2.0 ** ((note - A4_NOTE) / 12.0))


def freq_to_note(freq: float) -> float:
    """Convert frequency in Hz to MIDI note number (float for cents)."""
    return A4_NOTE + 12.0 * np.log2(freq / A4_FREQ)


def db_to_gain(db: float) -> float:
    """Convert decibels to linear gain."""
    return 10.0 ** (db / 20.0)


def gain_to_db(gain: float) -> float:
    """Convert linear gain to decibels."""
    if gain <= 0:
        return -120.0
    return 20.0 * np.log10(gain)
