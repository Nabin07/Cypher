"""MIDI message types for the CYPHER engine.

Lightweight dataclasses — the engine receives these, never opens MIDI ports.
The real-time layer (built later) will parse raw MIDI and create these.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

_DC_KWARGS = {"frozen": True}
if sys.version_info >= (3, 10):
    _DC_KWARGS["slots"] = True


@dataclass(**_DC_KWARGS)
class NoteOn:
    note: int       # 0–127
    velocity: int   # 0–127
    channel: int = 0

    @property
    def velocity_float(self) -> float:
        """Normalized velocity 0.0–1.0."""
        return self.velocity / 127.0


@dataclass(**_DC_KWARGS)
class NoteOff:
    note: int       # 0–127
    channel: int = 0


@dataclass(**_DC_KWARGS)
class ControlChange:
    cc: int         # 0–127
    value: int      # 0–127
    channel: int = 0

    @property
    def value_float(self) -> float:
        """Normalized CC value 0.0–1.0."""
        return self.value / 127.0


@dataclass(**_DC_KWARGS)
class AllNotesOff:
    """Special message — kill everything."""
    channel: int = 0


# Union type for convenience
from typing import Union
MidiMessage = Union[NoteOn, NoteOff, ControlChange, AllNotesOff]
