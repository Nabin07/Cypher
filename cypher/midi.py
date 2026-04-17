"""MIDI message types for the CYPHER engine.

Lightweight dataclasses — the engine receives these, never opens MIDI ports.
The real-time layer (built later) will parse raw MIDI and create these.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NoteOn:
    note: int       # 0–127
    velocity: int   # 0–127
    channel: int = 0

    @property
    def velocity_float(self) -> float:
        """Normalized velocity 0.0–1.0."""
        return self.velocity / 127.0


@dataclass(frozen=True, slots=True)
class NoteOff:
    note: int       # 0–127
    channel: int = 0


@dataclass(frozen=True, slots=True)
class ControlChange:
    cc: int         # 0–127
    value: int      # 0–127
    channel: int = 0

    @property
    def value_float(self) -> float:
        """Normalized CC value 0.0–1.0."""
        return self.value / 127.0


@dataclass(frozen=True, slots=True)
class AllNotesOff:
    """Special message — kill everything."""
    channel: int = 0


# Union type for convenience
MidiMessage = NoteOn | NoteOff | ControlChange | AllNotesOff
