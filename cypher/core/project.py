"""Project — global state shared across all engines.

Lives once per app. Holds transport + musical context (key, scale, BPM)
that multiple engines read. Modelled on DAW project state.

C++ portability notes:
    - Plain struct-of-primitives. No virtuals, no allocations.
    - Engines take a reference/pointer at construction; read-only from
      their point of view (setters only called from UI/MIDI layer).
"""

from __future__ import annotations


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class Project:
    """Global musical context: key, scale, BPM."""

    def __init__(
        self,
        key: int = 0,
        scale_idx: int = 1,  # MINOR by default
        bpm: float = 120.0,
    ) -> None:
        self._key = key % 12
        self._scale_idx = scale_idx
        self._bpm = max(20.0, min(300.0, bpm))

    # ── key (0–11 chromatic) ─────────────────────────────────────────

    @property
    def key(self) -> int:
        return self._key

    @key.setter
    def key(self, value: int) -> None:
        self._key = int(value) % 12

    @property
    def key_name(self) -> str:
        return NOTE_NAMES[self._key]

    # ── scale (index into SCALE_NAMES maintained by callers) ──────────

    @property
    def scale_idx(self) -> int:
        return self._scale_idx

    @scale_idx.setter
    def scale_idx(self, value: int) -> None:
        self._scale_idx = max(0, int(value))

    # ── bpm ───────────────────────────────────────────────────────────

    @property
    def bpm(self) -> float:
        return self._bpm

    @bpm.setter
    def bpm(self, value: float) -> None:
        self._bpm = max(20.0, min(300.0, float(value)))

    def get_state(self) -> dict:
        return {
            "key": self._key,
            "key_name": self.key_name,
            "scale_idx": self._scale_idx,
            "bpm": self._bpm,
        }
