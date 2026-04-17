"""Base voice class — contract for all CYPHER sound voices."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .parameter import Parameter
from .types import AudioBuffer, DEFAULT_SAMPLE_RATE


class Voice(ABC):
    """Abstract base for a synthesized voice.

    Every voice follows the same contract:
    - trigger(velocity) starts sound production
    - release() handles note-off
    - process(num_frames) returns audio samples
    - get_state() returns a UI-friendly snapshot
    - params is an ordered list mapping to hardware encoders
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

    @property
    @abstractmethod
    def params(self) -> list[Parameter]:
        """Ordered parameter list — maps to encoder positions.

        Index 0 = leftmost encoder, index 3 = rightmost.
        In advanced mode, additional params beyond index 3 are
        accessible via page navigation.
        """
        ...

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """True if the voice is currently producing sound."""
        ...

    @abstractmethod
    def trigger(self, note: int, velocity: float) -> None:
        """Start sound production.

        Args:
            note: MIDI note number (0–127).
            velocity: Normalized velocity (0.0–1.0).
        """
        ...

    @abstractmethod
    def release(self, note: int) -> None:
        """Handle note-off.

        Args:
            note: MIDI note number being released.
        """
        ...

    @abstractmethod
    def process(self, num_frames: int) -> AudioBuffer:
        """Generate audio samples.

        Args:
            num_frames: Number of samples to generate.

        Returns:
            Float32 audio buffer of length num_frames.
        """
        ...

    @abstractmethod
    def get_state(self) -> dict:
        """Return a snapshot dict for UI animation.

        Called by the UI at its own frame rate (e.g. 60fps).
        Never called from the audio thread.
        """
        ...

    def all_notes_off(self) -> None:
        """Kill all sound immediately. Override if needed."""
        self.release(0)
