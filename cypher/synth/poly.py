"""PolySynthVoice — polyphonic wrapper around MonoSynthVoice.

Manages a pool of MonoSynthVoice instances. All voices share the same
parameters (same timbre), but can play different notes simultaneously.
Voice allocation uses oldest-steal when the pool is exhausted.

C++ portability notes:
  - Voice pool is a fixed-size array, not a dynamic vector.
  - Note map is a small fixed-size lookup (128 MIDI notes max).
  - Allocation order tracked as a simple circular index list.
"""

from __future__ import annotations

import numpy as np

from ..core.parameter import Parameter
from ..core.types import AudioBuffer, DEFAULT_SAMPLE_RATE
from ..core.voice import Voice
from .mono import MonoSynthVoice, WAVE_NAMES, FILTER_MODE_NAMES

MAX_VOICES = 8


class PolySynthVoice(Voice):
    """Polyphonic synth — allocates MonoSynthVoice instances per note."""

    def __init__(
        self, sample_rate: int = DEFAULT_SAMPLE_RATE, max_voices: int = MAX_VOICES
    ) -> None:
        super().__init__(sample_rate)

        # Master params — shared by all voices in the pool
        self._master = MonoSynthVoice(sample_rate)
        self._pool: list[MonoSynthVoice] = []
        for _ in range(max_voices):
            v = MonoSynthVoice(sample_rate)
            v._params = self._master._params  # share params
            self._pool.append(v)

        # Note → voice mapping (only for currently held/active notes)
        self._note_map: dict[int, MonoSynthVoice] = {}
        # Allocation order — oldest first, for voice stealing
        self._alloc_order: list[MonoSynthVoice] = []
        # Smoothed voice-count scaling to prevent pops when voices come/go
        self._smoothed_scale: float = 1.0

    # ------------------------------------------------------------------
    # Voice interface
    # ------------------------------------------------------------------

    @property
    def params(self) -> list[Parameter]:
        return self._master._params

    @property
    def is_active(self) -> bool:
        return any(v.is_active for v in self._pool)

    @property
    def wave_a_name(self) -> str:
        self._master._update_params()
        return self._master.wave_a_name

    @property
    def wave_b_name(self) -> str:
        self._master._update_params()
        return self._master.wave_b_name

    @property
    def filter_mode_name(self) -> str:
        self._master._update_params()
        return self._master.filter_mode_name

    def trigger(self, note: int, velocity: float) -> None:
        # Reuse voice if this note is already sounding
        if note in self._note_map:
            voice = self._note_map[note]
        else:
            voice = self._allocate_voice()

        voice.trigger(note, velocity)
        self._note_map[note] = voice

        # Track allocation order
        if voice in self._alloc_order:
            self._alloc_order.remove(voice)
        self._alloc_order.append(voice)

    def release(self, note: int) -> None:
        if note in self._note_map:
            self._note_map[note].release(note)
            del self._note_map[note]

    def all_notes_off(self) -> None:
        for v in self._pool:
            v.all_notes_off()
        self._note_map.clear()
        self._alloc_order.clear()

    def process(self, num_frames: int) -> AudioBuffer:
        buf = np.zeros(num_frames, dtype=np.float32)
        active = 0
        for v in self._pool:
            if v.is_active:
                buf += v.process(num_frames)
                active += 1

        # Ramp the voice-count scaling across the block to avoid pops when
        # voices enter/exit. Previously jumped instantly between blocks.
        target_scale = 1.0 / np.sqrt(max(1, active))
        start_scale = self._smoothed_scale
        ramp = np.linspace(start_scale, target_scale, num_frames, dtype=np.float32)
        buf *= ramp
        self._smoothed_scale = target_scale

        return buf

    def get_state(self) -> dict:
        self._master._update_params()
        return {
            "active": self.is_active,
            "wave_a": self._master.wave_a_name,
            "wave_b": self._master.wave_b_name,
            "osc_mix": self._master._osc_mix,
            "filter_mode": self._master.filter_mode_name,
            "active_notes": list(self._note_map.keys()),
            "active_voices": sum(1 for v in self._pool if v.is_active),
            "max_voices": len(self._pool),
            "amp_env_stage": "poly",
            "filter_env_stage": "poly",
        }

    # ------------------------------------------------------------------
    # Poly-specific
    # ------------------------------------------------------------------

    def trigger_chord(self, notes: list[int], velocity: float) -> None:
        """Trigger multiple notes simultaneously."""
        for note in notes:
            self.trigger(note, velocity)

    def release_all(self) -> None:
        """Release all currently held notes (enter release stage)."""
        for note in list(self._note_map.keys()):
            self._note_map[note].release(note)
        self._note_map.clear()

    @property
    def active_voice_count(self) -> int:
        return sum(1 for v in self._pool if v.is_active)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _allocate_voice(self) -> MonoSynthVoice:
        """Find a free voice, or steal the oldest active one."""
        # Prefer idle voices
        for v in self._pool:
            if not v.is_active:
                return v

        # All busy — steal oldest. Don't hard-kill the voice; trigger() will
        # transition smoothly from current envelope level into the new attack.
        if self._alloc_order:
            stolen = self._alloc_order.pop(0)
            for note, voice in list(self._note_map.items()):
                if voice is stolen:
                    del self._note_map[note]
                    break
            return stolen

        # Fallback (shouldn't happen)
        return self._pool[0]
