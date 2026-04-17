"""DrumEngine — composition root for the Drum Designer.

Owns 808 and kick voices, handles MIDI dispatch, mixes output,
and aggregates state for the UI layer.
"""

from __future__ import annotations

import numpy as np

from .core.parameter import Parameter
from .core.types import AudioBuffer, DEFAULT_SAMPLE_RATE
from .drum.kick import KickVoice
from .drum.sub808 import Sub808Voice
from .midi import AllNotesOff, ControlChange, MidiMessage, NoteOff, NoteOn


class DrumEngine:
    """Drum Designer engine — 808 + kick voices with MIDI dispatch.

    Default MIDI note mapping (GM-compatible, configurable):
        808 Sub: 36 (C1) — also responds to 35
        Kick:    37

    CC mapping: configurable, maps CC numbers to (voice, param_index).
    """

    # Default note-to-voice mapping
    DEFAULT_NOTE_MAP: dict[int, str] = {
        35: "sub808",
        36: "sub808",   # C1 — main 808 note
        37: "kick",     # Side stick position
    }

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

        # Voices
        self.sub808 = Sub808Voice(sample_rate)
        self.kick = KickVoice(sample_rate)

        self._voices: dict[str, Sub808Voice | KickVoice] = {
            "sub808": self.sub808,
            "kick": self.kick,
        }

        # Configurable mappings
        self.note_map: dict[int, str] = dict(self.DEFAULT_NOTE_MAP)
        self.cc_map: dict[int, tuple[str, int]] = {}  # CC# -> (voice_name, param_index)

        # Master level
        self._master_level: float = 0.8
        self._output_amplitude: float = 0.0

        # Currently focused voice (for encoder routing)
        self._focused_voice: str = "sub808"

    @property
    def focused_voice(self) -> str:
        return self._focused_voice

    @focused_voice.setter
    def focused_voice(self, name: str) -> None:
        if name in self._voices:
            self._focused_voice = name

    @property
    def focused_params(self) -> list[Parameter]:
        """Parameters for the currently focused voice (for encoder display)."""
        return self._voices[self._focused_voice].params

    @property
    def master_level(self) -> float:
        return self._master_level

    @master_level.setter
    def master_level(self, value: float) -> None:
        self._master_level = max(0.0, min(1.0, value))

    def handle_midi(self, msg: MidiMessage) -> None:
        """Dispatch a MIDI message to the appropriate voice(s)."""
        if isinstance(msg, NoteOn):
            voice_name = self.note_map.get(msg.note)
            if voice_name and voice_name in self._voices:
                self._voices[voice_name].trigger(msg.note, msg.velocity_float)

        elif isinstance(msg, NoteOff):
            voice_name = self.note_map.get(msg.note)
            if voice_name and voice_name in self._voices:
                self._voices[voice_name].release(msg.note)

        elif isinstance(msg, ControlChange):
            mapping = self.cc_map.get(msg.cc)
            if mapping:
                voice_name, param_idx = mapping
                voice = self._voices.get(voice_name)
                if voice and param_idx < len(voice.params):
                    voice.params[param_idx].value = msg.value_float

        elif isinstance(msg, AllNotesOff):
            self.all_notes_off()

    def process(self, num_frames: int) -> AudioBuffer:
        """Mix all voices and return the output buffer."""
        output = np.zeros(num_frames, dtype=np.float32)

        # Pass live 808 freq to kick when paired (tracks pitch changes)
        if self.kick._paired:
            if self.sub808.is_active:
                self.kick.set_linked_808_freq(self.sub808._current_pitch_hz)
            else:
                self.kick.set_linked_808_freq(0.0)

        for voice in self._voices.values():
            if voice.is_active:
                output += voice.process(num_frames)

        output *= self._master_level

        # Soft limit to prevent clipping
        peak = np.max(np.abs(output))
        if peak > 1.0:
            output /= peak

        self._output_amplitude = float(np.max(np.abs(output))) if len(output) > 0 else 0.0

        return output

    def all_notes_off(self) -> None:
        """Kill all voices immediately."""
        for voice in self._voices.values():
            voice.all_notes_off()
        self._output_amplitude = 0.0

    def get_state(self) -> dict:
        """Aggregated state for the UI layer."""
        return {
            "sub808": self.sub808.get_state(),
            "kick": self.kick.get_state(),
            "master_amplitude": self._output_amplitude,
            "master_level": self._master_level,
            "focused_voice": self._focused_voice,
        }
