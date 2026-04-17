"""Real MIDI input from hardware controllers.

Uses mido + python-rtmidi for cross-platform MIDI I/O (Mac + Pi).
Install: pip install mido python-rtmidi

On Raspberry Pi you also need: sudo apt install libasound2-dev
"""

from __future__ import annotations

from collections.abc import Callable

from .midi import ControlChange, MidiMessage, NoteOff, NoteOn

try:
    import mido

    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False


def available() -> bool:
    """True if mido + rtmidi are installed."""
    return HAS_MIDO


def list_inputs() -> list[str]:
    """Return available MIDI input port names."""
    if not HAS_MIDO:
        return []
    try:
        return list(mido.get_input_names())
    except Exception:
        return []


def find_device(preferred: list[str] | None = None) -> str | None:
    """Auto-detect a MIDI input. Prefers Arturia/KeyLab by default."""
    names = list_inputs()
    if not names:
        return None

    keywords = preferred or ["arturia", "keylab", "essential"]
    for name in names:
        lower = name.lower()
        if any(kw in lower for kw in keywords):
            return name

    # Skip system/virtual ports
    for name in names:
        lower = name.lower()
        if "midi through" not in lower and "timer" not in lower:
            return name

    return names[0]


class MidiInput:
    """Reads MIDI from a hardware port, dispatches as MidiMessage.

    Uses mido's callback mode — messages arrive on a background thread.
    Your callback must be thread-safe.
    """

    def __init__(self, callback: Callable[[MidiMessage], None]) -> None:
        if not HAS_MIDO:
            raise RuntimeError(
                "mido not installed — run: pip install mido python-rtmidi"
            )
        self.callback = callback
        self._port = None
        self.device_name: str | None = None

    def open(self, device_name: str | None = None) -> str:
        """Open MIDI input port. Auto-detects if no name given. Returns name."""
        name = device_name or find_device()
        if name is None:
            raise RuntimeError("No MIDI input devices found")

        self._port = mido.open_input(name, callback=self._on_raw)
        self.device_name = name
        return name

    def _on_raw(self, msg: "mido.Message") -> None:
        """Convert raw mido Message to our MidiMessage types."""
        if msg.type == "note_on":
            if msg.velocity == 0:
                # Note On vel=0 is Note Off per MIDI spec
                self.callback(NoteOff(note=msg.note, channel=msg.channel))
            else:
                self.callback(
                    NoteOn(
                        note=msg.note, velocity=msg.velocity, channel=msg.channel
                    )
                )
        elif msg.type == "note_off":
            self.callback(NoteOff(note=msg.note, channel=msg.channel))
        elif msg.type == "control_change":
            self.callback(
                ControlChange(cc=msg.control, value=msg.value, channel=msg.channel)
            )
        # Pitch bend, aftertouch, etc. — ignored for now

    def close(self) -> None:
        if self._port:
            self._port.close()
            self._port = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
