"""Chord definitions and progressions for the synth engine.

Intervals in semitones from root. Progressions use semitone offsets
from the key root so they transpose cleanly.
"""

from __future__ import annotations

# Chord intervals (semitones from root)
CHORD_TYPES: dict[str, list[int]] = {
    "MAJ":  [0, 4, 7],
    "MIN":  [0, 3, 7],
    "MAJ7": [0, 4, 7, 11],
    "MIN7": [0, 3, 7, 10],
    "DOM7": [0, 4, 7, 10],
    "DIM":  [0, 3, 6],
    "AUG":  [0, 4, 8],
    "SUS2": [0, 2, 7],
    "SUS4": [0, 5, 7],
    "PWR":  [0, 7],
    "MIN9": [0, 3, 7, 10, 14],
    "MAJ9": [0, 4, 7, 11, 14],
}

CHORD_TYPE_LIST: list[str] = list(CHORD_TYPES.keys())

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Progressions: (semitone_offset_from_key, chord_type)
# All assume minor key root — that's where trap lives.
#
# Natural minor scale degrees -> semitone offsets:
#   i=0  ii°=2  III=3  iv=5  v=7  VI=8  VII=10
PROGRESSIONS: dict[str, list[tuple[int, str]]] = {
    "TRAP I":   [(0, "MIN"), (8, "MAJ"), (3, "MAJ"), (10, "MAJ")],      # i - VI - III - VII
    "TRAP II":  [(0, "MIN"), (5, "MIN"), (8, "MAJ"), (7, "MAJ")],       # i - iv - VI - V
    "DARK":     [(0, "MIN"), (10, "MAJ"), (8, "MAJ"), (10, "MAJ")],     # i - VII - VI - VII
    "SAD BOI":  [(0, "MIN7"), (8, "MAJ7"), (3, "MAJ7"), (10, "DOM7")],  # i7 - VImaj7 - IIImaj7 - VII7
    "BOUNCE":   [(0, "MIN"), (3, "MAJ"), (5, "MIN"), (8, "MAJ")],       # i - III - iv - VI
    "DRILL":    [(0, "MIN"), (7, "MIN"), (5, "MIN"), (3, "MAJ")],       # i - v - iv - III
}

PROGRESSION_LIST: list[str] = list(PROGRESSIONS.keys())


def build_chord(root_midi: int, chord_type: str) -> list[int]:
    """Build a chord from a root MIDI note and chord type name."""
    intervals = CHORD_TYPES.get(chord_type, CHORD_TYPES["MAJ"])
    return [root_midi + i for i in intervals]


def build_progression_chord(
    key_midi: int, progression_name: str, step: int
) -> tuple[list[int], str]:
    """Build a chord from a progression step.

    Returns (midi_notes, label) e.g. ([60, 63, 67], "Cmin").
    """
    prog = PROGRESSIONS.get(progression_name, PROGRESSIONS["TRAP I"])
    step = step % len(prog)
    offset, chord_type = prog[step]
    root = key_midi + offset
    notes = build_chord(root, chord_type)

    root_name = NOTE_NAMES[root % 12]
    if chord_type == "MAJ":
        label = root_name
    elif chord_type == "MIN":
        label = f"{root_name}m"
    else:
        label = f"{root_name}{chord_type.lower()}"

    return notes, label


def progression_length(progression_name: str) -> int:
    """Number of chords in a progression."""
    return len(PROGRESSIONS.get(progression_name, []))
