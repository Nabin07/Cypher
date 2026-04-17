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
    # ── Trap / hip hop ──
    "TRAP I":    [(0, "MIN"), (8, "MAJ"), (3, "MAJ"), (10, "MAJ")],        # i-VI-III-VII
    "TRAP II":   [(0, "MIN"), (5, "MIN"), (8, "MAJ"), (7, "MAJ")],         # i-iv-VI-V
    "DRILL":     [(0, "MIN"), (7, "MIN"), (5, "MIN"), (3, "MAJ")],         # i-v-iv-III
    "BOUNCE":    [(0, "MIN"), (3, "MAJ"), (5, "MIN"), (8, "MAJ")],         # i-III-iv-VI

    # ── Dark / cinematic ──
    "DARK":      [(0, "MIN"), (10, "MAJ"), (8, "MAJ"), (10, "MAJ")],       # i-VII-VI-VII
    "HORROR":    [(0, "MIN"), (6, "DIM"), (8, "MAJ"), (11, "DIM")],        # i-ii°-VI-vii°
    "EPIC":      [(0, "MIN"), (3, "MAJ"), (10, "MAJ"), (5, "MIN")],        # i-III-VII-iv
    "ANIME":     [(5, "MAJ"), (7, "MAJ"), (3, "MIN"), (8, "MIN")],         # IV-V-iii-vi

    # ── Sad / emotional ──
    "SAD BOI":   [(0, "MIN7"), (8, "MAJ7"), (3, "MAJ7"), (10, "DOM7")],    # i7-VImaj7-IIImaj7-VII7
    "TEARS":     [(0, "MIN"), (8, "MAJ"), (3, "MAJ"), (7, "MIN")],         # i-VI-III-v
    "LOSS":      [(0, "MIN9"), (5, "MIN7"), (10, "DOM7"), (3, "MAJ7")],    # i9-iv7-VII7-IIImaj7
    "REGRET":    [(0, "MIN"), (3, "MAJ"), (7, "MIN"), (5, "MIN")],         # i-III-v-iv

    # ── Dreamy / ambient ──
    "DREAM":     [(0, "MAJ7"), (4, "MIN7"), (9, "MIN7"), (5, "MAJ7")],     # Imaj7-iii7-vi7-IVmaj7
    "FLOAT":     [(0, "MAJ9"), (7, "SUS2"), (9, "MIN7"), (5, "MAJ7")],    # Imaj9-Vsus2-vi7-IVmaj7
    "LOFI":      [(0, "MAJ7"), (4, "MIN7"), (9, "MIN7"), (2, "MIN7")],     # Imaj7-iii7-vi7-ii7
    "MIST":      [(0, "SUS2"), (5, "SUS2"), (7, "SUS2"), (9, "MIN")],      # I-IV-V-vi (sus)

    # ── Pop / happy ──
    "POP":       [(0, "MAJ"), (7, "MAJ"), (9, "MIN"), (5, "MAJ")],         # I-V-vi-IV
    "DOOWOP":    [(0, "MAJ"), (9, "MIN"), (5, "MAJ"), (7, "MAJ")],         # I-vi-IV-V
    "SUNSHINE":  [(0, "MAJ"), (5, "MAJ"), (7, "MAJ"), (0, "MAJ")],         # I-IV-V-I

    # ── Jazz / soul ──
    "JAZZ":      [(2, "MIN7"), (7, "DOM7"), (0, "MAJ7"), (0, "MAJ7")],    # ii7-V7-Imaj7
    "RNB":       [(0, "MAJ7"), (4, "MIN7"), (2, "MIN7"), (5, "MAJ7")],    # Imaj7-iii7-ii7-IVmaj7
    "NEOSOUL":   [(0, "MAJ9"), (5, "MAJ7"), (4, "MIN7"), (9, "MIN9")],    # Imaj9-IVmaj7-iii7-vi9

    # ── House / dance ──
    "HOUSE":     [(0, "MIN"), (8, "MAJ"), (10, "MAJ"), (0, "MIN")],        # i-VI-VII-i
    "UPLIFT":    [(0, "SUS2"), (5, "SUS2"), (7, "MAJ"), (9, "MIN")],       # I-IV-V-vi

    # ── Funk / groove ──
    "FUNK":      [(0, "DOM7"), (5, "DOM7"), (0, "DOM7"), (7, "DOM7")],    # I7-IV7-I7-V7
    "GROOVE":    [(0, "MIN7"), (5, "DOM7"), (0, "MIN7"), (10, "DOM7")],   # i7-IV7-i7-VII7
}

PROGRESSION_LIST: list[str] = list(PROGRESSIONS.keys())


# Scales — semitone offsets from the root. Used to highlight the
# "safe" melody notes on the keyboard visualizer when a key is chosen.
SCALES: dict[str, list[int]] = {
    "MINOR":      [0, 2, 3, 5, 7, 8, 10],       # natural minor
    "MAJOR":      [0, 2, 4, 5, 7, 9, 11],       # major
    "DORIAN":     [0, 2, 3, 5, 7, 9, 10],       # minor with raised 6
    "PHRYGIAN":   [0, 1, 3, 5, 7, 8, 10],       # dark minor (spanish/metal)
    "LYDIAN":     [0, 2, 4, 6, 7, 9, 11],       # major with raised 4
    "MIXOLYDIAN": [0, 2, 4, 5, 7, 9, 10],       # major with flat 7 (funk/blues)
    "HARM MIN":   [0, 2, 3, 5, 7, 8, 11],       # harmonic minor (bollywood/middle-east)
    "PENT MIN":   [0, 3, 5, 7, 10],             # minor pentatonic
    "PENT MAJ":   [0, 2, 4, 7, 9],              # major pentatonic
    "BLUES":      [0, 3, 5, 6, 7, 10],          # blues (minor pent + blue note)
}

SCALE_LIST: list[str] = list(SCALES.keys())


def is_note_in_scale(midi_note: int, root_midi: int, scale_name: str) -> bool:
    """Check if a MIDI note belongs to the given scale/key."""
    intervals = SCALES.get(scale_name, SCALES["MINOR"])
    return ((midi_note - root_midi) % 12) in intervals


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
