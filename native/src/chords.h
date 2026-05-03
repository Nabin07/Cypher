#pragma once
#include <string.h>
#include <stdio.h>
#include "types.h"

#define MAX_CHORD_NOTES 5
#define MAX_PROG_STEPS 8
#define NUM_CHORD_TYPES 12
#define NUM_PROGRESSIONS 6

typedef struct {
    const char *name;
    int intervals[MAX_CHORD_NOTES];
    int count;
} ChordType;

typedef struct {
    int semitone_offset;
    int chord_type_idx;
} ProgStep;

typedef struct {
    const char *name;
    ProgStep steps[MAX_PROG_STEPS];
    int length;
} Progression;

static const char *NOTE_NAMES_C[] = {
    "C","C#","D","D#","E","F","F#","G","G#","A","A#","B"
};

static const ChordType CHORD_TYPES[] = {
    {"MAJ",  {0,4,7},        3},
    {"MIN",  {0,3,7},        3},
    {"MAJ7", {0,4,7,11},     4},
    {"MIN7", {0,3,7,10},     4},
    {"DOM7", {0,4,7,10},     4},
    {"DIM",  {0,3,6},        3},
    {"AUG",  {0,4,8},        3},
    {"SUS2", {0,2,7},        3},
    {"SUS4", {0,5,7},        3},
    {"PWR",  {0,7},          2},
    {"MIN9", {0,3,7,10,14},  5},
    {"MAJ9", {0,4,7,11,14},  5},
};

/* Chord type name to index */
static inline int chord_type_idx(const char *name) {
    for (int i = 0; i < NUM_CHORD_TYPES; i++)
        if (strcmp(CHORD_TYPES[i].name, name) == 0) return i;
    return 0;
}

/* Progressions — minor key, trap territory */
static const Progression PROGRESSIONS[] = {
    {"TRAP I",  {{0,1},{8,0},{3,0},{10,0}}, 4},   /* i-VI-III-VII */
    {"TRAP II", {{0,1},{5,1},{8,0},{7,0}},  4},    /* i-iv-VI-V */
    {"DARK",    {{0,1},{10,0},{8,0},{10,0}},4},    /* i-VII-VI-VII */
    {"SAD BOI", {{0,3},{8,2},{3,2},{10,4}}, 4},    /* i7-VImaj7-IIImaj7-VII7 */
    {"BOUNCE",  {{0,1},{3,0},{5,1},{8,0}},  4},    /* i-III-iv-VI */
    {"DRILL",   {{0,1},{7,1},{5,1},{3,0}},  4},    /* i-v-iv-III */
};

/* Build a chord: fills notes[], returns count */
static inline int build_chord(int root_midi, int chord_type_idx_v, int *notes) {
    const ChordType *ct = &CHORD_TYPES[chord_type_idx_v];
    for (int i = 0; i < ct->count; i++)
        notes[i] = root_midi + ct->intervals[i];
    return ct->count;
}

/* Build chord from progression step */
static inline int build_prog_chord(int key_midi, int prog_idx, int step, int *notes, char *label, int label_sz) {
    const Progression *p = &PROGRESSIONS[prog_idx % NUM_PROGRESSIONS];
    step = step % p->length;
    int offset = p->steps[step].semitone_offset;
    int ct_idx = p->steps[step].chord_type_idx;
    int root = key_midi + offset;
    int count = build_chord(root, ct_idx, notes);

    const char *rn = NOTE_NAMES_C[root % 12];
    const char *tn = CHORD_TYPES[ct_idx].name;
    if (ct_idx == 0) snprintf(label, label_sz, "%s", rn);
    else if (ct_idx == 1) snprintf(label, label_sz, "%sm", rn);
    else snprintf(label, label_sz, "%s%s", rn, tn);

    return count;
}

static inline int prog_length(int prog_idx) {
    return PROGRESSIONS[prog_idx % NUM_PROGRESSIONS].length;
}
