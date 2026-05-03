#pragma once
#include "types.h"

enum {
    GB_OFF, GB_HALF, GB_GATE, GB_STUTTER, GB_TRIPLET,
    GB_SIDECHAIN, GB_TAPE_STOP, GB_COUNT
};

typedef struct {
    int preset;
    float bpm;
    float phase; /* 0-1 beat position */
    int sample_rate;
} GrossBeat;

void grossbeat_init(GrossBeat *gb, int sample_rate);
void grossbeat_process(GrossBeat *gb, float *buf, int n);
