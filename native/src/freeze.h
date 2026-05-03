#pragma once
#include "types.h"

enum { MOTION_HOLD, MOTION_DRIFT, MOTION_OSCILLATE, MOTION_DECAY };

typedef struct {
    const float *source;
    int source_len;
    float position;    /* 0-1 fraction of source */
    int grain_size;    /* samples */
    int motion;
    float rate;        /* motion-dependent */
    float depth;       /* for oscillate */
    int active;
    /* Internal */
    float phase;       /* grain phase */
    float motion_pos;  /* drifting position */
    float decay_gain;  /* for decay motion */
} FreezeState;

typedef struct {
    FreezeState state;
    float *window;     /* Hann window for grain */
    int sample_rate;
} FreezeProc;

void freeze_init(FreezeProc *fp, int sample_rate);
void freeze_start(FreezeProc *fp, const float *source, int source_len,
                  float position, int grain_size, int motion,
                  float rate, float depth);
void freeze_process(FreezeProc *fp, float *out, int n);
void freeze_stop(FreezeProc *fp);
