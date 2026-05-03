#pragma once
#include <stdlib.h>
#include "types.h"

enum { WAVE_SAW, WAVE_SINE, WAVE_SQUARE, WAVE_TRI, WAVE_COUNT };

typedef struct {
    float phase;
    float phase_inc;
    int waveform;
} Osc;

static inline void osc_init(Osc *o) {
    o->phase = 0; o->phase_inc = 0; o->waveform = WAVE_SAW;
}

static inline void osc_set_freq(Osc *o, float freq) {
    o->phase_inc = freq / (float)SR;
}

static inline float osc_tick(Osc *o) {
    float out;
    float p = o->phase;
    switch (o->waveform) {
        case WAVE_SAW:    out = 2.0f * p - 1.0f; break;
        case WAVE_SINE:   out = sinf(TWO_PI * p); break;
        case WAVE_SQUARE: out = p < 0.5f ? 1.0f : -1.0f; break;
        case WAVE_TRI:    out = 4.0f * fabsf(p - 0.5f) - 1.0f; break;
        default:          out = 0; break;
    }
    o->phase += o->phase_inc;
    if (o->phase >= 1.0f) o->phase -= 1.0f;
    return out;
}

/* Noise generator */
static inline float noise_tick(void) {
    return (float)rand() / (float)(RAND_MAX / 2) - 1.0f;
}
