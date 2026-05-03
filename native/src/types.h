#pragma once
#include <stdint.h>
#include <math.h>

#define SR 48000
#define BLOCK_SIZE 1024
#define TWO_PI (2.0f * M_PI)
#define MAX_POLY 8

typedef float sample_t;

static inline float note_to_freq(int note) {
    return 440.0f * powf(2.0f, (note - 69) / 12.0f);
}

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline float lerpf(float a, float b, float t) {
    return a + t * (b - a);
}
