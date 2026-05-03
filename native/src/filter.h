#pragma once
#include "types.h"

enum { FILT_LP, FILT_HP, FILT_BP };

typedef struct {
    float b0, b1, b2, a1, a2;
    float x1, x2, y1, y2;
    int mode;
} Biquad;

static inline void biquad_init(Biquad *f) {
    f->b0 = 1; f->b1 = f->b2 = f->a1 = f->a2 = 0;
    f->x1 = f->x2 = f->y1 = f->y2 = 0;
    f->mode = FILT_LP;
}

static inline void biquad_set(Biquad *f, int mode, float freq, float q) {
    float w0 = TWO_PI * clampf(freq, 20.0f, SR * 0.49f) / SR;
    float alpha = sinf(w0) / (2.0f * clampf(q, 0.1f, 20.0f));
    float cs = cosf(w0);
    float a0;
    f->mode = mode;
    switch (mode) {
        case FILT_LP:
            f->b0 = (1.0f - cs) * 0.5f;
            f->b1 = 1.0f - cs;
            f->b2 = (1.0f - cs) * 0.5f;
            a0 = 1.0f + alpha;
            f->a1 = -2.0f * cs;
            f->a2 = 1.0f - alpha;
            break;
        case FILT_HP:
            f->b0 = (1.0f + cs) * 0.5f;
            f->b1 = -(1.0f + cs);
            f->b2 = (1.0f + cs) * 0.5f;
            a0 = 1.0f + alpha;
            f->a1 = -2.0f * cs;
            f->a2 = 1.0f - alpha;
            break;
        case FILT_BP:
            f->b0 = alpha;
            f->b1 = 0;
            f->b2 = -alpha;
            a0 = 1.0f + alpha;
            f->a1 = -2.0f * cs;
            f->a2 = 1.0f - alpha;
            break;
        default: return;
    }
    float inv = 1.0f / a0;
    f->b0 *= inv; f->b1 *= inv; f->b2 *= inv;
    f->a1 *= inv; f->a2 *= inv;
}

static inline float biquad_tick(Biquad *f, float in) {
    float out = f->b0 * in + f->b1 * f->x1 + f->b2 * f->x2
              - f->a1 * f->y1 - f->a2 * f->y2;
    f->x2 = f->x1; f->x1 = in;
    f->y2 = f->y1; f->y1 = out;
    return out;
}
