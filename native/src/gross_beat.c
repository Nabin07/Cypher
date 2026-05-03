/*
 * Gross Beat — volume pattern presets synced to BPM.
 * Applied to the mixed sampler output.
 */
#include "gross_beat.h"
#include <math.h>
#include <string.h>

void grossbeat_init(GrossBeat *gb, int sample_rate) {
    gb->preset = GB_OFF;
    gb->bpm = 120.0f;
    gb->phase = 0;
    gb->sample_rate = sample_rate;
}

/* Get volume multiplier at a given beat phase (0-1 = one beat) */
static float pattern_vol(int preset, float phase) {
    float p = fmodf(phase, 1.0f);
    switch (preset) {
        case GB_OFF: return 1.0f;
        case GB_HALF:
            /* Half-time: full volume for first half, silence second half (per 2 beats) */
            return fmodf(phase, 2.0f) < 1.0f ? 1.0f : 0.0f;
        case GB_GATE:
            /* 1/8 gate */
            return fmodf(phase * 2.0f, 1.0f) < 0.5f ? 1.0f : 0.0f;
        case GB_STUTTER:
            /* 1/16 rapid stutter */
            return fmodf(phase * 4.0f, 1.0f) < 0.5f ? 1.0f : 0.0f;
        case GB_TRIPLET:
            /* Triplet gate */
            return fmodf(phase * 3.0f, 1.0f) < 0.5f ? 1.0f : 0.0f;
        case GB_SIDECHAIN:
            /* Fake sidechain pump — exponential decay per beat */
            return 1.0f - expf(-4.0f * p);
        case GB_TAPE_STOP:
            /* Gradual slowdown feel — volume dip */
            return 0.3f + 0.7f * (1.0f - p * p);
        default: return 1.0f;
    }
}

void grossbeat_process(GrossBeat *gb, float *buf, int n) {
    if (gb->preset == GB_OFF) return;
    float beat_inc = gb->bpm / (60.0f * gb->sample_rate);

    for (int i = 0; i < n; i++) {
        float vol = pattern_vol(gb->preset, gb->phase);
        buf[i] *= vol;
        gb->phase += beat_inc;
        /* Keep phase in reasonable range */
        if (gb->phase > 1000.0f) gb->phase -= 1000.0f;
    }
}
