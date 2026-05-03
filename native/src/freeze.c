/*
 * Granular freeze processor — loops a grain with windowed overlap.
 * 4 motion modes: HOLD, DRIFT, OSCILLATE, DECAY.
 */
#include "freeze.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_GRAIN 24000 /* 500ms at 48kHz */

void freeze_init(FreezeProc *fp, int sample_rate) {
    memset(fp, 0, sizeof(*fp));
    fp->sample_rate = sample_rate;
    fp->window = malloc(MAX_GRAIN * sizeof(float));
    /* Pre-compute max size Hann, we'll use a subset */
    for (int i = 0; i < MAX_GRAIN; i++)
        fp->window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / MAX_GRAIN));
}

void freeze_start(FreezeProc *fp, const float *source, int source_len,
                  float position, int grain_size, int motion,
                  float rate, float depth) {
    FreezeState *s = &fp->state;
    s->source = source;
    s->source_len = source_len;
    s->position = clampf(position, 0, 1);
    s->grain_size = grain_size < 64 ? 64 : (grain_size > MAX_GRAIN ? MAX_GRAIN : grain_size);
    s->motion = motion;
    s->rate = rate;
    s->depth = depth;
    s->active = 1;
    s->phase = 0;
    s->motion_pos = s->position;
    s->decay_gain = 1.0f;

    /* Recompute window for this grain size */
    for (int i = 0; i < s->grain_size; i++)
        fp->window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / s->grain_size));
}

void freeze_process(FreezeProc *fp, float *out, int n) {
    FreezeState *s = &fp->state;
    if (!s->active || !s->source) { memset(out, 0, n * sizeof(float)); return; }

    int gs = s->grain_size;
    int slen = s->source_len;
    float inv_sr = 1.0f / fp->sample_rate;

    for (int i = 0; i < n; i++) {
        /* Current read position in source */
        float center = s->motion_pos * slen;
        int grain_start = (int)center - gs / 2;

        /* Read grain sample with window */
        int gi = (int)s->phase;
        float sample = 0;
        if (gi >= 0 && gi < gs) {
            int src_idx = grain_start + gi;
            if (src_idx >= 0 && src_idx < slen) {
                sample = s->source[src_idx] * fp->window[gi];
            }
        }

        /* Second grain offset by half for overlap */
        int gi2 = (gi + gs / 2) % gs;
        int src_idx2 = grain_start + gi2;
        if (src_idx2 >= 0 && src_idx2 < slen) {
            sample += s->source[src_idx2] * fp->window[gi2];
        }

        out[i] = sample * s->decay_gain;

        /* Advance grain phase */
        s->phase += 1.0f;
        if (s->phase >= gs) s->phase -= gs;

        /* Motion update */
        switch (s->motion) {
            case MOTION_HOLD:
                break;
            case MOTION_DRIFT:
                s->motion_pos += s->rate * inv_sr;
                if (s->motion_pos > 1.0f) s->motion_pos = 1.0f;
                if (s->motion_pos < 0.0f) s->motion_pos = 0.0f;
                break;
            case MOTION_OSCILLATE: {
                float osc = sinf(s->phase * s->rate * 2.0f * M_PI * inv_sr);
                s->motion_pos = s->position + osc * s->depth;
                s->motion_pos = clampf(s->motion_pos, 0, 1);
                break;
            }
            case MOTION_DECAY:
                /* rate = decay time in seconds */
                if (s->rate > 0.01f)
                    s->decay_gain *= (1.0f - inv_sr / s->rate);
                if (s->decay_gain < 0.001f) {
                    s->active = 0;
                    break;
                }
                break;
        }
    }
}

void freeze_stop(FreezeProc *fp) {
    fp->state.active = 0;
}
