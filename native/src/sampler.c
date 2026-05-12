/*
 * Sampler engine — 16 pad slots, 8-voice polyphonic.
 * Modes: PAD, CLASSIC (pitched spread), CHOP (sliced).
 * Phase vocoder pitch shift, granular freeze, gross beat.
 */
#include "sampler.h"
#include "pitch_shift.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void slot_init_params(SampleSlot *s) {
    Param defaults[] = {
        {"mode",      "MODE",      0, 2,     0.0f, 0.0f, "",   CURVE_LINEAR, 3},
        {"pitch",     "PITCH",     -24, 24,  0.5f, 0.5f, "st", CURVE_LINEAR, 0},
        {"reverse",   "REVERSE",   0, 1,     0.0f, 0.0f, "",   CURVE_LINEAR, 2},
        {"gain",      "GAIN",      0, 2,     0.5f, 0.5f, "",   CURVE_LINEAR, 0},
        {"start",     "START",     0, 1,     0.0f, 0.0f, "%",  CURVE_LINEAR, 0},
        {"end",       "END",       0, 1,     1.0f, 1.0f, "%",  CURVE_LINEAR, 0},
        {"attack",    "ATTACK",    0.1f, 2000, 0.0f, 0.0f, "ms", CURVE_EXP, 0},
        {"decay",     "DECAY",     10, 10000, 1.0f, 1.0f, "ms", CURVE_EXP, 0},
        {"slices",    "SLICES",    1, 32,    0.1f, 0.1f, "",   CURVE_LINEAR, 32},
        {"filter",    "FILTER",    20, 20000, 1.0f, 1.0f, "Hz", CURVE_EXP, 0},
        {"fz_pos",    "FZ POS",    0, 1,     1.0f, 1.0f, "%",  CURVE_LINEAR, 0},
        {"fz_grain",  "FZ GRAIN",  20, 500,  0.4f, 0.4f, "ms", CURVE_EXP, 0},
        {"fz_motion", "FZ MOTION", 0, 3,     0.0f, 0.0f, "",   CURVE_LINEAR, 4},
        {"fz_rate",   "FZ RATE",   0, 1,     0.5f, 0.5f, "",   CURVE_LINEAR, 0},
    };
    memcpy(s->params, defaults, sizeof(defaults));
}

static void slot_clear_pitch_cache(SampleSlot *s) {
    for (int i = 0; i < SAMPLER_PITCH_CACHE_RANGE; i++) {
        if (s->pitch_cache[i]) { free(s->pitch_cache[i]); s->pitch_cache[i] = NULL; }
        s->pitch_cache_len[i] = 0;
    }
}

static float *slot_get_pitched(SampleSlot *s, int semitones) {
    if (!s->loaded || semitones == 0) return s->data;
    int ci = semitones + 12;
    if (ci < 0 || ci >= SAMPLER_PITCH_CACHE_RANGE) return s->data;
    if (s->pitch_cache[ci]) return s->pitch_cache[ci];

    float *shifted; int shifted_len;
    if (pvoc_pitch_shift(s->data, s->length, (float)semitones, &shifted, &shifted_len) == 0) {
        s->pitch_cache[ci] = shifted;
        s->pitch_cache_len[ci] = shifted_len;
        return shifted;
    }
    return s->data;
}

static void slot_refresh_slices(SampleSlot *s) {
    int n = (int)roundf(param_mapped(&s->params[SP_SLICES]));
    if (n < 1) n = 1; if (n > 32) n = 32;
    if (!s->loaded) return;
    /* Keep manual edits as long as the count param hasn't changed. */
    if (n == s->slice_count) return;
    int slice_w = s->length / n;
    for (int i = 0; i < n; i++) {
        s->slice_starts[i] = i * slice_w;
        s->slice_ends[i] = (i < n - 1) ? (i + 1) * slice_w : s->length;
    }
    s->slice_count = n;
    s->slice_manual = 0;
}

static int slot_mode(SampleSlot *s) {
    return (int)roundf(clampf(param_mapped(&s->params[SP_MODE]), 0, 2));
}

void sampler_init(SamplerEngine *s, int sample_rate) {
    memset(s, 0, sizeof(*s));
    for (int i = 0; i < SAMPLER_PAD_COUNT; i++) slot_init_params(&s->slots[i]);
    for (int i = 0; i < SAMPLER_VOICE_COUNT; i++) {
        adenv_init(&s->voices[i].amp_env);
        biquad_init(&s->voices[i].filt);
        freeze_init(&s->voices[i].freeze, sample_rate);
    }
    memset(s->note_map, -1, sizeof(s->note_map));
    grossbeat_init(&s->gross_beat, sample_rate);
    s->project_bpm = 120.0f;
}

void sampler_load_slot(SamplerEngine *s, int slot_idx, const char *path) {
    if (slot_idx < 0 || slot_idx >= SAMPLER_PAD_COUNT) return;
    SampleSlot *sl = &s->slots[slot_idx];
    sampler_clear_slot(s, slot_idx);

    WavData w;
    if (wav_load(path, &w) == 0) {
        sl->data = w.data;
        sl->length = w.length;
        sl->source_rate = w.sample_rate;
        sl->loaded = 1;
        strncpy(sl->path, path, sizeof(sl->path) - 1);
        /* Extract filename for name */
        const char *fname = strrchr(path, '/');
        fname = fname ? fname + 1 : path;
        strncpy(sl->name, fname, sizeof(sl->name) - 1);
    }
}

void sampler_clear_slot(SamplerEngine *s, int slot_idx) {
    if (slot_idx < 0 || slot_idx >= SAMPLER_PAD_COUNT) return;
    SampleSlot *sl = &s->slots[slot_idx];
    if (sl->data) { free(sl->data); sl->data = NULL; }
    slot_clear_pitch_cache(sl);
    sl->loaded = 0; sl->length = 0;
    sl->name[0] = 0; sl->path[0] = 0;
    sl->slice_count = 0;
}

static int allocate_voice(SamplerEngine *s) {
    for (int i = 0; i < SAMPLER_VOICE_COUNT; i++)
        if (!s->voices[i].active) return i;
    if (s->alloc_count > 0) {
        int stolen = s->alloc_order[0];
        memmove(s->alloc_order, s->alloc_order + 1, (s->alloc_count - 1) * sizeof(int));
        s->alloc_count--;
        for (int n = 0; n < 128; n++)
            if (s->note_map[n] == stolen) { s->note_map[n] = -1; break; }
        return stolen;
    }
    return 0;
}

static void touch_alloc(SamplerEngine *s, int vi) {
    for (int i = 0; i < s->alloc_count; i++)
        if (s->alloc_order[i] == vi) {
            memmove(s->alloc_order + i, s->alloc_order + i + 1,
                    (s->alloc_count - i - 1) * sizeof(int));
            s->alloc_count--;
            break;
        }
    s->alloc_order[s->alloc_count++] = vi;
}

/* Resolve which slot/mode/slice for a pad trigger */
typedef struct { SampleSlot *slot; int mode; int slice_idx; int semi_offset; } PadResolve;

static int resolve_pad(SamplerEngine *s, int pad_idx, PadResolve *r) {
    /* 1. Direct PAD */
    SampleSlot *direct = &s->slots[pad_idx];
    if (direct->loaded && slot_mode(direct) == SMODE_PAD) {
        r->slot = direct; r->mode = SMODE_PAD; r->slice_idx = -1; r->semi_offset = 0;
        return 1;
    }
    /* 2. Scan backward for CLASSIC or CHOP */
    for (int home = pad_idx; home >= 0; home--) {
        SampleSlot *sl = &s->slots[home];
        if (!sl->loaded) continue;
        int offset = pad_idx - home;
        int m = slot_mode(sl);
        if (m == SMODE_CLASSIC && offset >= 0) {
            r->slot = sl; r->mode = SMODE_CLASSIC; r->slice_idx = -1; r->semi_offset = offset;
            return 1;
        }
        if (m == SMODE_CHOP) {
            slot_refresh_slices(sl);
            if (offset >= 0 && offset < sl->slice_count) {
                r->slot = sl; r->mode = SMODE_CHOP; r->slice_idx = offset; r->semi_offset = 0;
                return 1;
            }
        }
    }
    /* Fallback: direct slot any mode */
    if (direct->loaded) {
        int m = slot_mode(direct);
        r->slot = direct; r->mode = m; r->slice_idx = (m == SMODE_CHOP) ? 0 : -1; r->semi_offset = 0;
        if (m == SMODE_CHOP) slot_refresh_slices(direct);
        return 1;
    }
    return 0;
}

static void voice_trigger(SamplerVoice *v, SampleSlot *sl, int note, float vel,
                          float *buffer, int buf_len, int slice_start, int slice_end) {
    v->slot = sl;
    v->note = note;
    v->velocity = clampf(vel, 0, 1);
    v->buffer = buffer;
    v->buf_len = buf_len;

    Param *p = sl->params;
    if (slice_start >= 0 && slice_end > 0) {
        v->start_pos = (float)slice_start;
        v->end_pos = (float)slice_end;
    } else {
        float sf = clampf(param_mapped(&p[SP_START]), 0, 1);
        float ef = clampf(param_mapped(&p[SP_END]), 0, 1);
        if (ef <= sf) ef = sf + 0.001f;
        v->start_pos = sf * buf_len;
        v->end_pos = ef * buf_len;
    }

    v->reversed = (int)roundf(param_mapped(&p[SP_REVERSE]));
    v->position = v->reversed ? v->end_pos - 1 : v->start_pos;
    v->rate = (float)sl->source_rate / SR;
    v->gain = param_mapped(&p[SP_GAIN]) * v->velocity;
    v->cutoff_hz = param_mapped(&p[SP_FILTER]);

    v->amp_env.attack = param_mapped(&p[SP_ATTACK]) / 1000.0f;
    v->amp_env.decay = param_mapped(&p[SP_DECAY]) / 1000.0f;
    v->amp_env.sustain = 1.0f;
    v->amp_env.release = 0.02f;
    adenv_trigger(&v->amp_env);
    v->freeze_released = 0;

    if (v->cutoff_hz < 18000.0f) biquad_set(&v->filt, FILT_LP, v->cutoff_hz, 0.707f);
    v->active = 1;
}

void sampler_trigger(SamplerEngine *s, int note, float vel) {
    int pad_idx = note - SAMPLER_PAD_MIDI_START;
    if (pad_idx < 0 || pad_idx >= SAMPLER_PAD_COUNT) return;

    PadResolve r;
    if (!resolve_pad(s, pad_idx, &r)) return;

    int manual_pitch = (int)roundf(param_mapped(&r.slot->params[SP_PITCH]));
    int total_semi = r.semi_offset + manual_pitch;
    total_semi = total_semi < -12 ? -12 : (total_semi > 12 ? 12 : total_semi);

    float *buf = slot_get_pitched(r.slot, total_semi);
    int buf_len = (total_semi != 0 && r.slot->pitch_cache[total_semi + 12])
                  ? r.slot->pitch_cache_len[total_semi + 12] : r.slot->length;

    int vi = (s->note_map[note] >= 0) ? s->note_map[note] : allocate_voice(s);
    SamplerVoice *v = &s->voices[vi];

    if (r.mode == SMODE_CHOP && r.slice_idx >= 0 && r.slice_idx < r.slot->slice_count) {
        voice_trigger(v, r.slot, note, vel, buf, buf_len,
                      r.slot->slice_starts[r.slice_idx], r.slot->slice_ends[r.slice_idx]);
    } else {
        voice_trigger(v, r.slot, note, vel, buf, buf_len, -1, -1);
    }

    s->note_map[note] = vi;
    touch_alloc(s, vi);
}

void sampler_release(SamplerEngine *s, int note) {
    int vi = s->note_map[note];
    if (vi >= 0 && vi < SAMPLER_VOICE_COUNT) {
        adenv_release(&s->voices[vi].amp_env);
        if (s->voices[vi].freeze.state.active) {
            freeze_stop(&s->voices[vi].freeze);
        }
        s->voices[vi].freeze_released = 1;
        s->note_map[note] = -1;
    }
}

static void voice_process(SamplerVoice *v, float *out, int n) {
    if (!v->active) { memset(out, 0, n * sizeof(float)); return; }

    /* Freeze takeover */
    if (v->freeze.state.active) {
        freeze_process(&v->freeze, out, n);
        for (int i = 0; i < n; i++) {
            float env = adenv_tick(&v->amp_env);
            if (v->cutoff_hz < 18000.0f) out[i] = biquad_tick(&v->filt, out[i]);
            out[i] *= env * v->gain;
            if (v->amp_env.stage == ENV_IDLE && !v->freeze.state.active) {
                v->active = 0; break;
            }
        }
        return;
    }

    float *buf = v->buffer;
    int buf_len = v->buf_len;
    int hit_end = 0;

    for (int i = 0; i < n; i++) {
        if (v->reversed) {
            if (v->position <= v->start_pos) { hit_end = 1; break; }
        } else {
            if (v->position >= v->end_pos - 1) { hit_end = 1; break; }
        }

        int idx = (int)v->position;
        float frac = v->position - idx;
        float sample = 0;
        if (idx >= 0 && idx < buf_len - 1) {
            sample = buf[idx] + frac * (buf[idx + 1] - buf[idx]);
        }

        float env = adenv_tick(&v->amp_env);
        if (v->cutoff_hz < 18000.0f) sample = biquad_tick(&v->filt, sample);
        out[i] = sample * env * v->gain;

        v->position += v->reversed ? -v->rate : v->rate;
    }

    if (hit_end) {
        if (v->slot && v->slot->freeze_armed && !v->freeze_released) {
            /* Engage freeze */
            Param *p = v->slot->params;
            float pos = param_mapped(&p[SP_FZ_POS]);
            int grain = (int)(param_mapped(&p[SP_FZ_GRAIN]) * SR / 1000.0f);
            int motion = (int)roundf(param_mapped(&p[SP_FZ_MOTION]));
            float rate_norm = param_mapped(&p[SP_FZ_RATE]);
            float rate = 0, depth = 0;
            switch (motion) {
                case MOTION_DRIFT: rate = rate_norm * 0.5f; break;
                case MOTION_OSCILLATE: rate = 0.1f * powf(60.0f, rate_norm); depth = 0.15f; break;
                case MOTION_DECAY: rate = 0.1f * powf(300.0f, rate_norm); break;
            }
            freeze_start(&v->freeze, v->buffer, v->buf_len, pos, grain, motion, rate, depth);
        } else {
            adenv_release(&v->amp_env);
        }
    }

    if (v->amp_env.stage == ENV_IDLE && !v->freeze.state.active) {
        v->active = 0;
    }
}

void sampler_process(SamplerEngine *s, float *buf, int n) {
    memset(buf, 0, n * sizeof(float));
    float voice_buf[BLOCK_SIZE];
    int active = 0;

    s->gross_beat.bpm = s->project_bpm;

    for (int i = 0; i < SAMPLER_VOICE_COUNT; i++) {
        if (!s->voices[i].active) continue;
        voice_process(&s->voices[i], voice_buf, n);
        for (int j = 0; j < n; j++) buf[j] += voice_buf[j];
        active++;
    }
    if (active > 1) {
        float scale = 1.0f / sqrtf((float)active);
        for (int i = 0; i < n; i++) buf[i] *= scale;
    }
    grossbeat_process(&s->gross_beat, buf, n);
}

int sampler_active(SamplerEngine *s) {
    for (int i = 0; i < SAMPLER_VOICE_COUNT; i++)
        if (s->voices[i].active) return 1;
    return 0;
}

void sampler_ensure_slices(SamplerEngine *s, int slot_idx) {
    if (slot_idx < 0 || slot_idx >= SAMPLER_PAD_COUNT) return;
    slot_refresh_slices(&s->slots[slot_idx]);
}

void sampler_set_slice(SamplerEngine *s, int slot_idx, int slice_idx,
                       float start_frac, float end_frac) {
    if (slot_idx < 0 || slot_idx >= SAMPLER_PAD_COUNT) return;
    SampleSlot *sl = &s->slots[slot_idx];
    if (!sl->loaded || sl->length <= 0) return;
    if (slice_idx < 0 || slice_idx >= sl->slice_count) return;
    if (start_frac < 0) start_frac = 0; if (start_frac > 1) start_frac = 1;
    if (end_frac   < 0) end_frac   = 0; if (end_frac   > 1) end_frac   = 1;
    int st = (int)(start_frac * sl->length);
    int en = (int)(end_frac   * sl->length);
    if (en <= st) en = st + 1;
    if (en > sl->length) en = sl->length;
    if (st >= en) st = en - 1;
    if (st < 0) st = 0;
    sl->slice_starts[slice_idx] = st;
    sl->slice_ends[slice_idx]   = en;
    sl->slice_manual = 1;
}

void sampler_reset_slices(SamplerEngine *s, int slot_idx) {
    if (slot_idx < 0 || slot_idx >= SAMPLER_PAD_COUNT) return;
    SampleSlot *sl = &s->slots[slot_idx];
    sl->slice_count = 0;
    sl->slice_manual = 0;
    slot_refresh_slices(sl);
}
