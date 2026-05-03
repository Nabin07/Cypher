#pragma once
#include "types.h"

/* ── AD Envelope (synth amp/filter) ── */
enum { ENV_IDLE, ENV_ATTACK, ENV_DECAY, ENV_SUSTAIN, ENV_RELEASE };

typedef struct {
    int stage;
    float level;
    float retrigger_level;
    float attack, decay, sustain, release;
    float curve;
    int pos;
} ADEnv;

static inline void adenv_init(ADEnv *e) {
    e->stage = ENV_IDLE; e->level = 0; e->retrigger_level = 0;
    e->attack = 0.005f; e->decay = 0.3f; e->sustain = 0.7f;
    e->release = 0.2f; e->curve = -4.0f; e->pos = 0;
}

static inline void adenv_trigger(ADEnv *e) {
    e->retrigger_level = e->level;
    e->stage = ENV_ATTACK; e->pos = 0;
}

static inline void adenv_release(ADEnv *e) {
    if (e->stage != ENV_IDLE) {
        e->stage = ENV_RELEASE; e->pos = 0;
    }
}

static inline float adenv_tick(ADEnv *e) {
    if (e->stage == ENV_IDLE) return 0;
    float t;
    int samps;
    switch (e->stage) {
        case ENV_ATTACK:
            samps = (int)(e->attack * SR);
            if (samps < 1) samps = 1;
            t = (float)e->pos / samps;
            e->level = e->retrigger_level + (1.0f - e->retrigger_level) * t;
            if (++e->pos >= samps) {
                e->level = 1.0f;
                e->stage = (e->sustain > 0) ? ENV_SUSTAIN : ENV_DECAY;
                e->pos = 0;
            }
            break;
        case ENV_DECAY:
            samps = (int)(e->decay * SR);
            if (samps < 1) samps = 1;
            t = (float)e->pos / samps;
            e->level = e->sustain + (1.0f - e->sustain) * expf(e->curve * t);
            if (++e->pos >= samps || e->level < 0.001f) {
                e->level = e->sustain > 0.001f ? e->sustain : 0;
                e->stage = e->sustain > 0.001f ? ENV_SUSTAIN : ENV_IDLE;
                e->pos = 0;
            }
            break;
        case ENV_SUSTAIN:
            e->level = e->sustain;
            break;
        case ENV_RELEASE:
            samps = (int)(e->release * SR);
            if (samps < 1) samps = 1;
            t = (float)e->pos / samps;
            e->level = e->retrigger_level * expf(e->curve * t);
            if (++e->pos >= samps || e->level < 0.001f) {
                e->level = 0; e->stage = ENV_IDLE;
            }
            break;
    }
    return e->level;
}

/* ── Trap Envelope (808 sub bass) ── */
enum { TRAP_IDLE, TRAP_ATTACK, TRAP_HOLD, TRAP_DECAY1, TRAP_SUSTAIN, TRAP_DECAY2 };

typedef struct {
    int stage, pos;
    float level, release_level;
    float attack, hold, decay1, sustain_level, decay2;
    float d1_curve, d2_curve;
    int gate_mode;
} TrapEnv;

static inline void trapenv_init(TrapEnv *e) {
    e->stage = TRAP_IDLE; e->level = 0; e->pos = 0;
    e->attack = 0.003f; e->hold = 0.005f;
    e->decay1 = 0.05f; e->sustain_level = 0.75f; e->decay2 = 2.0f;
    e->d1_curve = -3.0f; e->d2_curve = -2.5f; e->gate_mode = 0;
}

static inline void trapenv_trigger(TrapEnv *e) {
    e->stage = TRAP_ATTACK; e->pos = 0;
    e->release_level = e->level;
}

static inline void trapenv_release(TrapEnv *e) {
    if (e->stage != TRAP_IDLE) {
        e->stage = TRAP_DECAY2; e->release_level = e->level; e->pos = 0;
    }
}

static inline float trapenv_tick(TrapEnv *e) {
    if (e->stage == TRAP_IDLE) return 0;
    int samps; float t;
    switch (e->stage) {
        case TRAP_ATTACK:
            samps = (int)(e->attack * SR); if (samps < 1) samps = 1;
            t = (float)e->pos / samps;
            e->level = e->release_level + (1.0f - e->release_level) * t;
            if (++e->pos >= samps) { e->level = 1.0f; e->stage = TRAP_HOLD; e->pos = 0; }
            break;
        case TRAP_HOLD:
            samps = (int)(e->hold * SR); if (samps < 1) samps = 1;
            e->level = 1.0f;
            if (++e->pos >= samps) { e->stage = TRAP_DECAY1; e->pos = 0; }
            break;
        case TRAP_DECAY1:
            samps = (int)(e->decay1 * SR); if (samps < 1) samps = 1;
            t = (float)e->pos / samps;
            e->level = 1.0f - (1.0f - e->sustain_level) * (1.0f - expf(e->d1_curve * t));
            if (++e->pos >= samps) {
                e->level = e->sustain_level;
                e->stage = e->gate_mode ? TRAP_SUSTAIN : TRAP_DECAY2;
                e->release_level = e->sustain_level; e->pos = 0;
            }
            break;
        case TRAP_SUSTAIN:
            e->level = e->sustain_level;
            break;
        case TRAP_DECAY2:
            samps = (int)(e->decay2 * SR); if (samps < 1) samps = 1;
            t = (float)e->pos / samps;
            e->level = e->release_level * expf(e->d2_curve * t);
            if (++e->pos >= samps || e->level < 0.001f) { e->level = 0; e->stage = TRAP_IDLE; }
            break;
    }
    return e->level;
}

/* ── Pitch Envelope (808 sweep) ── */
typedef struct {
    float current_hz, end_hz, start_hz;
    float slide_time;
    int pos, mode; /* 0=idle, 1=sweep, 2=settled */
} PitchEnv;

static inline void pitchenv_init(PitchEnv *e) {
    e->current_hz = 50; e->end_hz = 50; e->start_hz = 200;
    e->slide_time = 0.05f; e->pos = 0; e->mode = 0;
}

static inline void pitchenv_trigger(PitchEnv *e, float target_hz) {
    e->end_hz = target_hz; e->mode = 1; e->pos = 0; e->current_hz = e->start_hz;
}

static inline float pitchenv_tick(PitchEnv *e) {
    if (e->mode == 0 || e->mode == 2) { e->current_hz = e->end_hz; return e->end_hz; }
    int samps = (int)(e->slide_time * SR); if (samps < 1) samps = 1;
    float t = (float)e->pos / samps;
    e->current_hz = e->end_hz + (e->start_hz - e->end_hz) * expf(-5.0f * t);
    if (++e->pos >= samps) { e->mode = 2; e->current_hz = e->end_hz; }
    return e->current_hz;
}
