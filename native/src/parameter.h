#pragma once
#include "types.h"

typedef enum { CURVE_LINEAR, CURVE_EXP, CURVE_LOG } ParamCurve;

typedef struct {
    const char *name;
    const char *label;
    float min_val, max_val, value, default_val;
    const char *unit;
    ParamCurve curve;
    int snap; /* 0=continuous, >0=discrete steps */
} Param;

static inline float param_mapped(const Param *p) {
    float t = p->value;
    if (p->curve == CURVE_EXP) {
        if (p->min_val <= 0) return p->min_val + t * (p->max_val - p->min_val);
        return p->min_val * powf(p->max_val / p->min_val, t);
    } else if (p->curve == CURVE_LOG) {
        if (t <= 0) return p->min_val;
        float lt = log10f(1.0f + 9.0f * t) / log10f(10.0f);
        return p->min_val + lt * (p->max_val - p->min_val);
    }
    return p->min_val + t * (p->max_val - p->min_val);
}

static inline void param_nudge(Param *p, float delta) {
    if (p->snap > 1) {
        float step = 1.0f / (p->snap - 1);
        int idx = (int)roundf(p->value / step);
        idx += (delta > 0) ? 1 : -1;
        p->value = clampf(idx * step, 0.0f, 1.0f);
    } else {
        p->value = clampf(p->value + delta, 0.0f, 1.0f);
    }
}

static inline void param_reset(Param *p) { p->value = p->default_val; }
