#pragma once
#include "types.h"
#include "envelope.h"
#include "oscillator.h"
#include "parameter.h"

#define SUB808_PARAMS 12

typedef struct {
    Osc osc;
    TrapEnv amp_env;
    PitchEnv pitch_env;
    Param params[SUB808_PARAMS];
    float current_pitch_hz;
    int active;
    int note;
    char trigger_mode; /* 0=classic, 1=oneshot */
} Sub808;

void sub808_init(Sub808 *s);
void sub808_trigger(Sub808 *s, int note, float vel);
void sub808_release(Sub808 *s, int note);
void sub808_process(Sub808 *s, float *buf, int n);
