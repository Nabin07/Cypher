#pragma once
#include "types.h"
#include "envelope.h"
#include "oscillator.h"
#include "filter.h"
#include "parameter.h"

#define KICK_PARAMS 8

typedef struct {
    Osc body_osc;
    ADEnv knock_env;
    TrapEnv body_env;
    PitchEnv pitch_env;
    Biquad hpf;
    Param params[KICK_PARAMS];
    int active, note;
    float linked_808_freq;
} Kick;

void kick_init(Kick *k);
void kick_trigger(Kick *k, int note, float vel);
void kick_release(Kick *k, int note);
void kick_process(Kick *k, float *buf, int n);
