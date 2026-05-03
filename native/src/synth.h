#pragma once
#include "types.h"
#include "oscillator.h"
#include "envelope.h"
#include "filter.h"
#include "parameter.h"

#define SYNTH_PARAMS 16

/* Single mono voice */
typedef struct {
    Osc osc_a, osc_b;
    ADEnv amp_env, filter_env;
    Biquad filt;
    float osc_mix, detune, noise_amt;
    float cutoff, reso, fenv_amt;
    int filter_mode;
    int note;
    int active;
} MonoVoice;

/* Polyphonic wrapper */
typedef struct {
    MonoVoice voices[MAX_POLY];
    Param params[SYNTH_PARAMS];
    int note_map[128]; /* note -> voice index, -1 if none */
    int alloc_order[MAX_POLY];
    int alloc_count;
} PolySynth;

void polysynth_init(PolySynth *s);
void polysynth_trigger(PolySynth *s, int note, float vel);
void polysynth_release(PolySynth *s, int note);
void polysynth_release_all(PolySynth *s);
void polysynth_process(PolySynth *s, float *buf, int n);
int polysynth_active(PolySynth *s);
