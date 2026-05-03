#pragma once
#include "types.h"
#include "parameter.h"
#include "envelope.h"
#include "filter.h"
#include "freeze.h"
#include "gross_beat.h"
#include "wav_loader.h"

#define SAMPLER_PAD_COUNT 16
#define SAMPLER_VOICE_COUNT 8
#define SAMPLER_PAD_MIDI_START 36
#define SAMPLER_PAD_MIDI_END 51
#define SAMPLER_SLOT_PARAMS 14
#define SAMPLER_PITCH_CACHE_RANGE 25  /* -12 to +12 */

enum { SMODE_PAD, SMODE_CLASSIC, SMODE_CHOP };

/* Param indices per slot */
enum {
    SP_MODE, SP_PITCH, SP_REVERSE, SP_GAIN,
    SP_START, SP_END, SP_ATTACK, SP_DECAY,
    SP_SLICES, SP_FILTER,
    SP_FZ_POS, SP_FZ_GRAIN, SP_FZ_MOTION, SP_FZ_RATE
};

typedef struct {
    char name[64];
    char path[256];
    float *data;
    int length;
    int source_rate;
    int loaded;
    Param params[SAMPLER_SLOT_PARAMS];

    /* Pitch cache: index = semitones + 12 (so 0==-12st, 12==0st, 24==+12st) */
    float *pitch_cache[SAMPLER_PITCH_CACHE_RANGE];
    int pitch_cache_len[SAMPLER_PITCH_CACHE_RANGE];

    /* Slice points */
    int slice_starts[32];
    int slice_ends[32];
    int slice_count;

    int freeze_armed;
    float sample_bpm;
    int match_mode;
} SampleSlot;

typedef struct {
    float *buffer;
    int buf_len;
    SampleSlot *slot;
    float position;
    float rate;
    int active;
    int note;
    float velocity;
    int reversed;
    float start_pos, end_pos;
    float gain;
    ADEnv amp_env;
    Biquad filt;
    float cutoff_hz;
    FreezeProc freeze;
    int freeze_released;
} SamplerVoice;

typedef struct {
    SampleSlot slots[SAMPLER_PAD_COUNT];
    SamplerVoice voices[SAMPLER_VOICE_COUNT];
    int alloc_order[SAMPLER_VOICE_COUNT];
    int alloc_count;
    int note_map[128]; /* note -> voice index, -1 if none */
    int focused_slot;
    GrossBeat gross_beat;
    float project_bpm;
} SamplerEngine;

void sampler_init(SamplerEngine *s, int sample_rate);
void sampler_load_slot(SamplerEngine *s, int slot_idx, const char *path);
void sampler_clear_slot(SamplerEngine *s, int slot_idx);
void sampler_trigger(SamplerEngine *s, int note, float vel);
void sampler_release(SamplerEngine *s, int note);
void sampler_process(SamplerEngine *s, float *buf, int n);
int sampler_active(SamplerEngine *s);
