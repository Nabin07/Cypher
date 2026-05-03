#include "synth.h"
#include <string.h>
#include <math.h>

void polysynth_init(PolySynth *s) {
    memset(s, 0, sizeof(*s));
    for (int i = 0; i < MAX_POLY; i++) {
        osc_init(&s->voices[i].osc_a);
        osc_init(&s->voices[i].osc_b);
        adenv_init(&s->voices[i].amp_env);
        adenv_init(&s->voices[i].filter_env);
        biquad_init(&s->voices[i].filt);
    }
    memset(s->note_map, -1, sizeof(s->note_map));

    Param defaults[] = {
        {"wave_a",  "WAVE A",  0, 3, 0.0f, 0.0f, "",   CURVE_LINEAR, 4},
        {"wave_b",  "WAVE B",  0, 3, 0.0f, 0.0f, "",   CURVE_LINEAR, 4},
        {"osc_mix", "MIX",     0, 1, 0.0f, 0.0f, "",   CURVE_LINEAR, 0},
        {"detune",  "DETUNE",  -24, 24, 0.5f, 0.5f, "st",CURVE_LINEAR, 0},
        {"cutoff",  "CUTOFF",  50, 16000, 0.70f, 0.70f, "Hz", CURVE_EXP, 0},
        {"reso",    "RESO",    0.1f, 15.0f, 0.15f, 0.15f, "", CURVE_EXP, 0},
        {"fenv",    "F.ENV",   -1, 1, 0.50f, 0.50f, "", CURVE_LINEAR, 0},
        {"fmode",   "MODE",    0, 2, 0.0f, 0.0f, "",   CURVE_LINEAR, 3},
        {"attack",  "ATTACK",  0.001f, 2.0f, 0.05f, 0.05f, "s", CURVE_EXP, 0},
        {"decay",   "DECAY",   0.01f, 4.0f, 0.30f, 0.30f, "s", CURVE_EXP, 0},
        {"sustain", "SUSTAIN", 0, 1, 0.65f, 0.65f, "",  CURVE_LINEAR, 0},
        {"release", "RELEASE", 0.01f, 4.0f, 0.25f, 0.25f, "s", CURVE_EXP, 0},
        {"lfo_rate","LFO RATE",0.1f, 20.0f, 0.30f, 0.30f, "Hz", CURVE_EXP, 0},
        {"lfo_dep", "LFO DEP", 0, 1, 0.0f, 0.0f, "",   CURVE_LINEAR, 0},
        {"lfo_dst", "LFO DST", 0, 1, 0.0f, 0.0f, "",   CURVE_LINEAR, 2},
        {"noise",   "NOISE",   0, 1, 0.0f, 0.0f, "",   CURVE_LINEAR, 0},
    };
    memcpy(s->params, defaults, sizeof(defaults));
}

static void update_voice(PolySynth *s, MonoVoice *v) {
    v->osc_a.waveform = (int)roundf(param_mapped(&s->params[0]));
    v->osc_b.waveform = (int)roundf(param_mapped(&s->params[1]));
    v->osc_mix = param_mapped(&s->params[2]);
    v->detune = param_mapped(&s->params[3]);
    v->cutoff = param_mapped(&s->params[4]);
    v->reso = param_mapped(&s->params[5]);
    v->fenv_amt = param_mapped(&s->params[6]);
    v->filter_mode = (int)roundf(param_mapped(&s->params[7]));
    v->amp_env.attack = param_mapped(&s->params[8]);
    v->amp_env.decay = param_mapped(&s->params[9]);
    v->amp_env.sustain = param_mapped(&s->params[10]);
    v->amp_env.release = param_mapped(&s->params[11]);
    v->filter_env.attack = param_mapped(&s->params[8]) * 0.5f;
    v->filter_env.decay = param_mapped(&s->params[9]);
    v->filter_env.sustain = 0;
    v->filter_env.release = param_mapped(&s->params[11]);
    v->noise_amt = param_mapped(&s->params[15]);
}

static int alloc_voice(PolySynth *s) {
    /* Find idle voice */
    for (int i = 0; i < MAX_POLY; i++)
        if (!s->voices[i].active) return i;
    /* Steal oldest */
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

void polysynth_trigger(PolySynth *s, int note, float vel) {
    int vi;
    if (s->note_map[note] >= 0) {
        vi = s->note_map[note];
    } else {
        vi = alloc_voice(s);
    }
    MonoVoice *v = &s->voices[vi];
    update_voice(s, v);

    float freq = note_to_freq(note);
    osc_set_freq(&v->osc_a, freq);
    osc_set_freq(&v->osc_b, freq * powf(2.0f, v->detune / 12.0f));
    adenv_trigger(&v->amp_env);
    adenv_trigger(&v->filter_env);
    v->note = note;
    v->active = 1;
    s->note_map[note] = vi;

    /* Track allocation order */
    for (int i = 0; i < s->alloc_count; i++)
        if (s->alloc_order[i] == vi) {
            memmove(s->alloc_order + i, s->alloc_order + i + 1, (s->alloc_count - i - 1) * sizeof(int));
            s->alloc_count--;
            break;
        }
    s->alloc_order[s->alloc_count++] = vi;
}

void polysynth_release(PolySynth *s, int note) {
    int vi = s->note_map[note];
    if (vi >= 0) {
        adenv_release(&s->voices[vi].amp_env);
        adenv_release(&s->voices[vi].filter_env);
        s->note_map[note] = -1;
    }
}

void polysynth_release_all(PolySynth *s) {
    for (int i = 0; i < MAX_POLY; i++) {
        adenv_release(&s->voices[i].amp_env);
        adenv_release(&s->voices[i].filter_env);
    }
    memset(s->note_map, -1, sizeof(s->note_map));
}

void polysynth_process(PolySynth *s, float *buf, int n) {
    memset(buf, 0, n * sizeof(float));
    int active_count = 0;

    for (int vi = 0; vi < MAX_POLY; vi++) {
        MonoVoice *v = &s->voices[vi];
        if (!v->active) continue;
        update_voice(s, v);
        active_count++;

        for (int i = 0; i < n; i++) {
            float a = osc_tick(&v->osc_a);
            float b = osc_tick(&v->osc_b);
            float mix = a * (1.0f - v->osc_mix) + b * v->osc_mix;
            if (v->noise_amt > 0.01f) mix += noise_tick() * v->noise_amt;

            float fenv = adenv_tick(&v->filter_env);
            float fc = v->cutoff * (1.0f + v->fenv_amt * fenv * 4.0f);
            biquad_set(&v->filt, v->filter_mode, fc, v->reso);
            mix = biquad_tick(&v->filt, mix);

            float amp = adenv_tick(&v->amp_env);
            buf[i] += mix * amp;

            if (v->amp_env.stage == ENV_IDLE) { v->active = 0; break; }
        }
    }
    /* Scale for polyphony */
    if (active_count > 1) {
        float scale = 1.0f / sqrtf((float)active_count);
        for (int i = 0; i < n; i++) buf[i] *= scale;
    }
}

int polysynth_active(PolySynth *s) {
    for (int i = 0; i < MAX_POLY; i++)
        if (s->voices[i].active) return 1;
    return 0;
}
