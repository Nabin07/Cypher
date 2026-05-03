#include "kick.h"
#include <string.h>

void kick_init(Kick *k) {
    memset(k, 0, sizeof(*k));
    osc_init(&k->body_osc);
    k->body_osc.waveform = WAVE_SINE;
    adenv_init(&k->knock_env);
    trapenv_init(&k->body_env);
    pitchenv_init(&k->pitch_env);
    biquad_init(&k->hpf);

    Param defaults[] = {
        {"pitch",  "PITCH",  30, 120, 0.35f, 0.35f, "Hz", CURVE_EXP, 0},
        {"decay",  "DECAY",  0.05f, 2.0f, 0.35f, 0.35f, "s", CURVE_EXP, 0},
        {"tone",   "TONE",   200, 8000, 0.50f, 0.50f, "Hz", CURVE_EXP, 0},
        {"drive",  "DRIVE",  0, 1.0f, 0.15f, 0.15f, "", CURVE_LINEAR, 0},
        {"knock",  "KNOCK",  0, 1.0f, 0.50f, 0.50f, "", CURVE_LINEAR, 0},
        {"sweep",  "SWEEP",  80, 500, 0.40f, 0.40f, "Hz", CURVE_EXP, 0},
        {"slide",  "SLIDE",  0.005f, 0.1f, 0.40f, 0.40f, "s", CURVE_EXP, 0},
        {"body",   "BODY",   0.05f, 1.0f, 0.50f, 0.50f, "s", CURVE_EXP, 0},
    };
    memcpy(k->params, defaults, sizeof(defaults));
}

void kick_trigger(Kick *k, int note, float vel) {
    float freq = param_mapped(&k->params[0]);
    k->note = note;
    k->active = 1;

    k->body_env.decay2 = param_mapped(&k->params[7]);
    trapenv_trigger(&k->body_env);

    k->knock_env.attack = 0.001f;
    k->knock_env.decay = 0.02f;
    k->knock_env.sustain = 0;
    adenv_trigger(&k->knock_env);

    k->pitch_env.start_hz = param_mapped(&k->params[5]);
    k->pitch_env.slide_time = param_mapped(&k->params[6]);
    pitchenv_trigger(&k->pitch_env, freq);
}

void kick_release(Kick *k, int note) {
    if (note == k->note) trapenv_release(&k->body_env);
}

void kick_process(Kick *k, float *buf, int n) {
    if (!k->active) { memset(buf, 0, n * sizeof(float)); return; }
    float drive = param_mapped(&k->params[3]);
    float knock_amt = param_mapped(&k->params[4]);
    float tone_freq = param_mapped(&k->params[2]);
    biquad_set(&k->hpf, FILT_LP, tone_freq, 0.7f);

    for (int i = 0; i < n; i++) {
        float freq = pitchenv_tick(&k->pitch_env);
        float body_amp = trapenv_tick(&k->body_env);
        float knock_amp = adenv_tick(&k->knock_env);

        osc_set_freq(&k->body_osc, freq);
        float body = osc_tick(&k->body_osc) * body_amp;
        float knock = noise_tick() * knock_amp * knock_amt;

        float mix = body + knock;
        mix = biquad_tick(&k->hpf, mix);
        if (drive > 0.01f) { mix *= (1.0f + drive * 3.0f); mix = tanhf(mix); }
        buf[i] = mix;

        if (k->body_env.stage == TRAP_IDLE && k->knock_env.stage == ENV_IDLE) {
            k->active = 0; break;
        }
    }
}
