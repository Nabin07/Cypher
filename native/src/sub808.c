#include "sub808.h"
#include <string.h>

void sub808_init(Sub808 *s) {
    memset(s, 0, sizeof(*s));
    osc_init(&s->osc);
    s->osc.waveform = WAVE_SINE;
    trapenv_init(&s->amp_env);
    pitchenv_init(&s->pitch_env);
    s->trigger_mode = 0;

    /* Params: PITCH, DECAY, SUSTAIN, DRIVE, SWEEP, SLIDE, ATTACK, HOLD, D1TIME, SATLEVEL, SATMODE, BODY */
    Param defaults[] = {
        {"pitch",   "PITCH",   30, 80, 0.30f, 0.30f, "Hz", CURVE_EXP, 0},
        {"decay",   "DECAY",   0.1f, 8.0f, 0.45f, 0.45f, "s", CURVE_EXP, 0},
        {"sustain", "SUSTAIN", 0, 1.0f, 0.60f, 0.60f, "", CURVE_LINEAR, 0},
        {"drive",   "DRIVE",   0, 1.0f, 0.20f, 0.20f, "", CURVE_LINEAR, 0},
        {"sweep",   "SWEEP",   50, 400, 0.50f, 0.50f, "Hz", CURVE_EXP, 0},
        {"slide",   "SLIDE",   0.01f, 0.3f, 0.40f, 0.40f, "s", CURVE_EXP, 0},
        {"attack",  "ATTACK",  0.001f, 0.05f, 0.20f, 0.20f, "s", CURVE_EXP, 0},
        {"hold",    "HOLD",    0.001f, 0.05f, 0.50f, 0.50f, "s", CURVE_EXP, 0},
        {"d1time",  "PUNCH",   0.01f, 0.2f, 0.40f, 0.40f, "s", CURVE_EXP, 0},
        {"satlevel","SAT LVL", 0, 1.0f, 0.0f, 0.0f, "", CURVE_LINEAR, 0},
        {"satmode", "SAT MODE",0, 4, 0.0f, 0.0f, "", CURVE_LINEAR, 5},
        {"body",    "BODY",    0, 1.0f, 0.70f, 0.70f, "", CURVE_LINEAR, 0},
    };
    memcpy(s->params, defaults, sizeof(defaults));
}

static void update_params(Sub808 *s) {
    s->amp_env.attack = param_mapped(&s->params[6]);
    s->amp_env.hold = param_mapped(&s->params[7]);
    s->amp_env.decay1 = param_mapped(&s->params[8]);
    s->amp_env.sustain_level = param_mapped(&s->params[2]);
    s->amp_env.decay2 = param_mapped(&s->params[1]);
    s->pitch_env.start_hz = param_mapped(&s->params[4]);
    s->pitch_env.slide_time = param_mapped(&s->params[5]);
}

void sub808_trigger(Sub808 *s, int note, float vel) {
    update_params(s);
    float freq = note_to_freq(note);
    s->current_pitch_hz = freq;
    s->note = note;
    s->active = 1;
    trapenv_trigger(&s->amp_env);
    pitchenv_trigger(&s->pitch_env, freq);
}

void sub808_release(Sub808 *s, int note) {
    if (note == s->note) {
        trapenv_release(&s->amp_env);
    }
}

void sub808_process(Sub808 *s, float *buf, int n) {
    if (!s->active) { memset(buf, 0, n * sizeof(float)); return; }
    update_params(s);
    float drive = param_mapped(&s->params[3]);

    for (int i = 0; i < n; i++) {
        float freq = pitchenv_tick(&s->pitch_env);
        float amp = trapenv_tick(&s->amp_env);
        osc_set_freq(&s->osc, freq);
        float sample = osc_tick(&s->osc) * amp;

        /* Soft saturation */
        if (drive > 0.01f) {
            sample *= (1.0f + drive * 3.0f);
            sample = tanhf(sample);
        }
        buf[i] = sample;

        if (s->amp_env.stage == TRAP_IDLE) { s->active = 0; break; }
    }
    s->current_pitch_hz = s->pitch_env.current_hz;
}
