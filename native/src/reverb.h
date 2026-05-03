#pragma once
#include "types.h"

#define REV_MAX_DELAY 8192

typedef struct {
    float buf[REV_MAX_DELAY];
    int mask, wpos;
} DelayLine;

typedef struct {
    float mix, decay, damping, predelay_ms, mod_depth, mod_rate;
    float low_cut_hz;
    int mode; /* 0=room,1=chamber,2=hall,3=plate */

    DelayLine predelay;
    /* Input diffusion: 4 allpasses */
    DelayLine iap[4]; int iap_len[4]; float iap_g[4];
    /* Tank: 2 loops */
    DelayLine tap_dl[2]; int tap_len[2]; float tap_g[2]; /* mod allpass */
    DelayLine td[2]; int td_len[2];   /* tank delay */
    DelayLine td2[2]; int td2_len[2]; /* tank delay 2 */
    float damp_state[2];
    float tank_state[2];
    float lfo_phase;
    float hpf_prev_in, hpf_prev_out;
} Reverb;

void reverb_init(Reverb *r);
void reverb_clear(Reverb *r);
void reverb_process(Reverb *r, const float *in, float *out, int n);
void reverb_set_mode(Reverb *r, int mode);
