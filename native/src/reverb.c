#include "reverb.h"
#include <string.h>
#include <math.h>

/* Dattorro reference lengths at 29761Hz */
#define REF_SR 29761
static const int INPUT_DIFF_LEN[] = {142, 107, 379, 277};
static const float INPUT_DIFF_G[] = {0.75f, 0.75f, 0.625f, 0.625f};
static const int TANK_AP_LEN[] = {672, 908};
static const int TANK_DL_LEN[] = {4453, 4217};
static const int TANK_DL2_LEN[] = {3720, 3163};
static const int TAPS_L[] = {266, 2974, 1913, 1996, 1990, 187, 1066};
static const int TAPS_R[] = {353, 3627, 1228, 2673, 2111, 335, 121};

/* Mode presets: size_mult, mod_depth, mod_rate */
static const float MODES[][3] = {
    {0.30f, 0.3f, 0.5f}, /* ROOM */
    {0.50f, 0.4f, 0.7f}, /* CHAMBER */
    {1.00f, 0.6f, 0.9f}, /* HALL */
    {0.75f, 0.8f, 1.2f}, /* PLATE */
};

static int scale_len(int ref, int sr) { int v = (ref * sr) / REF_SR; return v < 1 ? 1 : v; }
static int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

static void dl_init(DelayLine *d, int max_len) {
    int sz = next_pow2(max_len + 16);
    if (sz > REV_MAX_DELAY) sz = REV_MAX_DELAY;
    d->mask = sz - 1; d->wpos = 0;
    memset(d->buf, 0, sizeof(d->buf));
}

static inline void dl_write(DelayLine *d, float s) {
    d->buf[d->wpos & d->mask] = s; d->wpos++;
}

static inline float dl_read(DelayLine *d, int delay) {
    return d->buf[(d->wpos - delay) & d->mask];
}

static inline float dl_read_frac(DelayLine *d, float delay) {
    int di = (int)delay; float frac = delay - di;
    float a = d->buf[(d->wpos - di) & d->mask];
    float b = d->buf[(d->wpos - di - 1) & d->mask];
    return a + frac * (b - a);
}

void reverb_init(Reverb *r) {
    memset(r, 0, sizeof(*r));
    r->mix = 0.3f; r->decay = 0.7f; r->damping = 0.3f;
    r->predelay_ms = 20.0f; r->mod_depth = 0.5f; r->mod_rate = 0.8f;
    r->low_cut_hz = 80.0f; r->mode = 2;

    dl_init(&r->predelay, (int)(0.15f * SR));
    for (int i = 0; i < 4; i++) {
        r->iap_len[i] = scale_len(INPUT_DIFF_LEN[i], SR);
        r->iap_g[i] = INPUT_DIFF_G[i];
        dl_init(&r->iap[i], r->iap_len[i]);
    }
    for (int i = 0; i < 2; i++) {
        r->tap_len[i] = scale_len(TANK_AP_LEN[i], SR);
        r->tap_g[i] = 0.7f;
        dl_init(&r->tap_dl[i], r->tap_len[i] + 64);
        r->td_len[i] = scale_len(TANK_DL_LEN[i], SR);
        dl_init(&r->td[i], r->td_len[i]);
        r->td2_len[i] = scale_len(TANK_DL2_LEN[i], SR);
        dl_init(&r->td2[i], r->td2_len[i]);
    }
}

void reverb_clear(Reverb *r) {
    memset(r->predelay.buf, 0, sizeof(r->predelay.buf));
    for (int i = 0; i < 4; i++) memset(r->iap[i].buf, 0, sizeof(r->iap[i].buf));
    for (int i = 0; i < 2; i++) {
        memset(r->tap_dl[i].buf, 0, sizeof(r->tap_dl[i].buf));
        memset(r->td[i].buf, 0, sizeof(r->td[i].buf));
        memset(r->td2[i].buf, 0, sizeof(r->td2[i].buf));
        r->damp_state[i] = 0; r->tank_state[i] = 0;
    }
    r->lfo_phase = 0; r->hpf_prev_in = 0; r->hpf_prev_out = 0;
}

void reverb_set_mode(Reverb *r, int mode) {
    r->mode = clampf(mode, 0, 3);
    r->mod_depth = MODES[r->mode][1];
    r->mod_rate = MODES[r->mode][2];
}

void reverb_process(Reverb *r, const float *in, float *out, int n) {
    int pd_samps = (int)(r->predelay_ms / 1000.0f * SR);
    if (pd_samps < 1) pd_samps = 1;
    float decay = clampf(r->decay, 0, 0.999f);
    float damp = clampf(r->damping, 0, 0.99f);
    float damp_inv = 1.0f - damp;
    float md = r->mod_depth * 12.0f;
    float lfo_inc = r->mod_rate * TWO_PI / SR;
    float wet = r->mix, dry = 1.0f - r->mix;
    float size_mult = MODES[r->mode][0];

    int td_s[2], td2_s[2];
    int tl[7], tr[7];
    for (int i = 0; i < 2; i++) {
        td_s[i] = (int)(r->td_len[i] * size_mult); if (td_s[i] < 1) td_s[i] = 1;
        td2_s[i] = (int)(r->td2_len[i] * size_mult); if (td2_s[i] < 1) td2_s[i] = 1;
    }
    for (int i = 0; i < 7; i++) {
        tl[i] = (int)(scale_len(TAPS_L[i], SR) * size_mult); if (tl[i] < 1) tl[i] = 1;
        tr[i] = (int)(scale_len(TAPS_R[i], SR) * size_mult); if (tr[i] < 1) tr[i] = 1;
    }

    float lc = clampf(r->low_cut_hz, 20, 500);
    float hpf_rc = 1.0f / (TWO_PI * lc);
    float hpf_dt = 1.0f / SR;
    float hpf_a = hpf_rc / (hpf_rc + hpf_dt);

    for (int i = 0; i < n; i++) {
        float x = in[i], dry_s = x;

        /* Pre-delay */
        dl_write(&r->predelay, x);
        x = dl_read(&r->predelay, pd_samps);

        /* Input diffusion */
        for (int j = 0; j < 4; j++) {
            float d = dl_read(&r->iap[j], r->iap_len[j]);
            float v = x + r->iap_g[j] * d;
            dl_write(&r->iap[j], v);
            x = d - r->iap_g[j] * v;
        }

        float lfo1 = sinf(r->lfo_phase) * md;
        float lfo2 = cosf(r->lfo_phase) * md;
        r->lfo_phase += lfo_inc;
        if (r->lfo_phase > TWO_PI) r->lfo_phase -= TWO_PI;

        /* Tank loop 0 */
        float tin = x + decay * r->tank_state[1];
        float rd = r->tap_len[0] + lfo1; if (rd < 1) rd = 1;
        float delayed = dl_read_frac(&r->tap_dl[0], rd);
        float v = tin + r->tap_g[0] * delayed;
        dl_write(&r->tap_dl[0], v);
        float ap_out = delayed - r->tap_g[0] * v;
        dl_write(&r->td[0], ap_out);
        float d0 = dl_read(&r->td[0], td_s[0]);
        r->damp_state[0] = damp_inv * d0 + damp * r->damp_state[0];
        d0 = r->damp_state[0] * decay;
        dl_write(&r->td2[0], d0);
        r->tank_state[0] = dl_read(&r->td2[0], td2_s[0]);

        /* Tank loop 1 */
        tin = x + decay * r->tank_state[0];
        rd = r->tap_len[1] + lfo2; if (rd < 1) rd = 1;
        delayed = dl_read_frac(&r->tap_dl[1], rd);
        v = tin + r->tap_g[1] * delayed;
        dl_write(&r->tap_dl[1], v);
        ap_out = delayed - r->tap_g[1] * v;
        dl_write(&r->td[1], ap_out);
        float d1 = dl_read(&r->td[1], td_s[1]);
        r->damp_state[1] = damp_inv * d1 + damp * r->damp_state[1];
        d1 = r->damp_state[1] * decay;
        dl_write(&r->td2[1], d1);
        r->tank_state[1] = dl_read(&r->td2[1], td2_s[1]);

        /* Output taps */
        float wl = dl_read(&r->td[0], tl[0]) + dl_read(&r->td[0], tl[1])
                  - dl_read(&r->td2[0], tl[2]) + dl_read(&r->td2[0], tl[3])
                  - dl_read(&r->td[1], tl[4]) - dl_read(&r->td2[1], tl[5])
                  - dl_read(&r->td2[1], tl[6]);
        float wr = dl_read(&r->td[1], tr[0]) + dl_read(&r->td[1], tr[1])
                  - dl_read(&r->td2[1], tr[2]) + dl_read(&r->td2[1], tr[3])
                  - dl_read(&r->td[0], tr[4]) - dl_read(&r->td2[0], tr[5])
                  - dl_read(&r->td2[0], tr[6]);

        float wet_s = (wl + wr) * 0.25f;
        float hpf_out = hpf_a * (r->hpf_prev_out + wet_s - r->hpf_prev_in);
        r->hpf_prev_in = wet_s; r->hpf_prev_out = hpf_out;

        out[i] = dry_s * dry + hpf_out * wet;
    }
}
