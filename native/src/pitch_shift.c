/*
 * Phase vocoder pitch shift + time stretch.
 * STFT → phase manipulation → ISTFT with overlap-add.
 * Uses pffft for FFT (NEON on ARM64).
 */
#include "pitch_shift.h"
#include "pffft.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PVOC_FFT_SIZE 2048
#define PVOC_HOP 512
#define PVOC_OVERLAP (PVOC_FFT_SIZE / PVOC_HOP) /* 4 */

static float *aligned_alloc_f(int n) {
    float *p;
    if (posix_memalign((void **)&p, 16, n * sizeof(float)) != 0) return NULL;
    memset(p, 0, n * sizeof(float));
    return p;
}

/* Hann window */
static void make_hann(float *w, int n) {
    for (int i = 0; i < n; i++)
        w[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / n));
}

/*
 * Core STFT time-stretch: reads input at analysis_hop, writes output at synthesis_hop.
 * ratio = synthesis_hop / analysis_hop. ratio<1 = compress (shorter), >1 = expand.
 */
static int stft_stretch(const float *in, int in_len, float ratio,
                        float **out, int *out_len) {
    int fft_size = PVOC_FFT_SIZE;
    int analysis_hop = PVOC_HOP;
    int synthesis_hop = (int)(analysis_hop * ratio + 0.5f);
    if (synthesis_hop < 1) synthesis_hop = 1;

    int num_frames = (in_len - fft_size) / analysis_hop + 1;
    if (num_frames < 1) num_frames = 1;
    int out_length = (num_frames - 1) * synthesis_hop + fft_size;

    PFFFT_Setup *fft = pffft_new_setup(fft_size, PFFFT_REAL);
    if (!fft) return -1;

    float *window = aligned_alloc_f(fft_size);
    float *fft_in = aligned_alloc_f(fft_size);
    float *fft_out = aligned_alloc_f(fft_size);
    float *ifft_out = aligned_alloc_f(fft_size);
    float *work = aligned_alloc_f(fft_size);
    float *output = aligned_alloc_f(out_length);
    float *phase_accum = aligned_alloc_f(fft_size / 2 + 1);
    float *prev_phase = aligned_alloc_f(fft_size / 2 + 1);

    if (!window || !fft_in || !fft_out || !ifft_out || !work || !output ||
        !phase_accum || !prev_phase) {
        pffft_destroy_setup(fft);
        free(window); free(fft_in); free(fft_out); free(ifft_out);
        free(work); free(output); free(phase_accum); free(prev_phase);
        return -2;
    }

    make_hann(window, fft_size);
    int half = fft_size / 2;
    float freq_per_bin = 2.0f * M_PI / fft_size;
    float expect = freq_per_bin * analysis_hop;

    for (int frame = 0; frame < num_frames; frame++) {
        int in_pos = frame * analysis_hop;
        int out_pos = frame * synthesis_hop;

        /* Window input */
        for (int i = 0; i < fft_size; i++) {
            int idx = in_pos + i;
            fft_in[i] = (idx < in_len ? in[idx] : 0) * window[i];
        }

        /* Forward FFT (ordered output: DC, R1, I1, R2, I2, ..., Nyquist) */
        pffft_transform_ordered(fft, fft_in, fft_out, work, PFFFT_FORWARD);

        /* Phase vocoder: accumulate phase differences */
        /* pffft ordered real format: [DC, R1, I1, R2, I2, ..., Rn/2-1, In/2-1, Nyquist] */
        for (int k = 0; k <= half; k++) {
            float re, im;
            if (k == 0) { re = fft_out[0]; im = 0; }
            else if (k == half) { re = fft_out[1]; im = 0; }
            else { re = fft_out[2 * k]; im = fft_out[2 * k + 1]; }

            float mag = sqrtf(re * re + im * im);
            float phase = atan2f(im, re);

            /* Phase difference from previous frame */
            float dp = phase - prev_phase[k];
            prev_phase[k] = phase;

            /* Remove expected phase advance, wrap to [-pi, pi] */
            dp -= k * expect;
            dp = fmodf(dp + M_PI, 2.0f * M_PI);
            if (dp < 0) dp += 2.0f * M_PI;
            dp -= M_PI;

            /* True frequency deviation + expected */
            float true_freq = k * freq_per_bin + dp / analysis_hop;

            /* Accumulate phase for synthesis */
            phase_accum[k] += true_freq * synthesis_hop;

            float new_re = mag * cosf(phase_accum[k]);
            float new_im = mag * sinf(phase_accum[k]);

            if (k == 0) { fft_out[0] = new_re; }
            else if (k == half) { fft_out[1] = new_re; }
            else { fft_out[2 * k] = new_re; fft_out[2 * k + 1] = new_im; }
        }

        /* Inverse FFT */
        pffft_transform_ordered(fft, fft_out, ifft_out, work, PFFFT_BACKWARD);

        /* Scale by 1/N (pffft doesn't normalize) and window, then overlap-add */
        float scale = 1.0f / fft_size;
        for (int i = 0; i < fft_size; i++) {
            int op = out_pos + i;
            if (op < out_length)
                output[op] += ifft_out[i] * scale * window[i];
        }
    }

    /* Normalize overlap-add (sum of squared windows) */
    float win_sum = 0;
    for (int i = 0; i < fft_size; i++) win_sum += window[i] * window[i];
    float norm = win_sum / synthesis_hop;
    if (norm > 0.001f) {
        for (int i = 0; i < out_length; i++) output[i] /= norm;
    }

    pffft_destroy_setup(fft);
    free(window); free(fft_in); free(fft_out); free(ifft_out);
    free(work); free(phase_accum); free(prev_phase);

    *out = output;
    *out_len = out_length;
    return 0;
}

int pvoc_time_stretch(const float *in, int in_len, float ratio,
                      float **out, int *out_len) {
    if (fabsf(ratio - 1.0f) < 0.001f) {
        /* No stretch needed — copy */
        float *buf = aligned_alloc_f(in_len);
        if (!buf) return -1;
        memcpy(buf, in, in_len * sizeof(float));
        *out = buf;
        *out_len = in_len;
        return 0;
    }
    return stft_stretch(in, in_len, ratio, out, out_len);
}

int pvoc_pitch_shift(const float *in, int in_len, float semitones,
                     float **out, int *out_len) {
    if (fabsf(semitones) < 0.01f) {
        float *buf = aligned_alloc_f(in_len);
        if (!buf) return -1;
        memcpy(buf, in, in_len * sizeof(float));
        *out = buf;
        *out_len = in_len;
        return 0;
    }

    /* Pitch shift = time stretch by inverse ratio, then resample back */
    float ratio = powf(2.0f, semitones / 12.0f);

    /* Step 1: time stretch to make it longer/shorter */
    float *stretched;
    int stretched_len;
    int err = stft_stretch(in, in_len, ratio, &stretched, &stretched_len);
    if (err) return err;

    /* Step 2: resample back to original length (linear interpolation) */
    float *result = aligned_alloc_f(in_len);
    if (!result) { free(stretched); return -1; }

    float step = (float)stretched_len / (float)in_len;
    for (int i = 0; i < in_len; i++) {
        float pos = i * step;
        int idx = (int)pos;
        float frac = pos - idx;
        if (idx >= stretched_len - 1) {
            result[i] = stretched[stretched_len - 1];
        } else {
            result[i] = stretched[idx] + frac * (stretched[idx + 1] - stretched[idx]);
        }
    }

    free(stretched);
    *out = result;
    *out_len = in_len;
    return 0;
}
