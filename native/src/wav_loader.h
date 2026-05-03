#pragma once
#include <stdint.h>

typedef struct {
    float *data;       /* mono float32 samples, 16-byte aligned */
    int length;        /* number of samples */
    int sample_rate;
} WavData;

/* Load a WAV file, convert to mono float32. Returns 0 on success. */
int wav_load(const char *path, WavData *out);

/* Free loaded WAV data */
void wav_free(WavData *w);
