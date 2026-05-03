/*
 * WAV file loader — manual RIFF parser, no external deps.
 * Supports: PCM 8/16/24/32-bit, mono/stereo, any sample rate.
 * Output: mono float32, 16-byte aligned (for pffft compatibility).
 */
#include "wav_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read32le(const uint8_t *p) {
    return p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
}
static int read16le(const uint8_t *p) {
    return p[0] | (p[1] << 8);
}

int wav_load(const char *path, WavData *out) {
    memset(out, 0, sizeof(*out));

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    /* Read entire file */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize < 44) { fclose(f); return -2; }

    uint8_t *raw = malloc(fsize);
    if (!raw) { fclose(f); return -3; }
    fread(raw, 1, fsize, f);
    fclose(f);

    /* Verify RIFF header */
    if (memcmp(raw, "RIFF", 4) != 0 || memcmp(raw + 8, "WAVE", 4) != 0) {
        free(raw); return -4;
    }

    /* Find fmt and data chunks */
    int fmt_offset = -1, data_offset = -1, data_size = 0;
    int pos = 12;
    while (pos + 8 <= fsize) {
        int chunk_size = read32le(raw + pos + 4);
        if (memcmp(raw + pos, "fmt ", 4) == 0) {
            fmt_offset = pos + 8;
        } else if (memcmp(raw + pos, "data", 4) == 0) {
            data_offset = pos + 8;
            data_size = chunk_size;
        }
        pos += 8 + chunk_size;
        if (pos & 1) pos++; /* chunks are word-aligned */
    }

    if (fmt_offset < 0 || data_offset < 0) { free(raw); return -5; }

    /* Parse fmt chunk */
    int audio_format = read16le(raw + fmt_offset);
    int channels = read16le(raw + fmt_offset + 2);
    int sample_rate = read32le(raw + fmt_offset + 4);
    int bits_per_sample = read16le(raw + fmt_offset + 14);

    /* Only PCM (1) and IEEE float (3) */
    if (audio_format != 1 && audio_format != 3) { free(raw); return -6; }

    int bytes_per_sample = bits_per_sample / 8;
    int frame_size = bytes_per_sample * channels;
    int num_frames = data_size / frame_size;

    /* Allocate output — 16-byte aligned for pffft */
    float *mono;
    if (posix_memalign((void **)&mono, 16, num_frames * sizeof(float)) != 0) {
        free(raw); return -7;
    }

    uint8_t *src = raw + data_offset;

    for (int i = 0; i < num_frames; i++) {
        float sample = 0;
        for (int ch = 0; ch < channels; ch++) {
            uint8_t *p = src + i * frame_size + ch * bytes_per_sample;
            float s = 0;
            if (audio_format == 3 && bits_per_sample == 32) {
                /* IEEE float32 */
                memcpy(&s, p, 4);
            } else if (bits_per_sample == 16) {
                int16_t v = (int16_t)(p[0] | (p[1] << 8));
                s = v / 32768.0f;
            } else if (bits_per_sample == 24) {
                int32_t v = p[0] | (p[1] << 8) | (p[2] << 16);
                if (v & 0x800000) v |= 0xFF000000; /* sign extend */
                s = v / 8388608.0f;
            } else if (bits_per_sample == 32) {
                int32_t v = read32le(p);
                s = v / 2147483648.0f;
            } else if (bits_per_sample == 8) {
                s = (p[0] - 128) / 128.0f;
            }
            sample += s;
        }
        mono[i] = sample / channels; /* mix to mono */
    }

    free(raw);
    out->data = mono;
    out->length = num_frames;
    out->sample_rate = sample_rate;
    return 0;
}

void wav_free(WavData *w) {
    if (w->data) { free(w->data); w->data = NULL; }
    w->length = 0;
}
