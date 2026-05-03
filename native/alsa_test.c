/*
 * CYPHER native audio engine — ALSA stereo output.
 * Compile: gcc -O2 -o alsa_test alsa_test.c -lasound -lm
 * Run: ./alsa_test hw:0,0
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <alsa/asoundlib.h>

#define SAMPLE_RATE 48000
#define CHANNELS 2
#define BUFFER_FRAMES 1024
#define FREQ_HZ 440.0

int main(int argc, char *argv[]) {
    snd_pcm_t *pcm;
    snd_pcm_hw_params_t *params;
    int err;
    const char *device = "hw:0,0";

    if (argc > 1) device = argv[1];

    printf("CYPHER audio test — %s @ %dHz stereo\n", device, SAMPLE_RATE);

    if ((err = snd_pcm_open(&pcm, device, SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
        fprintf(stderr, "Cannot open audio: %s\n", snd_strerror(err));
        return 1;
    }

    snd_pcm_hw_params_alloca(&params);
    snd_pcm_hw_params_any(pcm, params);
    snd_pcm_hw_params_set_access(pcm, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(pcm, params, SND_PCM_FORMAT_S16_LE);

    unsigned int rate = SAMPLE_RATE;
    snd_pcm_hw_params_set_rate_near(pcm, params, &rate, 0);
    snd_pcm_hw_params_set_channels(pcm, params, CHANNELS);

    snd_pcm_uframes_t buffer_size = BUFFER_FRAMES * 4;
    snd_pcm_uframes_t period_size = BUFFER_FRAMES;
    snd_pcm_hw_params_set_buffer_size_near(pcm, params, &buffer_size);
    snd_pcm_hw_params_set_period_size_near(pcm, params, &period_size, 0);

    if ((err = snd_pcm_hw_params(pcm, params)) < 0) {
        fprintf(stderr, "Cannot set params: %s\n", snd_strerror(err));
        snd_pcm_close(pcm);
        return 1;
    }

    printf("Rate: %u, Buffer: %lu, Period: %lu, Channels: %d\n",
           rate, buffer_size, period_size, CHANNELS);
    printf("Playing 440Hz sine for 3 seconds...\n");

    /* Interleaved stereo buffer: L R L R L R ... */
    int16_t buf[BUFFER_FRAMES * CHANNELS];
    double phase = 0.0;
    double phase_inc = 2.0 * M_PI * FREQ_HZ / (double)rate;
    int total_frames = rate * 3;
    int written = 0;

    while (written < total_frames) {
        int frames = BUFFER_FRAMES;
        if (written + frames > total_frames)
            frames = total_frames - written;

        for (int i = 0; i < frames; i++) {
            int16_t sample = (int16_t)(sin(phase) * 12000.0);
            buf[i * 2]     = sample;  /* Left */
            buf[i * 2 + 1] = sample;  /* Right */
            phase += phase_inc;
            if (phase > 2.0 * M_PI) phase -= 2.0 * M_PI;
        }

        err = snd_pcm_writei(pcm, buf, frames);
        if (err == -EPIPE) {
            snd_pcm_prepare(pcm);
        } else if (err < 0) {
            fprintf(stderr, "Write error: %s\n", snd_strerror(err));
            break;
        }
        written += frames;
    }

    snd_pcm_drain(pcm);
    snd_pcm_close(pcm);
    printf("Done.\n");
    return 0;
}
