/*
 * CYPHER audio test — raw ALSA via ioctl, zero dependencies.
 * Compile on board: gcc -O2 -o audio_test audio_test.c -lm
 * Or cross-compile: aarch64-linux-gnu-gcc -O2 -static -o audio_test audio_test.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/soundcard.h>

#define SAMPLE_RATE 48000
#define CHANNELS 1
#define BITS 16
#define DURATION_SEC 3
#define FREQ_HZ 440.0
#define BUFFER_FRAMES 1024

int main(int argc, char *argv[]) {
    const char *dev = "/dev/dsp";
    if (argc > 1) dev = argv[1];

    printf("CYPHER audio test — OSS compat on %s\n", dev);

    int fd = open(dev, O_WRONLY);
    if (fd < 0) {
        /* Try ALSA PCM device directly */
        dev = "/dev/snd/pcmC1D0p"; /* card 1 (rk809 codec), device 0, playback */
        fd = open(dev, O_WRONLY);
        if (fd < 0) {
            perror("Cannot open audio device");
            printf("Available devices:\n");
            system("ls -la /dev/snd/ 2>/dev/null");
            printf("\nTry: aplay -l\n");
            return 1;
        }
    }

    /* Set format via OSS ioctls */
    int fmt = AFMT_S16_LE;
    ioctl(fd, SNDCTL_DSP_SETFMT, &fmt);
    int ch = CHANNELS;
    ioctl(fd, SNDCTL_DSP_CHANNELS, &ch);
    int rate = SAMPLE_RATE;
    ioctl(fd, SNDCTL_DSP_SPEED, &rate);

    printf("Rate: %d, Channels: %d, Format: S16LE\n", rate, ch);
    printf("Playing %gHz sine for %d seconds...\n", FREQ_HZ, DURATION_SEC);

    int16_t buf[BUFFER_FRAMES];
    double phase = 0.0;
    double phase_inc = 2.0 * M_PI * FREQ_HZ / (double)rate;
    int total = rate * DURATION_SEC;
    int written = 0;

    while (written < total) {
        int n = BUFFER_FRAMES;
        if (written + n > total) n = total - written;
        for (int i = 0; i < n; i++) {
            buf[i] = (int16_t)(sin(phase) * 16000.0);
            phase += phase_inc;
            if (phase > 2.0 * M_PI) phase -= 2.0 * M_PI;
        }
        write(fd, buf, n * sizeof(int16_t));
        written += n;
    }

    close(fd);
    printf("Done.\n");
    return 0;
}
