/*
 * CYPHER audio test — plays a 440Hz sine via aplay pipe.
 * Statically compiled, zero library deps on target.
 * Usage: ./sine_test | aplay -f S16_LE -r 48000 -c 1 -D hw:1,0
 *   or:  ./sine_test > /tmp/test.raw && aplay -f S16_LE -r 48000 -c 1 -D hw:1,0 /tmp/test.raw
 */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>

#define SAMPLE_RATE 48000
#define DURATION_SEC 3
#define FREQ_HZ 440.0
#define AMPLITUDE 16000

int main(void) {
    double phase = 0.0;
    double phase_inc = 2.0 * M_PI * FREQ_HZ / (double)SAMPLE_RATE;
    int total = SAMPLE_RATE * DURATION_SEC;
    int16_t buf[1024];
    int written = 0;

    while (written < total) {
        int n = 1024;
        if (written + n > total) n = total - written;
        for (int i = 0; i < n; i++) {
            buf[i] = (int16_t)(sin(phase) * AMPLITUDE);
            phase += phase_inc;
            if (phase > 2.0 * M_PI) phase -= 2.0 * M_PI;
        }
        write(STDOUT_FILENO, buf, n * sizeof(int16_t));
        written += n;
    }
    return 0;
}
