CC = gcc
CFLAGS = -O2 -Wall -Wno-misleading-indentation -march=armv8-a
LDFLAGS = -lasound -lm -lpthread
SRCS = main.c sub808.c kick.c synth.c reverb.c sampler.c pitch_shift.c freeze.c gross_beat.c wav_loader.c pffft.c
TARGET = cypher

$(TARGET): $(SRCS) *.h
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)
