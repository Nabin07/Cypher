CC = gcc
CFLAGS = -O2 -Wall -march=armv8-a
LDFLAGS = -lasound -lm -lpthread
SRCS = main.c sub808.c kick.c synth.c reverb.c
TARGET = cypher

$(TARGET): $(SRCS) *.h
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)
