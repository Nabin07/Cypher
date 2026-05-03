#pragma once

/* Phase vocoder pitch shift — offline, returns new buffer (caller frees).
   semitones: positive = up, negative = down.
   Returns 16-byte aligned float buffer via *out, length via *out_len. */
int pvoc_pitch_shift(const float *in, int in_len, float semitones,
                     float **out, int *out_len);

/* Time stretch — change length without changing pitch.
   ratio < 1 = shorter/faster, > 1 = longer/slower.
   Returns 16-byte aligned float buffer. */
int pvoc_time_stretch(const float *in, int in_len, float ratio,
                      float **out, int *out_len);
