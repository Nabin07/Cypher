"""Granular Freeze — sustain a sample indefinitely by looping a grain.

Engages automatically when a sample reaches its end point (or slice end
in CHOP mode). Captures a grain at a chosen POSITION within the sample
and loops it with an equal-power crossfade so the seam is inaudible.

MOTION modes govern how the capture position evolves over time:
    HOLD      — static, same grain forever
    DRIFT     — capture position advances through the sample
    OSCILLATE — capture position swings around the POSITION by ±DEPTH
    DECAY     — fade amplitude to 0 over RATE seconds

Crossfade: 25% of grain size on each side, equal-power (sin/cos).
"""

from __future__ import annotations

import math

import numpy as np

from ..core.types import AudioBuffer


MOTION_HOLD = 0
MOTION_DRIFT = 1
MOTION_OSCILLATE = 2
MOTION_DECAY = 3

MOTION_NAMES = ["HOLD", "DRIFT", "OSCILLATE", "DECAY"]


class FreezeState:
    """Per-voice freeze state. Created when a voice enters freeze mode."""

    __slots__ = (
        "source",        # full sample buffer (shared, read-only)
        "grain_size",    # in samples
        "xfade_len",
        "motion",
        "rate",          # motion-dependent meaning (Hz / frac / sec)
        "depth",         # for OSCILLATE (fraction of sample length)
        "position",      # current capture position (float, in samples)
        "position0",     # initial POSITION at engagement (for OSCILLATE center)
        "play_idx",      # where in the current grain we are reading (float)
        "_lfo_phase",    # for OSCILLATE
        "_decay_samples",
        "_decay_counter",
        "active",
    )

    def __init__(
        self,
        source: AudioBuffer,
        position_frac: float,
        grain_size: int,
        motion: int,
        rate: float,
        depth: float,
        sample_rate: int,
    ) -> None:
        self.source = source
        self.grain_size = max(64, int(grain_size))
        self.xfade_len = max(8, self.grain_size // 4)
        self.motion = motion
        self.rate = rate
        self.depth = max(0.0, min(0.5, depth))

        length = len(source)
        self.position = max(0.0, min(1.0, position_frac)) * max(1, length - self.grain_size)
        self.position0 = self.position
        self.play_idx = 0.0
        self._lfo_phase = 0.0
        if motion == MOTION_DECAY:
            self._decay_samples = max(1, int(rate * sample_rate))
        else:
            self._decay_samples = 0
        self._decay_counter = 0
        self.active = True


def _sample_from_grain(
    source: AudioBuffer,
    capture_start: float,
    play_idx: float,
    grain_size: int,
    xfade_len: int,
) -> float:
    """Return one output sample at `play_idx` from a seamlessly looping grain.

    The loop period is (grain_size - xfade_len). The last xfade_len samples
    of each grain overlap with the first xfade_len of the next, so the
    transition from sample `grain_size - xfade_len - 1` to `grain_size -
    xfade_len` (where the loop wraps) is sample-adjacent — no jump.
    """
    n = grain_size
    loop_len = max(1, n - xfade_len)
    src_len = len(source)
    base = int(capture_start)

    # Wrap play_idx into [0, loop_len)
    p = play_idx - loop_len * math.floor(play_idx / loop_len)
    idx = int(p)
    frac = p - idx

    def _read(src_i: int) -> float:
        i = base + src_i
        if 0 <= i < src_len:
            return float(source[i])
        return 0.0

    def _read_frac(src_i: int, f: float) -> float:
        a = _read(src_i)
        b = _read(src_i + 1)
        return a + f * (b - a)

    # During the first xfade_len samples of each loop iteration, the output
    # is a crossfade between the previous iteration's tail (source[loop_len..n])
    # and the current iteration's head (source[0..xfade_len]).
    if idx < xfade_len:
        tail_idx = loop_len + idx  # in [loop_len, n)
        t = (idx + frac) / xfade_len
        fade_in = math.sin(t * math.pi * 0.5)
        fade_out = math.cos(t * math.pi * 0.5)
        a = _read_frac(idx, frac)          # head of current iteration
        b = _read_frac(tail_idx, frac)     # tail of previous iteration
        return a * fade_in + b * fade_out

    return _read_frac(idx, frac)


class FreezeProcessor:
    """Produces samples from a FreezeState. Use: fs.process(num_frames)."""

    def __init__(self, state: FreezeState, sample_rate: int) -> None:
        self.state = state
        self.sample_rate = sample_rate

    def process(self, num_frames: int) -> AudioBuffer:
        st = self.state
        out = np.zeros(num_frames, dtype=np.float32)
        if not st.active:
            return out

        src = st.source
        grain = st.grain_size
        xfade = st.xfade_len
        length = len(src)
        max_start = max(1, length - grain)

        for i in range(num_frames):
            # Update capture position per motion
            if st.motion == MOTION_HOLD:
                cap = st.position
            elif st.motion == MOTION_DRIFT:
                # rate = fraction-of-sample per second
                st.position += st.rate * (length / self.sample_rate)
                st.position = min(max_start, st.position)
                cap = st.position
            elif st.motion == MOTION_OSCILLATE:
                # rate = Hz, depth = fraction of max_start
                st._lfo_phase += 2.0 * math.pi * st.rate / self.sample_rate
                if st._lfo_phase > 2.0 * math.pi:
                    st._lfo_phase -= 2.0 * math.pi
                excursion = math.sin(st._lfo_phase) * st.depth * max_start
                cap = max(0.0, min(float(max_start),
                                   st.position0 + excursion))
                st.position = cap
            else:  # DECAY
                cap = st.position

            # Output sample (loop length = grain - xfade)
            sample = _sample_from_grain(src, cap, st.play_idx, grain, xfade)
            out[i] = sample
            st.play_idx += 1.0
            loop_len = grain - xfade
            if st.play_idx >= loop_len:
                st.play_idx -= loop_len

            # DECAY envelope
            if st.motion == MOTION_DECAY:
                st._decay_counter += 1
                if st._decay_counter >= st._decay_samples:
                    st.active = False
                    out[i + 1:] = 0.0
                    # Apply final fade so the last sample tails out
                    fade = 1.0 - (st._decay_counter / st._decay_samples)
                    out[i] *= max(0.0, fade)
                    return out
                fade = 1.0 - (st._decay_counter / st._decay_samples)
                out[i] *= max(0.0, fade)

        return out
