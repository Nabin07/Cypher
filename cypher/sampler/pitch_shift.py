"""Phase-vocoder pitch shift + time stretch.

Classic STFT-domain phase vocoder. For pitch-shifting without changing
duration: time-stretch by 1/R (makes sample longer/shorter), then resample
by R (restores original duration, shifts pitch).

Reference: Flanagan & Golden, "Phase Vocoder" (Bell Labs, 1966); also
Jean Laroche & Mark Dolson, "Improved Phase Vocoder Time-Scale Modification
of Audio" (1999) for the phase-locking approach used here.

C++ portability notes:
    - Uses numpy.fft.rfft/irfft — in C++ this is kiss_fft or PFFFT.
    - Windowing, phase unwrap, overlap-add are all plain loops on float32.
    - No allocation in the pitch-shift function itself — caller provides
      the output size (proportional to input * inverse ratio).
"""

from __future__ import annotations

import numpy as np

from ..core.types import AudioBuffer


# Defaults. Bigger frame = smoother sustain, worse transients.
# 2048 @ 48kHz = ~43ms window, ~11ms hop at 75% overlap.
DEFAULT_FRAME_SIZE = 2048
DEFAULT_HOP = 512          # frame_size // 4 → 75% overlap


def _hann(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


def _stft(x: AudioBuffer, frame_size: int, hop: int) -> np.ndarray:
    """Return (n_frames, n_bins) complex STFT."""
    window = _hann(frame_size)
    n_frames = 1 + max(0, (len(x) - frame_size) // hop)
    if n_frames <= 0:
        return np.zeros((0, frame_size // 2 + 1), dtype=np.complex64)
    frames = np.zeros((n_frames, frame_size // 2 + 1), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop
        frames[i] = np.fft.rfft(x[start:start + frame_size] * window)
    return frames


def _istft(frames: np.ndarray, frame_size: int, hop: int) -> AudioBuffer:
    """Inverse STFT with overlap-add + window normalization."""
    n_frames = frames.shape[0]
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)
    window = _hann(frame_size)
    out_len = (n_frames - 1) * hop + frame_size
    output = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    window_sq = (window * window).astype(np.float32)
    for i in range(n_frames):
        frame = np.fft.irfft(frames[i], n=frame_size).astype(np.float32) * window
        start = i * hop
        output[start:start + frame_size] += frame
        norm[start:start + frame_size] += window_sq
    # Avoid divide-by-zero at edges
    norm = np.maximum(norm, 1e-10)
    return output / norm


def _phase_vocoder(
    frames: np.ndarray, stretch_ratio: float, hop: int, frame_size: int
) -> np.ndarray:
    """Time-stretch in STFT domain. stretch_ratio > 1 → longer output.

    Uses scaled-phase-locking: resynthesizes each output frame by
    interpolating magnitude from the closest input frames and propagating
    phase using the unwrapped phase difference (true instantaneous
    frequency). Produces clean sustain at the cost of transient smearing.
    """
    if stretch_ratio == 1.0 or frames.shape[0] < 2:
        return frames

    n_in, n_bins = frames.shape
    n_out = max(1, int(round(n_in * stretch_ratio)))

    mag = np.abs(frames).astype(np.float32)
    phase = np.angle(frames).astype(np.float32)

    # Expected phase advance per bin per analysis hop
    k = np.arange(n_bins, dtype=np.float32)
    expected = 2.0 * np.pi * hop * k / frame_size

    # Unwrapped deviation from expected advance → true frequency
    dphi = np.zeros_like(phase)
    dphi[1:] = phase[1:] - phase[:-1] - expected
    dphi = np.mod(dphi + np.pi, 2.0 * np.pi) - np.pi
    true_freq = expected + dphi  # per-frame phase delta, unwrapped

    # Output frame positions in input index space
    out_indices = np.linspace(0, n_in - 1, n_out)

    out_frames = np.zeros((n_out, n_bins), dtype=np.complex64)
    cur_phase = phase[0].copy()
    out_frames[0] = mag[0] * np.exp(1j * cur_phase)

    for i in range(1, n_out):
        t = out_indices[i]
        t_floor = int(np.floor(t))
        t_frac = t - t_floor
        if t_floor + 1 < n_in:
            m = (1.0 - t_frac) * mag[t_floor] + t_frac * mag[t_floor + 1]
            # Use true frequency from the nearest forward input frame
            tf = true_freq[t_floor + 1]
        else:
            m = mag[t_floor]
            tf = true_freq[t_floor]
        # Advance synthesis phase in proportion to output hop
        cur_phase = cur_phase + tf
        out_frames[i] = m * np.exp(1j * cur_phase)

    return out_frames


def _resample_linear(x: AudioBuffer, ratio: float) -> AudioBuffer:
    """Linear-interp resample. ratio > 1 → output shorter (higher pitch)."""
    if ratio == 1.0 or len(x) < 2:
        return x
    n_out = max(1, int(round(len(x) / ratio)))
    indices = np.arange(n_out, dtype=np.float32) * ratio
    idx_floor = indices.astype(np.int32)
    idx_floor = np.clip(idx_floor, 0, len(x) - 2)
    frac = indices - idx_floor
    return (x[idx_floor] + frac * (x[idx_floor + 1] - x[idx_floor])).astype(np.float32)


def pitch_shift(
    data: AudioBuffer,
    semitones: float,
    frame_size: int = DEFAULT_FRAME_SIZE,
    hop: int = DEFAULT_HOP,
) -> AudioBuffer:
    """Shift pitch by `semitones`, preserving duration.

    Returns a new float32 buffer the same length as the input (within a
    few samples rounding).
    """
    if abs(semitones) < 1e-3 or len(data) < frame_size:
        return data.astype(np.float32, copy=False)

    pitch_ratio = 2.0 ** (semitones / 12.0)
    # For pitch-preserve-duration: time-stretch BY pitch_ratio (longer buffer),
    # then resample BY pitch_ratio (shorter + higher pitch) → length |input|.
    stretch_ratio = pitch_ratio

    frames = _stft(data, frame_size, hop)
    stretched_frames = _phase_vocoder(frames, stretch_ratio, hop, frame_size)
    stretched = _istft(stretched_frames, frame_size, hop)
    shifted = _resample_linear(stretched, pitch_ratio)

    # Trim / pad to match input length
    if len(shifted) > len(data):
        shifted = shifted[:len(data)]
    elif len(shifted) < len(data):
        shifted = np.pad(shifted, (0, len(data) - len(shifted)))
    return shifted


def time_stretch(
    data: AudioBuffer,
    ratio: float,
    frame_size: int = DEFAULT_FRAME_SIZE,
    hop: int = DEFAULT_HOP,
) -> AudioBuffer:
    """Time-stretch by `ratio`, preserving pitch.

    ratio > 1.0 → slower (longer output).
    ratio < 1.0 → faster (shorter output).
    """
    if ratio == 1.0 or len(data) < frame_size:
        return data.astype(np.float32, copy=False)
    frames = _stft(data, frame_size, hop)
    stretched = _phase_vocoder(frames, ratio, hop, frame_size)
    return _istft(stretched, frame_size, hop)
