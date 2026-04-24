"""BPM detection via spectral-flux onset envelope + autocorrelation.

Pure numpy/scipy — no librosa, no heavy deps. Portable to C++ (tiny FFT +
flat arrays). Returns (bpm, confidence) where confidence is the normalized
peak height of the autocorrelation, 0..1.

Algorithm:
    1. STFT magnitude, half-rectified diff across time → onset envelope.
    2. Normalize, then autocorrelate.
    3. Find the lag in [60, 200] BPM that has the highest correlation.
    4. Parabolic interpolation on the peak for sub-lag precision.
    5. Confidence = normalized peak / second-peak ratio, clipped.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft

from ..core.types import AudioBuffer


BPM_MIN = 60.0
BPM_MAX = 200.0


def detect_bpm(audio: AudioBuffer, sr: int) -> tuple[float, float]:
    """Return (bpm, confidence ∈ 0..1). Returns (0.0, 0.0) on failure."""
    if audio is None or len(audio) < sr:  # need at least 1s
        return 0.0, 0.0

    mono = audio.astype(np.float32, copy=False)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)

    # STFT — 2048/512 gives ~93 Hz onset-frame rate at 48k (adequate).
    hop = 512
    nperseg = 2048
    _, _, Z = stft(mono, fs=sr, nperseg=nperseg, noverlap=nperseg - hop,
                   padded=False, boundary=None)
    if Z.shape[1] < 4:
        return 0.0, 0.0

    mag = np.abs(Z).astype(np.float32)
    # Spectral flux: sum of positive magnitude increases per frame.
    diff = np.diff(mag, axis=1)
    flux = np.maximum(diff, 0.0).sum(axis=0)
    if flux.size < 8:
        return 0.0, 0.0

    # Normalize onset envelope.
    flux = flux - flux.mean()
    std = flux.std()
    if std < 1e-6:
        return 0.0, 0.0
    flux = flux / std

    onset_sr = sr / hop  # frames per second of the onset envelope

    # Lag range in onset-frames corresponding to [BPM_MIN, BPM_MAX].
    lag_max = int(onset_sr * 60.0 / BPM_MIN)
    lag_min = int(onset_sr * 60.0 / BPM_MAX)
    lag_min = max(2, lag_min)
    lag_max = min(len(flux) - 1, lag_max)
    if lag_max <= lag_min + 2:
        return 0.0, 0.0

    # Autocorrelation via FFT.
    n = 1 << int(np.ceil(np.log2(2 * len(flux))))
    F = np.fft.rfft(flux, n=n)
    ac = np.fft.irfft(F * np.conj(F), n=n)[: len(flux)]
    ac = ac / max(ac[0], 1e-9)  # normalize so ac[0] = 1

    window = ac[lag_min : lag_max + 1]
    peak_rel = int(np.argmax(window))
    peak_lag = lag_min + peak_rel
    peak_val = float(window[peak_rel])

    # Parabolic interpolation for sub-lag accuracy.
    if 0 < peak_rel < len(window) - 1:
        y0 = float(window[peak_rel - 1])
        y1 = peak_val
        y2 = float(window[peak_rel + 1])
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-9:
            delta = 0.5 * (y0 - y2) / denom
            peak_lag_f = peak_lag + float(delta)
        else:
            peak_lag_f = float(peak_lag)
    else:
        peak_lag_f = float(peak_lag)

    bpm = 60.0 * onset_sr / max(peak_lag_f, 1e-6)

    # Confidence: peak height vs. median of the search window.
    med = float(np.median(np.abs(window)))
    conf = peak_val / (peak_val + 4.0 * med) if med > 0 else peak_val
    conf = float(np.clip(conf, 0.0, 1.0))

    # Fold extreme results into the common musical range [70, 180] when
    # there is a doubled/halved harmonic that's more plausible.
    while bpm < 70.0:
        bpm *= 2.0
    while bpm > 180.0:
        bpm /= 2.0

    return float(bpm), conf
