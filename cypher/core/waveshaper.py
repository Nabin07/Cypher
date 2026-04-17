"""Waveshaping / distortion for the CYPHER engine.

Multiple distortion characters:
- Soft clip (tanh): warm analog saturation
- Hard clip: aggressive digital distortion
- Tape saturation: even harmonics + soft compression (the Zay 808 sound)
- Bitcrush: lo-fi sample rate / bit depth reduction
"""

from __future__ import annotations

import numpy as np

from .types import AudioBuffer


def soft_clip(signal: AudioBuffer, drive: float) -> AudioBuffer:
    """Tanh soft clipping. Warm, analog-style saturation."""
    return np.tanh(signal * drive).astype(np.float32)


def hard_clip(signal: AudioBuffer, drive: float, threshold: float = 0.8) -> AudioBuffer:
    """Hard clipping. Aggressive, adds harsh harmonics."""
    driven = signal * drive
    return np.clip(driven, -threshold, threshold).astype(np.float32)


def tape_saturate(signal: AudioBuffer, drive: float) -> AudioBuffer:
    """Tape saturation — even harmonics + soft compression.

    Models the asymmetric saturation of magnetic tape:
    - Adds primarily even harmonics (warmer than odd-harmonic distortion)
    - Soft knee compression at peaks
    - Slight bass bump from tape head proximity effect

    This is the sound of the Zay 808 — warm, fat, musical distortion
    that makes the 808 feel bigger without getting harsh.
    """
    # Asymmetric soft clip — positive peaks saturate differently than negative
    # This produces even harmonics (2nd, 4th, 6th) which sound warm
    driven = signal * drive
    pos = np.tanh(driven * 0.9)
    neg = np.tanh(driven * 1.1)
    asym = np.where(driven >= 0, pos, neg)

    # Soft knee compression — tames peaks gently
    # Mix between clean and saturated based on level
    level = np.abs(driven)
    mix = np.tanh(level * 2.0)  # Higher levels get more saturation
    result = signal * (1.0 - mix) + asym * mix

    # Slight bass warmth — tape proximity effect
    # Simple one-pole lowpass mixed in at low level
    warmth = np.zeros_like(result)
    prev = 0.0
    alpha = 0.995  # Very low cutoff
    for i in range(len(result)):
        prev = alpha * prev + (1.0 - alpha) * result[i]
        warmth[i] = prev
    result = result + warmth * 0.08 * min(drive / 5.0, 1.0)

    return result.astype(np.float32)


def bitcrush(signal: AudioBuffer, bit_depth: float, downsample: int = 1) -> AudioBuffer:
    """Bit depth reduction + optional sample rate reduction."""
    levels = 2.0 ** bit_depth
    crushed = np.round(signal * levels) / levels

    if downsample > 1:
        held = np.copy(crushed)
        for i in range(len(held)):
            if i % downsample != 0:
                held[i] = held[i - (i % downsample)]
        crushed = held

    return crushed.astype(np.float32)


def punch_drive(signal: AudioBuffer, drive: float) -> AudioBuffer:
    """Clean transient punch — gain + symmetric soft-knee compression.

    Adds loudness and peak control without harmonic coloring.
    No asymmetry, no warmth filter, no even-harmonic generation.
    Leave that to saturation.
    """
    driven = signal * drive

    # Symmetric soft-knee limiter — tames peaks cleanly
    level = np.abs(driven)
    # Gentle knee: only compress where signal exceeds ~0.5
    knee = np.clip((level - 0.5) * 2.0, 0.0, 1.0)
    # Blend: clean below knee, soft-clipped above
    clipped = np.tanh(driven)  # symmetric tanh — odd harmonics only, minimal color
    result = driven * (1.0 - knee) + clipped * knee

    # Gain compensation — keep perceived level consistent
    peak = np.max(np.abs(result))
    if peak > 0.001:
        result = result * min(1.0, 0.95 / peak)

    return result.astype(np.float32)


def apply_drive(signal: AudioBuffer, drive_amount: float, character: str = "soft") -> AudioBuffer:
    """Apply drive/distortion with selectable character.

    Characters:
        "soft" — tanh saturation, clean warmth
        "tape" — tape saturation, even harmonics, fat and musical
        "punch" — clean transient punch, no harmonic coloring
        "hard" — hard clipping, aggressive
        "crush" — bitcrush, lo-fi destruction
    """
    if drive_amount <= 0.001:
        return signal

    if character == "tape":
        drive = 1.0 + drive_amount * 8.0  # Tape doesn't need as much gain
        return tape_saturate(signal, drive)
    elif character == "punch":
        drive = 1.0 + drive_amount * 10.0
        return punch_drive(signal, drive)
    elif character == "hard":
        drive = 1.0 + drive_amount * 15.0
        return hard_clip(signal, drive)
    elif character == "crush":
        bit_depth = 16.0 - drive_amount * 12.0
        downsample = max(1, int(drive_amount * 8))
        return bitcrush(signal, bit_depth, downsample)
    else:
        drive = 1.0 + drive_amount * 15.0
        return soft_clip(signal, drive)
