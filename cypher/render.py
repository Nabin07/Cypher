"""WAV rendering utility — the only I/O in the CYPHER engine package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .core.types import AudioBuffer, DEFAULT_SAMPLE_RATE
from .core.voice import Voice
from .engine import DrumEngine
from .midi import NoteOff, NoteOn


def render_voice(
    voice: Voice,
    note: int = 36,
    velocity: float = 0.8,
    duration: float = 2.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    release_at: float | None = None,
) -> AudioBuffer:
    """Render a single voice trigger to a buffer."""
    total_frames = int(duration * sample_rate)
    release_frame = int(release_at * sample_rate) if release_at else None

    voice.trigger(note, velocity)

    chunk_size = 512
    chunks: list[AudioBuffer] = []
    rendered = 0

    while rendered < total_frames:
        n = min(chunk_size, total_frames - rendered)

        if release_frame and rendered >= release_frame and rendered - n < release_frame:
            voice.release(note)

        chunks.append(voice.process(n))
        rendered += n

    return np.concatenate(chunks)


def render_to_wav(
    buffer: AudioBuffer,
    path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    """Write audio buffer to WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    peak = np.max(np.abs(buffer))
    if peak > 1.0:
        buffer = buffer / peak
    elif peak > 0:
        buffer = buffer * (0.89 / peak)

    # Fade out last 5ms to avoid click at end of file
    fade_samples = min(int(0.005 * sample_rate), len(buffer))
    if fade_samples > 0:
        fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        buffer[-fade_samples:] *= fade

    sf.write(str(path), buffer, sample_rate, subtype="FLOAT")


def render_engine_pattern(
    engine: DrumEngine,
    events: list[tuple[float, NoteOn | NoteOff]],
    duration: float = 4.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> AudioBuffer:
    """Render a pattern of MIDI events through the engine."""
    total_frames = int(duration * sample_rate)
    chunk_size = 256

    events = sorted(events, key=lambda e: e[0])
    event_idx = 0

    chunks: list[AudioBuffer] = []
    rendered = 0

    while rendered < total_frames:
        n = min(chunk_size, total_frames - rendered)
        current_time = rendered / sample_rate

        while event_idx < len(events) and events[event_idx][0] <= current_time:
            engine.handle_midi(events[event_idx][1])
            event_idx += 1

        chunks.append(engine.process(n))
        rendered += n

    return np.concatenate(chunks)


def render_demo_sounds(output_dir: str | Path = "output") -> None:
    """Render demo WAV files for all voices."""
    output_dir = Path(output_dir)
    sr = DEFAULT_SAMPLE_RATE

    print(f"Rendering demo sounds to {output_dir}/")
    print(f"Sample rate: {sr}Hz\n")

    from .drum.sub808 import Sub808Voice
    from .drum.kick import KickVoice

    # =====================================================
    # 808 SUB — Serum-style + Zay: pitch envelope = transient
    # =====================================================
    # Param layout (12 params, 3 pages):
    #   0:DECAY  1:PUNCH  2:TONE  3:DRIVE
    #   4:SHAPE  5:NOISE  6:RELEASE  7:GLIDE
    #   8:FILTER  9:RESO  10:SAT  11:P.SUST
    print("--- 808 Sub (Serum + Zay) ---")

    # SHORT 808 — note held ~0.3s, fast release, medium punch
    print("  Short 808...")
    sub = Sub808Voice(sr)
    sub.params[0].value = 0.35   # DECAY: ~60ms pitch drop
    sub.params[1].value = 0.45   # PUNCH: ~16 semitones
    sub.params[6].value = 0.3    # RELEASE: ~300ms tail
    buf = render_voice(sub, note=36, velocity=0.95, duration=1.5, release_at=0.3)
    render_to_wav(buf, output_dir / "808_short.wav", sr)

    # MEDIUM 808 — note held ~1s
    print("  Medium 808...")
    sub = Sub808Voice(sr)
    sub.params[0].value = 0.35   # DECAY: ~60ms
    sub.params[1].value = 0.45   # PUNCH: ~16st
    sub.params[6].value = 0.4    # RELEASE: ~500ms tail
    buf = render_voice(sub, note=36, velocity=0.9, duration=3.0, release_at=1.0)
    render_to_wav(buf, output_dir / "808_medium.wav", sr)

    # LONG 808 — note held ~3s, long release
    print("  Long 808...")
    sub = Sub808Voice(sr)
    sub.params[0].value = 0.4    # DECAY: ~80ms
    sub.params[1].value = 0.4    # PUNCH: ~14st
    sub.params[6].value = 0.65   # RELEASE: ~2s tail
    buf = render_voice(sub, note=36, velocity=0.9, duration=8.0, release_at=3.0)
    render_to_wav(buf, output_dir / "808_long.wav", sr)

    # CLEAN — no drive, no tone, no punch, pure sustained sub bass (2000s feel)
    print("  Clean sub bass (no punch, no drive)...")
    sub = Sub808Voice(sr)
    sub.params[0].value = 0.3    # DECAY: doesn't matter much
    sub.params[1].value = 0.0    # PUNCH: zero — no pitch sweep
    sub.params[2].value = 0.0    # TONE: zero
    sub.params[3].value = 0.0    # DRIVE: zero
    sub.params[6].value = 0.4    # RELEASE: medium tail
    buf = render_voice(sub, note=36, velocity=0.9, duration=4.0, release_at=2.0)
    render_to_wav(buf, output_dir / "808_clean_sub.wav", sr)

    # SNAPPY PUNCH — fast decay, big pitch drop, short hold
    print("  Snappy punchy 808...")
    sub = Sub808Voice(sr)
    sub.params[0].value = 0.2    # DECAY: ~30ms — very fast pitch drop
    sub.params[1].value = 0.65   # PUNCH: ~23st — big drop
    sub.params[6].value = 0.3    # RELEASE: short tail
    buf = render_voice(sub, note=36, velocity=1.0, duration=1.5, release_at=0.4)
    render_to_wav(buf, output_dir / "808_snappy.wav", sr)

    # BOOMY — slow decay, moderate pitch drop, pitch sustain offset
    print("  Boomy 808...")
    sub = Sub808Voice(sr)
    sub.params[0].value = 0.55   # DECAY: ~120ms — slow pitch sweep
    sub.params[1].value = 0.35   # PUNCH: ~12st
    sub.params[6].value = 0.5    # RELEASE: medium-long tail
    sub.params[11].value = 0.3   # P.SUST: slight pitch offset
    buf = render_voice(sub, note=36, velocity=0.9, duration=4.0, release_at=1.5)
    render_to_wav(buf, output_dir / "808_boomy.wav", sr)

    # WARM DRIVE (soft clip)
    print("  Warm driven 808 (soft clip)...")
    sub = Sub808Voice(sr)
    sub.params[0].value = 0.35
    sub.params[1].value = 0.45
    sub.params[3].value = 0.4    # DRIVE: warm saturation
    sub.params[10].value = 0.25  # SAT: soft
    sub.params[6].value = 0.4
    buf = render_voice(sub, note=36, velocity=0.9, duration=3.0, release_at=1.0)
    render_to_wav(buf, output_dir / "808_warm_soft.wav", sr)

    # ZAY 808 — tape saturation, noise, filter
    print("  Zay 808 (tape sat, noise)...")
    sub = Sub808Voice(sr)
    sub.params[0].value = 0.4    # DECAY: ~80ms
    sub.params[1].value = 0.5    # PUNCH: ~18st
    sub.params[2].value = 0.3    # TONE: slight warmth
    sub.params[3].value = 0.65   # DRIVE: tape-driven
    sub.params[5].value = 0.35   # NOISE: ~5%
    sub.params[6].value = 0.45   # RELEASE
    sub.params[8].value = 0.7    # FILTER: ~6kHz
    sub.params[9].value = 0.15   # RESO: slight
    sub.params[10].value = 0.4   # SAT: tape
    buf = render_voice(sub, note=36, velocity=1.0, duration=3.0, release_at=1.0)
    render_to_wav(buf, output_dir / "808_zay.wav", sr)

    # HARD DRIVE
    print("  Hard driven 808...")
    sub = Sub808Voice(sr)
    sub.params[0].value = 0.35
    sub.params[1].value = 0.5
    sub.params[3].value = 0.6    # DRIVE: hard
    sub.params[10].value = 0.7   # SAT: hard clip
    sub.params[6].value = 0.4
    buf = render_voice(sub, note=36, velocity=1.0, duration=3.0, release_at=1.0)
    render_to_wav(buf, output_dir / "808_hard.wav", sr)

    # CRUSHED
    print("  Crushed 808...")
    sub = Sub808Voice(sr)
    sub.params[0].value = 0.35
    sub.params[1].value = 0.5
    sub.params[3].value = 0.85   # DRIVE: max
    sub.params[10].value = 1.0   # SAT: bitcrush
    sub.params[6].value = 0.45
    buf = render_voice(sub, note=36, velocity=1.0, duration=3.0, release_at=1.0)
    render_to_wav(buf, output_dir / "808_crushed.wav", sr)

    # TONE SWEEP — hear the harmonic blending
    print("  808 tone comparison (0%, 50%, 100%)...")
    for tone_name, tone_val in [("0pct", 0.0), ("50pct", 0.5), ("100pct", 1.0)]:
        sub = Sub808Voice(sr)
        sub.params[0].value = 0.35
        sub.params[1].value = 0.45
        sub.params[2].value = tone_val  # TONE
        sub.params[6].value = 0.35
        buf = render_voice(sub, note=36, velocity=0.9, duration=2.0, release_at=0.5)
        render_to_wav(buf, output_dir / f"808_tone_{tone_name}.wav", sr)

    # LEGATO GLIDE — trap bass slide
    print("  808 legato glide (trap slide)...")
    sub = Sub808Voice(sr)
    sub.params[0].value = 0.35
    sub.params[1].value = 0.45
    sub.params[7].value = 0.5    # GLIDE: medium
    sub.params[6].value = 0.5    # RELEASE
    sub.trigger(36, 0.9)         # C1
    frames1 = sub.process(int(0.6 * sr))
    sub.trigger(31, 0.9)         # G0 — slide down
    frames2 = sub.process(int(0.6 * sr))
    sub.trigger(36, 0.9)         # Back to C1
    frames3 = sub.process(int(0.8 * sr))
    sub.release(36)
    frames4 = sub.process(int(1.0 * sr))
    buf = np.concatenate([frames1, frames2, frames3, frames4])
    render_to_wav(buf, output_dir / "808_glide.wav", sr)

    # DIFFERENT DECAY TIMES — hear the pitch envelope
    print("  808 decay comparison (30ms, 80ms, 200ms)...")
    for decay_name, decay_val in [("30ms", 0.15), ("80ms", 0.38), ("200ms", 0.65)]:
        sub = Sub808Voice(sr)
        sub.params[0].value = decay_val
        sub.params[1].value = 0.5   # Same punch
        sub.params[6].value = 0.35  # RELEASE
        buf = render_voice(sub, note=36, velocity=0.9, duration=2.0, release_at=0.5)
        render_to_wav(buf, output_dir / f"808_decay_{decay_name}.wav", sr)

    # DIFFERENT PUNCH DEPTHS
    print("  808 punch comparison (0st, 12st, 24st, 36st)...")
    for st_name, punch_val in [("0st", 0.0), ("12st", 0.33), ("24st", 0.67), ("36st", 1.0)]:
        sub = Sub808Voice(sr)
        sub.params[0].value = 0.35  # Same decay
        sub.params[1].value = punch_val
        sub.params[6].value = 0.35  # RELEASE
        buf = render_voice(sub, note=36, velocity=0.9, duration=2.0, release_at=0.5)
        render_to_wav(buf, output_dir / f"808_punch_{st_name}.wav", sr)

    # =====================================================
    # KICK — sine + pitch envelope, Serum-style
    # =====================================================
    # Param layout (8 params, 2 pages):
    #   0:PUNCH  1:BODY  2:TONE  3:DRIVE
    #   4:CLICK  5:HOLD  6:ATTACK  7:CRUSH
    print("\n--- Kick ---")

    # DEFAULT — balanced hip hop kick
    print("  Default kick...")
    kick = KickVoice(sr)
    buf = render_voice(kick, note=36, velocity=0.9, duration=0.8)
    render_to_wav(buf, output_dir / "kick_default.wav", sr)

    # TRAP — subtle punch, tight body, snappy attack
    print("  Trap kick...")
    kick = KickVoice(sr)
    kick.params[0].value = 0.15  # PUNCH: subtle sweep (trap style)
    kick.params[1].value = 0.3   # BODY: tight
    kick.params[2].value = 0.6   # TONE: moderate brightness
    kick.params[5].value = 0.3   # HOLD: brief
    kick.params[6].value = 0.05  # ATTACK: snappy (0.9ms)
    buf = render_voice(kick, note=36, velocity=1.0, duration=0.5)
    render_to_wav(buf, output_dir / "kick_trap.wav", sr)

    # BOOM BAP — more punch, longer body, acoustic click, softer attack
    print("  Boom bap kick...")
    kick = KickVoice(sr)
    kick.params[0].value = 0.5   # PUNCH: noticeable sweep
    kick.params[1].value = 0.55  # BODY: medium
    kick.params[2].value = 0.4   # TONE: warm
    kick.params[4].value = 0.6   # CLICK: acoustic character
    kick.params[5].value = 0.5   # HOLD: more weight
    kick.params[6].value = 0.5   # ATTACK: softer (7ms)
    buf = render_voice(kick, note=36, velocity=0.9, duration=0.8)
    render_to_wav(buf, output_dir / "kick_boombap.wav", sr)

    # ACOUSTIC — heavy click, papery character
    print("  Acoustic kick...")
    kick = KickVoice(sr)
    kick.params[0].value = 0.35  # PUNCH: moderate
    kick.params[1].value = 0.45  # BODY: medium
    kick.params[2].value = 0.5   # TONE: natural
    kick.params[4].value = 0.9   # CLICK: heavy noise — woody/papery
    kick.params[5].value = 0.4   # HOLD: moderate
    buf = render_voice(kick, note=36, velocity=0.9, duration=0.8)
    render_to_wav(buf, output_dir / "kick_acoustic.wav", sr)

    # PAIRED WITH 808 — smart HPF, transient layer
    print("  Paired kick (with 808 at C1)...")
    kick = KickVoice(sr)
    kick.params[0].value = 0.2   # PUNCH
    kick.params[1].value = 0.25  # BODY: short
    kick.params[2].value = 0.65  # TONE: bright enough to cut through
    kick.params[4].value = 0.4   # CLICK: some snap
    kick.pair_808(32.7)           # Pair with 808 at C1
    buf = render_voice(kick, note=36, velocity=0.9, duration=0.5)
    render_to_wav(buf, output_dir / "kick_paired_808.wav", sr)

    # DRIVEN — soft clip warmth
    print("  Driven kick...")
    kick = KickVoice(sr)
    kick.params[0].value = 0.3
    kick.params[1].value = 0.4
    kick.params[3].value = 0.7   # DRIVE: warm saturation
    buf = render_voice(kick, note=36, velocity=1.0, duration=0.6)
    render_to_wav(buf, output_dir / "kick_driven.wav", sr)

    # CRUSHED — lo-fi destruction
    print("  Crushed kick...")
    kick = KickVoice(sr)
    kick.params[0].value = 0.3
    kick.params[1].value = 0.35
    kick.params[7].value = 0.6   # CRUSH
    buf = render_voice(kick, note=36, velocity=1.0, duration=0.6)
    render_to_wav(buf, output_dir / "kick_crushed.wav", sr)

    # =====================================================
    # FULL PATTERN — kick + 808
    # =====================================================
    print("\n--- Full pattern (140bpm) ---")
    engine = DrumEngine(sr)
    bpm = 140
    beat = 60.0 / bpm

    events: list[tuple[float, NoteOn | NoteOff]] = []

    for bar in range(4):
        offset = bar * 4 * beat

        # Kick on 1
        events.append((offset, NoteOn(37, 110)))

        # 808 — note on at beat 1, off just before beat 3, then new note
        events.append((offset, NoteOn(36, 100)))
        events.append((offset + 1.9 * beat, NoteOff(36)))
        events.append((offset + 2 * beat, NoteOn(36, 100)))
        events.append((offset + 3.9 * beat, NoteOff(36)))

    pattern_duration = 4 * 4 * beat + 1.0
    buf = render_engine_pattern(engine, events, duration=pattern_duration, sample_rate=sr)
    render_to_wav(buf, output_dir / "pattern_kick_808.wav", sr)

    wav_count = len(list(output_dir.glob("*.wav")))
    print(f"\nDone! {wav_count} WAV files in {output_dir}/")


if __name__ == "__main__":
    render_demo_sounds()
