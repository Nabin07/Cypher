# Sampler BPM Matching + Global Metronome

## What changes

### Backend
1. **`cypher/sampler/tempo.py`** (new)
   - `detect_bpm(audio, sr) -> (bpm, confidence)`
   - Spectral-flux onset envelope → autocorrelation → peak in 60–200 BPM window.
   - scipy-only (no librosa — stays portable to C++). Confidence = normalized peak height, 0..1.

2. **`cypher/sampler/sidecar.py`** (new)
   - `<sample>.cypher.json` next to the sample file.
   - Schema: `{"bpm": float, "confidence": float, "user_corrected": bool}`.
   - `read_sidecar(path)` / `write_sidecar(path, data)`.

3. **`cypher/sampler/loader.py`**
   - `load_sample_with_meta(path)` wraps `load_sample`. If sidecar exists, returns it. Otherwise runs detection, writes sidecar, returns it.

4. **`cypher/sampler/sampler.py`**
   - `SampleSlot` gains: `sample_bpm`, `bpm_confidence`, `match_mode`, `user_corrected`.
   - `get_matched_buffer(slot, semitones)` — cached phase-vocoder stretch for `(project_bpm / sample_bpm)` composed with pitch.
   - `trigger_pad` uses matched buffer when `slot.match_mode`.
   - `slot.invalidate_cache()` on BPM edit.

### UI
5. **Status line** — `Sample: 92 · Project: 140 · [Free] [Match]`
   - Pre-select Match if `confidence ≥ 0.6` AND `0.85 ≤ ratio ≤ 1.15`.
   - `?` + `[Tap tempo]` if detection failed.
6. **Fine-tune row** (Match active only) — `Sample BPM: 92.0 [−][+] ×2 ÷2 [Tap]`
   - 0.1 BPM nudge, debounce 150ms → invalidate cache.
   - Sets `user_corrected=True`, rewrites sidecar.

### Metronome
7. **`cypher/core/metronome.py`** (new) — `Metronome(project, sample_rate)`.
   - `running` flag, `play()/pause()`, `process(frames) → mono audio`.
   - Sine pip at 1000Hz (beat-1 accent 1400Hz), 2ms attack / 40ms decay, LP ~1.2kHz, peak -24 dB.
8. **Audio callback** — mix metronome dry into main buf.
9. **Bottom UI strip** — big BPM readout, `[−][+]` nudge, click-to-type, `[▶/⏸]`, beat-flash. Visible on all tabs.

## Order of work

1. Plan doc (this).
2. Tempo detector.
3. Sidecar.
4. SampleSlot BPM fields + Match stretch in trigger path.
5. Status line.
6. Fine-tune row.
7. Metronome engine.
8. Metronome UI strip.
9. End-to-end test.

## Out of scope (v1)

- Real-time varispeed (debounced re-render only).
- Auto-re-detect after user correction.
- Stretch ratio as %.
- Per-slice micro-timing.

## C++ port notes

- `tempo.py` is numpy-only → small FFT (kissfft) translation.
- Sidecar JSON lives next to WAV on SD.
- Metronome DSP allocation-free.
- Match cache: flat `(slot × 25 × 2) + (semi+12) × 2 + match_bit`.
