"""Global metronome — soft click synced to project BPM.

Emits short sine-pip clicks at each beat. Beat-1 gets a slightly brighter
and louder accent so downbeats are audible without being sharp.

Sound design:
    - Pitch: 1000 Hz (off-beats), 1400 Hz (beat 1).
    - Envelope: 2 ms attack, 40 ms exponential decay.
    - Gentle low-pass shaped via a half-raised-cosine tail so the click
      never has a hard edge.
    - Peak amplitude: -24 dB. Quiet by design. The UI offers no gain;
      if needed, user adjusts system/interface volume.

C++ portability notes:
    - No per-block allocation. Pre-rendered click buffers.
    - One float counter advances each process() call.
"""

from __future__ import annotations

import numpy as np

from .types import AudioBuffer, DEFAULT_SAMPLE_RATE


CLICK_MS = 45.0           # total click duration
ATTACK_MS = 2.0
CLICK_FREQ_BEAT = 1000.0
CLICK_FREQ_ACCENT = 1400.0
CLICK_PEAK = 10.0 ** (-24.0 / 20.0)   # -24 dB


def _render_click(freq: float, sr: int) -> AudioBuffer:
    n = int(sr * CLICK_MS / 1000.0)
    attack_n = int(sr * ATTACK_MS / 1000.0)
    t = np.arange(n, dtype=np.float32) / sr
    # Sine body
    body = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    # Smooth envelope: raised-cosine attack + exponential decay
    env = np.ones(n, dtype=np.float32)
    if attack_n > 1:
        env[:attack_n] = 0.5 * (1.0 - np.cos(np.pi * np.arange(attack_n) / attack_n))
    decay = np.exp(-5.0 * np.arange(n, dtype=np.float32) / n).astype(np.float32)
    env *= decay
    return (body * env * CLICK_PEAK).astype(np.float32)


class Metronome:
    """Generates a metronome click stream into an audio callback.

    Tempo follows `project.bpm`. Beats per bar is fixed at 4 for v1.
    """

    __slots__ = (
        "_project", "_sr", "_click_beat", "_click_accent",
        "_running", "_samples_per_beat", "_phase_samples",
        "_beat_counter", "_pending_click", "_pending_offset",
        "_last_tick_time",
    )

    def __init__(self, project, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self._project = project
        self._sr = int(sample_rate)
        self._click_beat = _render_click(CLICK_FREQ_BEAT, self._sr)
        self._click_accent = _render_click(CLICK_FREQ_ACCENT, self._sr)
        self._running: bool = False
        # Phase accumulator in samples. When it exceeds samples_per_beat,
        # a click fires and phase wraps.
        self._samples_per_beat: float = self._sr * 60.0 / max(1.0, project.bpm)
        self._phase_samples: float = 0.0
        # 0..3 counter; beat 0 is the accent.
        self._beat_counter: int = 0
        # Click playhead (for multi-block clicks): None when idle.
        self._pending_click: AudioBuffer | None = None
        self._pending_offset: int = 0
        # Last monotonic time a click fired (for UI flash).
        self._last_tick_time: float = 0.0

    # ── Transport ────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._running

    def play(self) -> None:
        if self._running:
            return
        # Restart from beat 1 on play.
        self._beat_counter = 0
        self._phase_samples = 0.0
        self._pending_click = self._click_accent
        self._pending_offset = 0
        self._running = True

    def pause(self) -> None:
        self._running = False
        self._pending_click = None
        self._pending_offset = 0

    def toggle(self) -> None:
        if self._running:
            self.pause()
        else:
            self.play()

    def reset_phase(self) -> None:
        """Call after project BPM changes to keep the next downbeat aligned."""
        self._phase_samples = 0.0
        self._beat_counter = 0

    @property
    def last_tick_time(self) -> float:
        return self._last_tick_time

    # ── Audio ────────────────────────────────────────────────────────

    def process(self, num_frames: int) -> AudioBuffer:
        out = np.zeros(num_frames, dtype=np.float32)
        if not self._running:
            return out

        # Keep samples_per_beat live with the project BPM
        self._samples_per_beat = self._sr * 60.0 / max(1.0, self._project.bpm)

        i = 0
        # Finish playing any click that started in a previous block.
        # Note: phase tracking and click emission are independent. Playing
        # pending samples must NOT stall the beat timer.
        if self._pending_click is not None:
            remaining = len(self._pending_click) - self._pending_offset
            n = min(remaining, num_frames)
            if n > 0:
                out[:n] += self._pending_click[
                    self._pending_offset:self._pending_offset + n
                ]
                self._pending_offset += n
            if self._pending_offset >= len(self._pending_click):
                self._pending_click = None
                self._pending_offset = 0
            # Do NOT advance i past these frames — phase must still tick
            # over them in the loop below.

        # Advance phase through the whole block, possibly firing new
        # clicks along the way. Click buffers are mixed on top of any
        # pending-click audio already written to `out`.
        import math
        while i < num_frames:
            frames_left = num_frames - i
            # Ceil so we never stall on sub-sample fractional distance.
            # Min 1 so we always advance at least one frame.
            delta = self._samples_per_beat - self._phase_samples
            frames_to_next_beat = max(1, int(math.ceil(delta)))
            if frames_to_next_beat > frames_left:
                self._phase_samples += frames_left
                i = num_frames
                break
            # Advance up to the beat boundary, then fire.
            self._phase_samples += frames_to_next_beat
            i += frames_to_next_beat
            if self._phase_samples >= self._samples_per_beat - 0.5:
                self._phase_samples -= self._samples_per_beat
                self._beat_counter = (self._beat_counter + 1) % 4
                click = (self._click_accent if self._beat_counter == 0
                         else self._click_beat)
                n = min(len(click), num_frames - i)
                if n > 0:
                    out[i:i + n] += click[:n]
                if n < len(click):
                    self._pending_click = click
                    self._pending_offset = n
                i += n
                import time as _t
                self._last_tick_time = _t.monotonic()

        return out
