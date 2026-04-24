"""Envelope generators for drum synthesis."""

from __future__ import annotations

import numpy as np

from .types import AudioBuffer, DEFAULT_SAMPLE_RATE


class ADEnvelope:
    """Attack-Decay envelope for percussive sounds.

    Produces a value from 0.0 to 1.0 over time:
      - Attack: ramp from 0 to 1 over attack_time
      - Decay: exponential fall from 1 to ~0 over decay_time

    Supports gate mode: if `sustain_level` > 0, the envelope holds at
    that level after attack until `release()` is called, then decays.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.attack_time: float = 0.001  # seconds (1ms click avoidance)
        self.decay_time: float = 0.5     # seconds
        self.sustain_level: float = 0.0  # 0.0 = no sustain (one-shot), >0 = gate mode
        self.release_time: float | None = None  # None = use decay_time
        self.curve: float = -4.0         # negative = exponential decay shape

        self._stage: str = "idle"  # idle, attack, decay, sustain, release
        self._position: int = 0   # sample position within current stage
        self._level: float = 0.0  # current envelope level
        self._release_level: float = 0.0
        self._retrigger_level: float = 0.0  # for soft retrigger (anti-pop)

    def trigger(self) -> None:
        """Start the envelope — soft retrigger from current level to avoid pops."""
        self._retrigger_level = self._level  # remember where we are
        self._stage = "attack"
        self._position = 0

    def release(self) -> None:
        """Release the envelope (for gate/sustain mode)."""
        if self._stage == "sustain":
            self._stage = "release"
            self._release_level = self._level
            self._position = 0
        elif self._stage == "attack":
            # Release during attack — jump to release
            self._stage = "release"
            self._release_level = self._level
            self._position = 0

    def process(self, num_frames: int) -> AudioBuffer:
        """Generate envelope values for num_frames samples."""
        output = np.zeros(num_frames, dtype=np.float32)

        if self._stage == "idle":
            return output

        i = 0
        while i < num_frames and self._stage != "idle":
            if self._stage == "attack":
                attack_samples = max(1, int(self.attack_time * self.sample_rate))
                remaining = attack_samples - self._position
                n = min(remaining, num_frames - i)

                t = (np.arange(self._position, self._position + n, dtype=np.float32)
                     / attack_samples)
                # Ramp from retrigger level to 1.0 (soft retrigger)
                base = self._retrigger_level
                output[i:i + n] = base + (1.0 - base) * t
                self._position += n
                self._level = output[i + n - 1] if n > 0 else self._level
                i += n

                if self._position >= attack_samples:
                    self._level = 1.0
                    if self.sustain_level > 0:
                        self._stage = "sustain"
                    else:
                        self._stage = "decay"
                    self._position = 0

            elif self._stage == "sustain":
                # Hold at sustain level until release
                n = num_frames - i
                output[i:i + n] = self.sustain_level
                self._level = self.sustain_level
                i += n

            elif self._stage == "decay":
                decay_samples = max(1, int(self.decay_time * self.sample_rate))
                n = min(num_frames - i, 512)  # process in chunks

                t = (np.arange(self._position, self._position + n, dtype=np.float32)
                     / decay_samples)
                # Exponential decay curve
                env = np.exp(self.curve * t)
                output[i:i + n] = env
                self._position += n
                self._level = float(output[i + n - 1]) if n > 0 else 0.0
                i += n

                # End when level is actually inaudible, not just when time is up
                if self._level < 0.001:
                    self._stage = "idle"
                    self._level = 0.0

            elif self._stage == "release":
                release_time = self.release_time if self.release_time is not None else self.decay_time
                release_samples = max(1, int(release_time * self.sample_rate))
                n = min(num_frames - i, 512)

                t = (np.arange(self._position, self._position + n, dtype=np.float32)
                     / release_samples)
                env = self._release_level * np.exp(self.curve * t)
                output[i:i + n] = env
                self._position += n
                self._level = float(output[i + n - 1]) if n > 0 else 0.0
                i += n

                # End when level is actually inaudible
                if self._level < 0.001:
                    self._stage = "idle"
                    self._level = 0.0

        return output

    @property
    def is_active(self) -> bool:
        return self._stage != "idle"

    @property
    def position(self) -> float:
        """Normalized envelope position 0.0–1.0 for UI animation."""
        if self._stage == "idle":
            return 1.0
        if self._stage == "attack":
            attack_samples = max(1, int(self.attack_time * self.sample_rate))
            return self._position / attack_samples
        if self._stage == "sustain":
            return 0.0  # holding
        if self._stage in ("decay", "release"):
            total = max(1, int(self.decay_time * self.sample_rate))
            return self._position / total
        return 0.0

    @property
    def level(self) -> float:
        return self._level

    @property
    def stage(self) -> str:
        return self._stage


class TrapEnvelope:
    """Two-stage decay envelope for modern trap 808s.

    Matches the amplitude shape found in real trap 808 samples:
      - Attack: fast ramp to peak (1-10ms)
      - Hold: brief hold at peak level
      - Decay 1: fast initial drop to a sustain level (the "punch" fading)
      - Sustain: holds at a level for an extended period (the "body")
      - Decay 2: slow fade from sustain to silence (the "tail")

    This creates the characteristic 808 shape where you get a punch
    up front, then the bass sits at a level before slowly dying.

    Gate mode: note-on sustains at the sustain level indefinitely,
    note-off starts decay 2.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

        # Stage times
        self.attack_time: float = 0.003     # 3ms — fast but not clicky
        self.hold_time: float = 0.005       # 5ms hold at peak
        self.decay1_time: float = 0.05      # 50ms — punch fade
        self.sustain_level: float = 0.75    # Body level (0-1)
        self.decay2_time: float = 2.0       # Tail fade in seconds

        # Curve shapes (negative = exponential)
        self.decay1_curve: float = -3.0     # Punch dropoff shape
        self.decay2_curve: float = -2.5     # Tail fade shape (gentler)

        # Gate mode
        self.gate_mode: bool = False

        # Internal state
        self._stage: str = "idle"  # idle, attack, hold, decay1, sustain, decay2, release
        self._position: int = 0
        self._level: float = 0.0
        self._release_level: float = 0.0
        self._retrigger_level: float = 0.0

    def trigger(self) -> None:
        self._retrigger_level = self._level
        self._stage = "attack"
        self._position = 0

    def release(self) -> None:
        """Note-off. In gate mode, starts decay2 from current level."""
        if self._stage in ("sustain", "hold", "decay1", "attack"):
            self._stage = "decay2"
            self._release_level = self._level
            self._position = 0

    def process(self, num_frames: int) -> AudioBuffer:
        output = np.zeros(num_frames, dtype=np.float32)

        if self._stage == "idle":
            return output

        i = 0
        while i < num_frames and self._stage != "idle":
            if self._stage == "attack":
                attack_samples = max(1, int(self.attack_time * self.sample_rate))
                remaining = attack_samples - self._position
                n = min(remaining, num_frames - i)

                t = (np.arange(self._position, self._position + n, dtype=np.float32)
                     / attack_samples)
                base = self._retrigger_level
                output[i:i + n] = base + (1.0 - base) * t
                self._position += n
                self._level = float(output[i + n - 1]) if n > 0 else self._level
                i += n

                if self._position >= attack_samples:
                    self._level = 1.0
                    self._stage = "hold"
                    self._position = 0

            elif self._stage == "hold":
                hold_samples = max(1, int(self.hold_time * self.sample_rate))
                remaining = hold_samples - self._position
                n = min(remaining, num_frames - i)

                output[i:i + n] = 1.0
                self._level = 1.0
                self._position += n
                i += n

                if self._position >= hold_samples:
                    self._stage = "decay1"
                    self._position = 0

            elif self._stage == "decay1":
                d1_samples = max(1, int(self.decay1_time * self.sample_rate))
                remaining = d1_samples - self._position
                n = min(remaining, num_frames - i)

                t = (np.arange(self._position, self._position + n, dtype=np.float32)
                     / d1_samples)
                # Exponential decay from 1.0 to sustain_level
                drop = 1.0 - self.sustain_level
                env = 1.0 - drop * (1.0 - np.exp(self.decay1_curve * t))
                output[i:i + n] = env
                self._position += n
                self._level = float(output[i + n - 1]) if n > 0 else self._level
                i += n

                if self._position >= d1_samples:
                    self._level = self.sustain_level
                    if self.gate_mode:
                        self._stage = "sustain"
                    else:
                        self._stage = "decay2"
                        self._release_level = self.sustain_level
                    self._position = 0

            elif self._stage == "sustain":
                # Hold at sustain level until release
                n = num_frames - i
                output[i:i + n] = self.sustain_level
                self._level = self.sustain_level
                i += n

            elif self._stage == "decay2":
                d2_samples = max(1, int(self.decay2_time * self.sample_rate))
                remaining = d2_samples - self._position
                n = min(remaining, num_frames - i)

                t = (np.arange(self._position, self._position + n, dtype=np.float32)
                     / d2_samples)
                start_level = self._release_level if self._release_level > 0 else self.sustain_level
                env = start_level * np.exp(self.decay2_curve * t)
                output[i:i + n] = env
                self._position += n
                self._level = float(output[i + n - 1]) if n > 0 else 0.0
                i += n

                if self._position >= d2_samples:
                    self._stage = "idle"
                    self._level = 0.0

        return output

    @property
    def is_active(self) -> bool:
        return self._stage != "idle"

    @property
    def position(self) -> float:
        """Normalized position 0-1 for UI."""
        if self._stage == "idle":
            return 1.0
        if self._stage == "attack":
            total = max(1, int(self.attack_time * self.sample_rate))
            return self._position / total
        if self._stage == "hold":
            return 0.0
        if self._stage == "decay1":
            total = max(1, int(self.decay1_time * self.sample_rate))
            return self._position / total
        if self._stage == "sustain":
            return 0.0  # Holding
        if self._stage == "decay2":
            total = max(1, int(self.decay2_time * self.sample_rate))
            return self._position / total
        return 0.0

    @property
    def level(self) -> float:
        return self._level

    @property
    def stage(self) -> str:
        return self._stage


class PitchEnvelope:
    """Exponential pitch sweep — the soul of the 808.

    Sweeps from start_hz to end_hz with an exponential curve.
    Also supports legato glide: smoothly transition between two
    target frequencies without retriggering.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.start_hz: float = 200.0   # trigger sweep start
        self.end_hz: float = 50.0      # trigger sweep end (base pitch)
        self.slide_time: float = 0.05  # seconds for trigger sweep
        self.glide_time: float = 0.08  # seconds for legato note-to-note glide

        self._mode: str = "idle"  # idle, sweep, glide, settled
        self._position: int = 0
        self._current_hz: float = 50.0
        self._glide_from_hz: float = 50.0
        self._glide_to_hz: float = 50.0

    def trigger(self, target_hz: float) -> None:
        """Trigger a fresh pitch sweep down to target_hz."""
        self.end_hz = target_hz
        self._mode = "sweep"
        self._position = 0
        self._current_hz = self.start_hz

    def glide_to(self, target_hz: float) -> None:
        """Legato glide from current pitch to new target (no retrigger)."""
        self._glide_from_hz = self._current_hz
        self._glide_to_hz = target_hz
        self.end_hz = target_hz
        self._mode = "glide"
        self._position = 0

    def process(self, num_frames: int) -> AudioBuffer:
        """Generate per-sample frequency values."""
        output = np.full(num_frames, self._current_hz, dtype=np.float32)

        if self._mode == "idle" or self._mode == "settled":
            output[:] = self.end_hz
            self._current_hz = self.end_hz
            return output

        i = 0
        while i < num_frames and self._mode not in ("idle", "settled"):
            if self._mode == "sweep":
                sweep_samples = max(1, int(self.slide_time * self.sample_rate))
                remaining = sweep_samples - self._position
                n = min(remaining, num_frames - i)

                t = (np.arange(self._position, self._position + n, dtype=np.float32)
                     / sweep_samples)
                # Exponential sweep from start to end
                output[i:i + n] = self.end_hz + (self.start_hz - self.end_hz) * np.exp(-5.0 * t)
                self._position += n
                self._current_hz = output[i + n - 1] if n > 0 else self._current_hz
                i += n

                if self._position >= sweep_samples:
                    self._mode = "settled"
                    self._current_hz = self.end_hz

            elif self._mode == "glide":
                glide_samples = max(1, int(self.glide_time * self.sample_rate))
                remaining = glide_samples - self._position
                n = min(remaining, num_frames - i)

                t = (np.arange(self._position, self._position + n, dtype=np.float32)
                     / glide_samples)
                # Exponential interpolation between frequencies (sounds musical)
                ratio = self._glide_to_hz / max(self._glide_from_hz, 0.001)
                output[i:i + n] = self._glide_from_hz * (ratio ** t)
                self._position += n
                self._current_hz = output[i + n - 1] if n > 0 else self._current_hz
                i += n

                if self._position >= glide_samples:
                    self._mode = "settled"
                    self._current_hz = self._glide_to_hz

        # Fill any remaining with settled frequency
        if i < num_frames:
            output[i:] = self._current_hz

        return output

    @property
    def current_hz(self) -> float:
        return self._current_hz

    @property
    def is_sweeping(self) -> bool:
        return self._mode == "sweep"

    @property
    def is_gliding(self) -> bool:
        return self._mode == "glide"

    @property
    def progress(self) -> float:
        """0.0 = just started, 1.0 = settled. For UI animation."""
        if self._mode == "sweep":
            total = max(1, int(self.slide_time * self.sample_rate))
            return self._position / total
        elif self._mode == "glide":
            total = max(1, int(self.glide_time * self.sample_rate))
            return self._position / total
        return 1.0
