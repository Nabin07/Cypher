"""Kick voice — sine + pitch envelope, Serum-style.

Architecture (same foundation as 808, tuned for kicks):

  OSCILLATOR:
    Pure sine, pitched down. The pitch envelope is what creates the
    "thump" — a fast sweep from elevated pitch down to the body tone.
    In trap kicks the sweep is subtle (2-8st, 5-15ms). Crank PUNCH
    for more boom bap / acoustic character.

  AMPLITUDE ENVELOPE:
    Attack (fast) → Hold (brief peak) → Decay (body) → silence.
    No sustain, no gate. Kicks are one-shot.
    The HOLD phase is what gives the kick weight — it sustains
    at peak level briefly before the body decays.

  NOISE CLICK:
    Optional filtered noise burst on attack. This is what makes it
    sound acoustic/papery vs clean/synthetic. 0% = pure sine thump,
    cranked = woody boom bap character.

  FILTER:
    Lowpass to tame highs. In the Serum tutorial this is applied
    after synthesis to shape the brightness.

  DISTORTION:
    Soft clip drive for warmth/aggression + optional bitcrush.

  PAIR 808:
    One-shot action that analyzes the 808's fundamental and applies
    a smart HPF to keep the kick out of the 808's sub range. The
    kick keeps its character — corrections are bounded (HPF capped
    at 150Hz, never more than ~3dB change in any band). Unpair to
    go back to the original sound.

Parameter layout (8 params, 2 pages):
  Simple (4 encoders):  PUNCH | BODY | TONE | DRIVE
  Advanced (4 encoders): CLICK | HOLD | ATTACK | CRUSH
"""

from __future__ import annotations

import numpy as np

from ..core.envelope import ADEnvelope
from ..core.filters import BiquadFilter
from ..core.oscillator import SineOscillator
from ..core.parameter import Curve, Parameter
from ..core.types import AudioBuffer, DEFAULT_SAMPLE_RATE
from ..core.voice import Voice
from ..core.waveshaper import apply_drive, bitcrush


class KickVoice(Voice):
    """Kick drum voice — sine + pitch envelope with 808 link mode."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        super().__init__(sample_rate)

        self._params = [
            # === Simple mode (page 1) ===
            Parameter(
                name="punch", label="PUNCH",
                min_val=0.5, max_val=24.0, default=1.0,
                unit="st", curve=Curve.EXPONENTIAL,
            ),
            Parameter(
                name="body", label="BODY",
                min_val=30.0, max_val=800.0, default=0.40,
                unit="ms", curve=Curve.EXPONENTIAL,
            ),
            Parameter(
                name="tone", label="TONE",
                min_val=200.0, max_val=12000.0, default=0.50,
                unit="Hz", curve=Curve.EXPONENTIAL,
            ),
            Parameter(
                name="drive", label="DRIVE",
                min_val=0.0, max_val=0.8, default=0.80,
                unit="%", curve=Curve.LINEAR,
            ),

            # === Advanced (page 2) ===
            Parameter(
                name="knock", label="KNOCK",
                min_val=5.0, max_val=80.0, default=0.75,
                unit="ms", curve=Curve.EXPONENTIAL,
            ),
            Parameter(
                name="hold", label="HOLD",
                min_val=1.0, max_val=80.0, default=0.16,
                unit="ms", curve=Curve.EXPONENTIAL,
            ),
            Parameter(
                name="attack", label="ATTACK",
                min_val=0.5, max_val=20.0, default=0.19,
                unit="ms", curve=Curve.EXPONENTIAL,
            ),
            Parameter(
                name="crush", label="CRUSH",
                min_val=0.0, max_val=1.0, default=0.05,
                unit="%", curve=Curve.LINEAR,
            ),
        ]

        # --- Sine oscillator ---
        self._osc = SineOscillator(sample_rate)

        # --- Pitch envelopes (dual: fast crack + slow body movement) ---
        self._pitch_env = ADEnvelope(sample_rate)
        self._pitch_env.attack_time = 0.0003  # near-instant
        self._pitch_env.curve = -6.0           # fast exponential drop

        self._pitch_env_slow = ADEnvelope(sample_rate)
        self._pitch_env_slow.attack_time = 0.0003
        self._pitch_env_slow.sustain_level = 0.0
        self._pitch_env_slow.curve = -2.5      # gentler — the body pitch movement

        # --- Amplitude envelope (attack → decay, one-shot) ---
        # Hold is handled manually between attack and decay.
        self._amp_env = ADEnvelope(sample_rate)
        self._amp_env.attack_time = 0.004  # ~4ms attack
        self._amp_env.sustain_level = 0.0   # one-shot — no sustain
        self._amp_env.curve = -3.5

        # --- Hold tracking ---
        self._hold_samples: int = 0
        self._hold_counter: int = 0
        self._amp_phase: str = "idle"
        self._amp_pos: int = 0
        self._attack_samples: int = max(1, int(0.004 * sample_rate))

        # --- Knock oscillator layer (mid-freq punch, the actual "hard knock") ---
        self._knock_osc = SineOscillator(sample_rate)
        self._knock_env = ADEnvelope(sample_rate)
        self._knock_env.attack_time = 0.0005   # 0.5ms — near-instant
        self._knock_env.sustain_level = 0.0
        self._knock_env.curve = -7.0           # fast punch decay
        self._knock_freq: float = 120.0        # updated per trigger

        # --- Filters ---
        self._lpf = BiquadFilter(sample_rate)   # tone filter
        self._hpf = BiquadFilter(sample_rate)   # link mode highpass

        # --- State ---
        self._active = False
        self._velocity: float = 1.0
        self._output_amplitude: float = 0.0
        self._body_freq: float = 60.0
        self._current_pitch_hz: float = 60.0

        # 808 pairing
        self._paired: bool = False
        self._paired_808_freq: float = 0.0
        self._linked_808_freq: float = 0.0

    @property
    def params(self) -> list[Parameter]:
        return self._params

    @property
    def is_active(self) -> bool:
        return self._active

    def set_linked_808_freq(self, freq_hz: float) -> None:
        """Called by the engine to pass the 808's current fundamental."""
        self._linked_808_freq = freq_hz

    def pair_808(self, freq_hz: float) -> dict:
        """Pair with an 808 — analyze and apply smart HPF.

        Returns a dict describing what was applied, for UI feedback.
        """
        self._paired = True
        self._paired_808_freq = freq_hz

        # HPF sits just above the 808's fundamental (1.2x)
        # Capped at 150Hz — beyond that we're cutting into the kick's body
        hpf = min(freq_hz * 1.2, 150.0)
        self._hpf.set_highpass(min(hpf, self.sample_rate * 0.45), 0.707)

        return {
            "paired": True,
            "808_freq": freq_hz,
            "hpf_freq": hpf,
        }

    def unpair_808(self) -> None:
        """Remove 808 pairing — kick returns to its original sound."""
        self._paired = False
        self._paired_808_freq = 0.0
        self._hpf.reset()

    def _get_hpf_freq(self) -> float:
        """Compute highpass when paired with an 808."""
        if not self._paired:
            return 0.0
        # Use live 808 freq if available (tracks pitch changes),
        # fall back to the freq from when we paired
        freq = self._linked_808_freq if self._linked_808_freq > 10.0 else self._paired_808_freq
        if freq > 10.0:
            return min(freq * 1.2, 150.0)
        return 0.0

    def _update_params(self) -> None:
        # Pitch envelope — fast drop for the thump
        # Sweep time: 5-15ms (fast for trap, scaled slightly with punch)
        punch_st = self._params[0].mapped
        sweep_ms = 5.0 + (punch_st / 24.0) * 10.0  # 5-15ms
        self._pitch_env.decay_time = sweep_ms / 1000.0

        # Amplitude envelope
        body_ms = self._params[1].mapped
        self._amp_env.decay_time = body_ms / 1000.0

        # Hold
        hold_ms = self._params[5].mapped
        self._hold_samples = int(hold_ms / 1000.0 * self.sample_rate)

        # Attack time from ATTACK param
        attack_ms = self._params[6].mapped
        self._attack_samples = max(1, int(attack_ms / 1000.0 * self.sample_rate))

        # Knock layer — mid-freq punch oscillator (capped at 200Hz to stay in knock zone)
        knock_decay_ms = self._params[4].mapped
        self._knock_env.decay_time = knock_decay_ms / 1000.0
        self._knock_freq = min(self._body_freq * 2.5, 200.0)

        # Slow pitch envelope — decay tracks body, adds the lower octave movement
        body_ms = self._params[1].mapped
        self._pitch_env_slow.decay_time = body_ms * 0.6 / 1000.0

        # Tone filter (lowpass)
        tone_freq = self._params[2].mapped
        self._lpf.set_lowpass(min(tone_freq, self.sample_rate * 0.45), 0.707)

        # Body frequency — set from MIDI note in trigger()
        # Pitch envelope sweeps down TO this frequency

    def trigger(self, note: int, velocity: float) -> None:
        self._velocity = velocity
        self._body_freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
        self._update_params()

        # Pitch envelopes
        self._pitch_env.trigger()
        self._pitch_env_slow.trigger()

        # Amplitude — we'll handle attack→hold→decay manually
        # Don't trigger the amp env yet — we build the envelope ourselves
        self._hold_counter = 0
        self._hold_active = False
        self._attack_done = False
        self._amp_phase = "attack"
        self._amp_pos = 0

        # Oscillator
        self._osc.reset()

        # Knock oscillator
        self._knock_env.trigger()
        self._knock_osc.reset()

        # Filters
        self._lpf.reset()
        self._hpf.reset()

        self._active = True

    def release(self, note: int) -> None:
        # Kicks are one-shot
        pass

    def all_notes_off(self) -> None:
        self._active = False
        self._amp_phase = "idle"
        self._pitch_env._stage = "idle"
        self._pitch_env._level = 0.0
        self._pitch_env_slow._stage = "idle"
        self._pitch_env_slow._level = 0.0
        self._knock_env._stage = "idle"
        self._knock_env._level = 0.0
        self._output_amplitude = 0.0

    def process(self, num_frames: int) -> AudioBuffer:
        if not self._active:
            return np.zeros(num_frames, dtype=np.float32)

        self._update_params()

        # === Dual pitch envelopes → frequency ===
        # Fast env: PUNCH semitones (the crack), Slow env: 12st fixed (the body movement)
        punch_st = self._params[0].mapped
        pitch_fast = self._pitch_env.process(num_frames)
        pitch_slow = self._pitch_env_slow.process(num_frames)

        combined_st = pitch_fast * punch_st + pitch_slow * 12.0
        freq = self._body_freq * np.power(2.0, combined_st / 12.0).astype(np.float32)
        self._current_pitch_hz = float(freq[-1]) if len(freq) > 0 else self._body_freq

        # === Sine oscillator ===
        signal = self._osc.process(freq)

        # === Knock layer — fast mid-freq punch ===
        if self._knock_env.is_active:
            knock_sig = self._knock_osc.process(
                np.full(num_frames, self._knock_freq, dtype=np.float32)
            )
            knock_env = self._knock_env.process(num_frames)
            signal = (signal + knock_sig * knock_env * 0.8).astype(np.float32)

        # === Amplitude envelope: attack → hold → decay (built manually) ===
        attack_samples = self._attack_samples
        hold_samples = self._hold_samples
        body_ms = self._params[1].mapped
        decay_samples = max(1, int(body_ms / 1000.0 * self.sample_rate))
        curve = -3.5

        amp = np.zeros(num_frames, dtype=np.float32)
        i = 0
        while i < num_frames and self._amp_phase != "idle":
            if self._amp_phase == "attack":
                n = min(attack_samples - self._amp_pos, num_frames - i)
                t = np.arange(self._amp_pos, self._amp_pos + n, dtype=np.float32) / attack_samples
                amp[i:i+n] = t
                self._amp_pos += n
                i += n
                if self._amp_pos >= attack_samples:
                    self._amp_phase = "hold"
                    self._amp_pos = 0

            elif self._amp_phase == "hold":
                n = min(hold_samples - self._amp_pos, num_frames - i)
                if n <= 0:
                    self._amp_phase = "decay"
                    self._amp_pos = 0
                    continue
                amp[i:i+n] = 1.0
                self._amp_pos += n
                i += n
                if self._amp_pos >= hold_samples:
                    self._amp_phase = "decay"
                    self._amp_pos = 0

            elif self._amp_phase == "decay":
                n = min(num_frames - i, 512)
                t = np.arange(self._amp_pos, self._amp_pos + n, dtype=np.float32) / decay_samples
                amp[i:i+n] = np.exp(curve * t)
                self._amp_pos += n
                level = float(amp[i+n-1]) if n > 0 else 0.0
                i += n
                if level < 0.001:
                    self._amp_phase = "idle"

        signal *= amp * self._velocity

        # === Tone filter (lowpass) ===
        tone_freq = self._params[2].mapped
        if tone_freq < 11000.0:
            signal = self._lpf.process(signal)

        # === Paired 808 HPF ===
        if self._paired:
            hpf_freq = self._get_hpf_freq()
            if hpf_freq > 25.0:
                self._hpf.set_highpass(min(hpf_freq, self.sample_rate * 0.45), 0.707)
                signal = self._hpf.process(signal)

        # === Drive (clean punch — transient emphasis, no saturation color) ===
        drive = self._params[3].mapped
        if drive > 0.005:
            signal = apply_drive(signal, drive, "punch")

        # === Crush ===
        crush = self._params[7].mapped  # CRUSH is still index 7
        if crush > 0.01:
            bit_depth = 16.0 - crush * 12.0
            downsample = max(1, int(crush * 6))
            signal = bitcrush(signal, bit_depth, downsample)

        self._output_amplitude = float(np.max(np.abs(signal))) if len(signal) > 0 else 0.0

        # Check if done
        if self._amp_phase == "idle" and not self._knock_env.is_active:
            self._active = False

        return signal

    def get_state(self) -> dict:
        return {
            "active": self._active,
            "amplitude": self._output_amplitude,
            "knock_env_level": self._knock_env.level,
            "knock_env_stage": self._knock_env.stage,
            "body_env_stage": self._amp_phase,
            "body_pitch_hz": self._current_pitch_hz,
            "paired": self._paired,
            "paired_808_freq": self._paired_808_freq,
            "linked_808_freq": self._linked_808_freq,
            "hpf_freq": self._get_hpf_freq(),
            "in_hold": self._amp_phase == "hold",
            "velocity": self._velocity,
        }
