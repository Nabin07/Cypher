"""808 Sub voice — covers both clean and Zay-style 808 sounds.

Architecture (Serum-style, expanded):

  OSCILLATOR STAGE:
    Pure sine blended with harmonically richer waveshape via TONE knob.
    TONE 0% = pure sine. TONE 100% = triangle-ish with more harmonics.
    This replaces Serum's wavetable position concept.

  NOISE LAYER:
    Filtered noise burst on attack. Adds brightness/presence to the
    transient. Amount is 0-15%, decays with its own fast envelope.

  PITCH ENVELOPE:
    The transient maker. Fast decay from (base + PUNCH semitones) down
    to base pitch. SHAPE controls the contour — where the pitch peak
    sits in time. Center = standard exp drop. Left = snappier shark-fin.
    Right = slower rise ("womp").

  FILTER:
    Lowpass filter with drive and resonance. Filter drive adds harmonics
    musically (different from post-distortion). Cutoff is clamped high
    enough to never lose the fundamental.

  DISTORTION:
    Post-filter saturation. Tape (Zay sound), soft clip, hard clip, or crush.
    Amount is limited so cranking it can't destroy the 808 character.

  AMPLITUDE:
    Gated envelope. Sustains while note held, fades on release.

Parameter layout:
  Simple (4 encoders):  DECAY | PUNCH | TONE | DRIVE
  Advanced page 2:      SHAPE | NOISE | RELEASE | GLIDE
  Advanced page 3:      FILTER | RESO | SAT TYPE | P.SUST
"""

from __future__ import annotations

import numpy as np

from ..core.envelope import ADEnvelope, PitchEnvelope
from ..core.filters import BiquadFilter
from ..core.oscillator import NoiseGenerator, SineOscillator
from ..core.parameter import Curve, Parameter
from ..core.types import AudioBuffer, DEFAULT_SAMPLE_RATE, note_to_freq
from ..core.voice import Voice
from ..core.waveshaper import apply_drive, bitcrush


class Sub808Voice(Voice):
    """808 Sub bass voice — clean through Zay and everything between."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        super().__init__(sample_rate)

        self._params = [
            # === Simple mode (page 1) ===
            Parameter(
                name="decay", label="DECAY",
                min_val=20.0, max_val=300.0, default=0.35,
                unit="ms", curve=Curve.EXPONENTIAL,
            ),
            Parameter(
                name="punch", label="PUNCH",
                min_val=0.0, max_val=36.0, default=0.45,
                unit="st", curve=Curve.LINEAR,
            ),
            Parameter(
                name="tone", label="TONE",
                min_val=0.0, max_val=1.0, default=0.0,
                unit="%", curve=Curve.LINEAR,
            ),
            Parameter(
                name="drive", label="DRIVE",
                min_val=0.0, max_val=0.85, default=0.0,
                unit="%", curve=Curve.LINEAR,
            ),

            # === Advanced page 2 ===
            Parameter(
                name="shape", label="SHAPE",
                min_val=0.0, max_val=1.0, default=0.25,
                unit="", curve=Curve.LINEAR,
            ),
            Parameter(
                name="noise", label="NOISE",
                min_val=0.0, max_val=0.15, default=0.0,
                unit="%", curve=Curve.LINEAR,
            ),
            Parameter(
                name="release", label="RELEASE",
                min_val=30.0, max_val=5000.0, default=0.35,
                unit="ms", curve=Curve.EXPONENTIAL,
            ),
            Parameter(
                name="glide", label="GLIDE",
                min_val=10.0, max_val=500.0, default=0.3,
                unit="ms", curve=Curve.EXPONENTIAL,
            ),

            # === Advanced page 3 ===
            Parameter(
                name="filter", label="FILTER",
                min_val=200.0, max_val=16000.0, default=0.85,
                unit="Hz", curve=Curve.EXPONENTIAL,
            ),
            Parameter(
                name="reso", label="RESO",
                min_val=0.0, max_val=0.6, default=0.0,
                unit="%", curve=Curve.LINEAR,
            ),
            Parameter(
                name="sat_type", label="SAT",
                min_val=0.0, max_val=4.0, default=0.0,
                unit="", curve=Curve.LINEAR, snap=5,
            ),
            Parameter(
                name="p_sustain", label="P.SUST",
                min_val=0.0, max_val=0.15, default=0.0,
                unit="%", curve=Curve.LINEAR,
            ),
        ]

        # --- Oscillator ---
        self._osc = SineOscillator(sample_rate)

        # --- Noise layer ---
        self._noise_gen = NoiseGenerator()
        self._noise_env = ADEnvelope(sample_rate)
        self._noise_env.attack_time = 0.0005
        self._noise_env.curve = -6.0
        self._noise_filter = BiquadFilter(sample_rate)

        # --- Pitch modulation envelope (transient) ---
        self._pitch_mod_env = ADEnvelope(sample_rate)
        self._pitch_mod_env.attack_time = 0.0005

        # --- Pitch glide (legato) ---
        self._pitch_glide = PitchEnvelope(sample_rate)

        # --- Amplitude gate ---
        self._amp_env = ADEnvelope(sample_rate)
        self._amp_env.attack_time = 0.002
        self._amp_env.sustain_level = 1.0
        self._amp_env.curve = -3.0

        # --- Filter ---
        self._filter = BiquadFilter(sample_rate)

        # --- State ---
        self._active = False
        self._current_note: int = -1
        self._base_freq: float = 32.7
        self._velocity: float = 1.0
        self._output_amplitude: float = 0.0
        self._current_pitch_hz: float = 32.7

        # --- Trigger mode ---
        self._trigger_mode: str = "classic"  # "classic" or "oneshot"

        # --- Anti-click crossfade ---
        self._xfade_len: int = int(0.003 * sample_rate)  # 3ms
        self._xfade_buf: np.ndarray | None = None
        self._xfade_pos: int = 0
        self._last_sample: float = 0.0  # last rendered sample value

    @property
    def params(self) -> list[Parameter]:
        return self._params

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def _sat_char_name(self) -> str:
        """Display name for current SAT position."""
        val = self._params[10].mapped
        if val < 0.05:
            return "off"
        elif val < 1.0:
            return "soft"
        elif val < 2.0:
            return "tape"
        elif val < 3.0:
            return "hard"
        else:
            return "crush"

    def _update_params(self) -> None:
        # Pitch mod envelope
        decay_ms = self._params[0].mapped
        self._pitch_mod_env.decay_time = decay_ms / 1000.0
        self._pitch_mod_env.sustain_level = 0.0
        self._pitch_mod_env.curve = -4.0

        # Noise envelope — decays slightly slower than pitch for brightness
        self._noise_env.decay_time = (decay_ms * 1.5) / 1000.0
        noise_center = 4000.0 + self._params[2].mapped * 8000.0  # TONE affects noise too
        self._noise_filter.set_bandpass(min(noise_center, 20000.0), q=1.0)

        # Amplitude gate release
        release_ms = self._params[6].mapped
        self._amp_env.decay_time = release_ms / 1000.0
        self._amp_env.sustain_level = 1.0

        # Glide
        glide_ms = self._params[7].mapped
        self._pitch_glide.glide_time = glide_ms / 1000.0

        # Filter
        cutoff = self._params[8].mapped
        reso = self._params[9].mapped
        q = 0.707 + reso * 8.0  # 0.707 (flat) to ~5.5 (resonant)
        self._filter.set_lowpass(min(cutoff, self.sample_rate * 0.45), q)

    def _apply_tone(self, sine_signal: AudioBuffer) -> AudioBuffer:
        """Blend harmonics into the sine based on TONE knob.

        TONE 0% = pure sine
        TONE 100% = sine + 2nd + 3rd harmonics (warm, rich, like moving
                    the wavetable position in Serum)

        We add harmonics via waveshaping — a polynomial that introduces
        even and odd harmonics in a controlled, musical way. Cheaper than
        additive synthesis and naturally band-limited.
        """
        tone = self._params[2].mapped
        if tone < 0.01:
            return sine_signal

        # Chebyshev-style polynomial waveshaping for controlled harmonics
        # T2(x) = 2x^2 - 1 adds 2nd harmonic (warm, even)
        # T3(x) = 4x^3 - 3x adds 3rd harmonic (adds body)
        x = sine_signal
        h2 = 2.0 * x * x - 1.0        # 2nd harmonic
        h3 = 4.0 * x * x * x - 3.0 * x  # 3rd harmonic

        # Blend: more TONE = more harmonics
        # 2nd harmonic comes in first (warmer), 3rd comes in later (richer)
        h2_amount = min(tone * 2.0, 1.0) * 0.3   # 0-30% of 2nd
        h3_amount = max(0, tone - 0.3) * 0.4 * 0.2  # 0-8% of 3rd, delayed

        result = sine_signal + h2 * h2_amount + h3 * h3_amount

        # Normalize to prevent level jump
        peak = np.max(np.abs(result))
        if peak > 1.0:
            result /= peak

        return result.astype(np.float32)

    def _apply_saturation(self, signal: AudioBuffer, drive_amount: float,
                          sat_val: float) -> AudioBuffer:
        """Continuous saturation with smooth crossfades between types.

        sat_val zones (mapped 0-4):
          0.0      = OFF (bypass)
          0.0-1.0  = SOFT — ramps in gradually
          1.0-2.0  = SOFT→TAPE crossfade
          2.0-3.0  = TAPE→HARD crossfade
          3.0-4.0  = HARD→CRUSH crossfade (crush keeps hard clip underneath)
        """
        if sat_val < 0.05:
            return signal

        if sat_val <= 1.0:
            # SOFT zone: intensity ramps with position
            intensity = sat_val  # 0→1
            return apply_drive(signal, drive_amount * intensity, "soft")

        elif sat_val <= 2.0:
            # SOFT→TAPE crossfade
            blend = sat_val - 1.0  # 0→1
            soft_out = apply_drive(signal, drive_amount, "soft")
            tape_out = apply_drive(signal, drive_amount, "tape")
            return (soft_out * (1.0 - blend) + tape_out * blend).astype(np.float32)

        elif sat_val <= 3.0:
            # TAPE→HARD crossfade
            blend = sat_val - 2.0  # 0→1
            tape_out = apply_drive(signal, drive_amount, "tape")
            hard_out = apply_drive(signal, drive_amount, "hard")
            return (tape_out * (1.0 - blend) + hard_out * blend).astype(np.float32)

        else:
            # HARD→CRUSH crossfade — crush keeps hard clip for distortion character
            blend = min(sat_val - 3.0, 1.0)  # 0→1
            hard_out = apply_drive(signal, drive_amount, "hard")
            # Bitcrush the hard-clipped signal so distortion is maintained
            bit_depth = 16.0 - blend * 12.0  # 16→4 bits
            downsample = max(1, int(blend * 6))
            crush_out = bitcrush(hard_out, bit_depth, downsample)
            return (hard_out * (1.0 - blend) + crush_out * blend).astype(np.float32)

    def _shaped_pitch_mod(self, pitch_mod: AudioBuffer) -> AudioBuffer:
        """Apply SHAPE to the pitch modulation curve.

        SHAPE controls where the pitch peak sits in time:
          0.0 = instant peak, fast exponential drop (snappy)
          0.25 = shark fin — fast rise to peak at 25%, slow fall (Zay style)
          0.5 = symmetric — rise and fall are equal
          1.0 = slow rise, fast drop (reverse, "womp")

        The standard exponential decay (shape=0) is what we had before.
        The shark fin (shape ~0.25) is the Zay 808 character.
        """
        shape = self._params[4].mapped
        if shape < 0.05:
            # Standard: no reshaping needed, already exponential decay
            return pitch_mod

        n = len(pitch_mod)
        if n == 0:
            return pitch_mod

        # Build a shaped multiplier curve
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)

        # Peak position (0 = start, 1 = end)
        peak_pos = shape

        # Build envelope: linear rise to peak, exponential fall after
        rise = np.where(t < peak_pos, t / max(peak_pos, 0.001), 1.0)
        fall_t = np.where(t >= peak_pos, (t - peak_pos) / max(1.0 - peak_pos, 0.001), 0.0)
        fall = np.exp(-4.0 * fall_t)

        shaped = np.where(t < peak_pos, rise, fall).astype(np.float32)

        # Blend between original pitch_mod and shaped version
        # At shape=0, use 100% original. At shape>0, increasingly use shaped.
        blend = min(shape * 4.0, 1.0)  # Full shaped by shape=0.25
        return (pitch_mod * (1.0 - blend) + shaped * blend).astype(np.float32)

    def trigger(self, note: int, velocity: float) -> None:
        self._velocity = velocity
        tune_st = self._params[3] if len(self._params) > 3 else None
        self._base_freq = note_to_freq(note)
        self._update_params()

        # --- Anti-click crossfade when retriggering active voice ---
        # Ramp from the exact last sample value down to 0 — no gap at block boundary
        if self._active and abs(self._last_sample) > 0.001:
            xn = self._xfade_len
            self._xfade_buf = np.linspace(self._last_sample, 0.0, xn, dtype=np.float32)
            self._xfade_pos = 0

        is_legato = (self._active and self._current_note >= 0
                     and self._trigger_mode == "classic")

        if is_legato:
            # Legato — glide pitch, keep oscillator + amp running
            self._pitch_glide.glide_to(self._base_freq)
            self._pitch_mod_env.trigger()
            noise_amount = self._params[5].mapped
            if noise_amount > 0.001:
                self._noise_env.trigger()
        else:
            # Fresh trigger or one-shot retrigger
            self._pitch_glide.end_hz = self._base_freq
            self._pitch_glide.start_hz = self._base_freq
            self._pitch_glide.slide_time = 0.001
            self._pitch_glide.trigger(self._base_freq)

            self._amp_env.trigger()
            self._pitch_mod_env.trigger()
            self._osc.reset()
            self._filter.reset()
            self._active = True

            noise_amount = self._params[5].mapped
            if noise_amount > 0.001:
                self._noise_env.trigger()
                self._noise_filter.reset()

        self._current_note = note

    def release(self, note: int) -> None:
        if self._trigger_mode == "oneshot":
            return  # One-shot ignores release — plays full decay
        if note != self._current_note and note != 0:
            return
        self._amp_env.release()
        self._current_note = -1

    def all_notes_off(self) -> None:
        self._active = False
        self._current_note = -1
        self._amp_env._stage = "idle"
        self._amp_env._level = 0.0
        self._pitch_mod_env._stage = "idle"
        self._noise_env._stage = "idle"
        self._output_amplitude = 0.0

    def process(self, num_frames: int) -> AudioBuffer:
        if not self._active:
            return np.zeros(num_frames, dtype=np.float32)

        self._update_params()

        # --- Base pitch (handles legato glide) ---
        base_freq = self._pitch_glide.process(num_frames)

        # --- Pitch mod envelope (transient) ---
        punch_st = self._params[1].mapped
        p_sustain = self._params[11].mapped
        pitch_mod_raw = self._pitch_mod_env.process(num_frames)

        pitch_mod = pitch_mod_raw

        # Semitone offset: punch * pitch_mod + sustain offset
        sustain_st = p_sustain * punch_st
        st_offset = pitch_mod * punch_st + sustain_st

        # Semitones to frequency multiplier
        freq_mult = np.power(2.0, st_offset / 12.0).astype(np.float32)
        freq = base_freq * freq_mult

        self._current_pitch_hz = float(freq[-1]) if len(freq) > 0 else self._base_freq

        # --- Oscillator (pure sine) ---
        signal = self._osc.process(freq)

        # --- TONE: blend in harmonics ---
        signal = self._apply_tone(signal)

        # --- Noise burst layer ---
        noise_amount = self._params[5].mapped
        if noise_amount > 0.001 and self._noise_env.is_active:
            noise = self._noise_gen.process(num_frames)
            noise = self._noise_filter.process(noise)
            noise_env = self._noise_env.process(num_frames)
            signal = signal * (1.0 - noise_amount) + noise * noise_env * noise_amount * 3.0
            signal = signal.astype(np.float32)

        # --- Distortion (pre-filter so filter tames the harmonics) ---
        drive_amount = self._params[3].mapped
        sat_val = self._params[10].mapped  # 0=OFF, 1=SOFT, 2=TAPE, 3=HARD, 4=CRUSH

        if drive_amount > 0.005 and sat_val > 0.05:
            signal = self._apply_saturation(signal, drive_amount, sat_val)

        # --- Filter (post-distortion so it controls brightness) ---
        cutoff = self._params[8].mapped
        if cutoff < 15000.0:
            signal = self._filter.process(signal)

        # --- Amplitude gate ---
        amp = self._amp_env.process(num_frames)
        signal *= amp * self._velocity

        # --- Anti-click crossfade ---
        if self._xfade_buf is not None:
            remain = len(self._xfade_buf) - self._xfade_pos
            n = min(remain, num_frames)
            if n > 0:
                fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
                xfade = self._xfade_buf[self._xfade_pos:self._xfade_pos + n]
                signal[:n] = signal[:n] * fade_in + xfade
                self._xfade_pos += n
            if self._xfade_pos >= len(self._xfade_buf):
                self._xfade_buf = None

        self._last_sample = float(signal[-1]) if len(signal) > 0 else 0.0
        self._output_amplitude = float(np.max(np.abs(signal))) if len(signal) > 0 else 0.0

        if not self._amp_env.is_active:
            self._active = False

        return signal

    def get_state(self) -> dict:
        return {
            "active": self._active,
            "amplitude": self._output_amplitude,
            "current_pitch_hz": self._current_pitch_hz,
            "base_freq_hz": self._base_freq,
            "pitch_mod_level": self._pitch_mod_env.level,
            "pitch_mod_stage": self._pitch_mod_env.stage,
            "amp_env_level": self._amp_env.level,
            "amp_env_stage": self._amp_env.stage,
            "is_gliding": self._pitch_glide.is_gliding,
            "sat_character": self._sat_char_name,
            "tone_amount": self._params[2].mapped,
            "noise_active": self._noise_env.is_active,
            "current_note": self._current_note,
            "velocity": self._velocity,
            "trigger_mode": self._trigger_mode,
        }
