"""MonoSynthVoice — two-oscillator subtractive synth voice.

Signal chain (PRD spec):
    OSC A [saw/sine/square/tri]  ──┐
                                   ├─ MIX ─→ VCF [LP/HP/BP + reso + filter env] ─→ VCA [ADSR] ─→ drive ─→ out
    OSC B [saw/sine/square/tri]  ──┘
                                         ↑ LFO (routable to filter or pitch)

Parameters (16, 4 pages of 4 encoders):
    OSC:    WAVE A | WAVE B | MIX    | DETUNE
    FILTER: CUTOFF | RESO   | F.ENV  | MODE
    AMP:    ATTACK | DECAY  | SUSTAIN | RELEASE
    MOD:    LFO RATE | LFO DEPTH | LFO DEST | NOISE

C++ portability notes:
  - All state is plain floats/ints in a flat struct.
  - process() operates on contiguous float32 buffers.
  - No closures, no generators, no dynamic dispatch in the hot path.
  - Parameter reads are cached per-block (one indirection, not per-sample).
  - Waveform generation uses phase accumulation — maps to a tight C++ loop.
"""

from __future__ import annotations

import numpy as np

from ..core.envelope import ADEnvelope
from ..core.filters import BiquadFilter, FILTER_LP, FILTER_MODE_NAMES
from ..core.lfo import LFO, LFO_NAMES
from ..core.oscillator import NoiseGenerator
from ..core.parameter import Curve, Parameter
from ..core.types import AudioBuffer, DEFAULT_SAMPLE_RATE, note_to_freq
from ..core.voice import Voice
from ..core.waveshaper import apply_drive

# Waveform types
WAVE_SAW = 0
WAVE_SINE = 1
WAVE_SQUARE = 2
WAVE_TRI = 3
WAVE_NAMES = ["SAW", "SIN", "SQR", "TRI"]


class MonoSynthVoice(Voice):
    """Two-oscillator subtractive synth voice.

    Each instance is one voice in the poly pool. All voices in a pool
    share the same Parameter list (params are owned by PolySynthVoice).
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        super().__init__(sample_rate)

        # --- Oscillator state ---
        # Two independent oscillators, each with its own phase
        self._phase_a: float = 0.0
        self._phase_b: float = 0.0

        # --- Noise ---
        self._noise_gen = NoiseGenerator()

        # --- LFO ---
        self._lfo = LFO(sample_rate)

        # --- Amp envelope (ADSR) ---
        self._amp_env = ADEnvelope(sample_rate)
        self._amp_env.sustain_level = 0.7

        # --- Filter envelope (independent ADSR, decays to 0) ---
        self._filter_env = ADEnvelope(sample_rate)
        self._filter_env.sustain_level = 0.0

        # --- Multimode filter ---
        self._filter = BiquadFilter(sample_rate)

        # --- Pitch ---
        self._current_freq: float = 440.0
        self._target_freq: float = 440.0
        self._glide_active: bool = False

        # --- Voice state ---
        self._current_note: int = -1
        self._velocity: float = 0.0
        self._active: bool = False

        # --- Cached param values (refreshed each process block) ---
        self._wave_a: int = WAVE_SAW
        self._wave_b: int = WAVE_SAW
        self._osc_mix: float = 0.0  # 0.0 = all A, 1.0 = all B
        self._detune_cents: float = 0.0
        self._cutoff_hz: float = 5000.0
        self._resonance: float = 0.0
        self._fenv_amount: float = 0.0
        self._filter_mode: int = FILTER_LP
        self._lfo_rate: float = 1.0
        self._lfo_depth: float = 0.0
        self._lfo_dest: int = 0  # 0=filter, 1=pitch
        self._noise_level: float = 0.0
        self._drive_amount: float = 0.0
        self._glide_ms: float = 0.0

        self._params: list[Parameter] = [
            # Page 1: OSC
            Parameter("wave_a", "WAVE A", 0.0, 3.0, 0.0, "", snap=4),
            Parameter("wave_b", "WAVE B", 0.0, 3.0, 0.0, "", snap=4),
            Parameter("mix", "MIX", 0.0, 1.0, 0.0, "%"),
            Parameter("detune", "DETUNE", 0.0, 50.0, 0.1, "ct"),
            # Page 2: FILTER
            Parameter("cutoff", "CUTOFF", 20.0, 20000.0, 0.8, "Hz", Curve.EXPONENTIAL),
            Parameter("reso", "RESO", 0.0, 0.95, 0.0, "%"),
            Parameter("fenv", "F.ENV", -1.0, 1.0, 0.5, ""),  # 0.5 norm = 0.0 mapped
            Parameter("filter_mode", "MODE", 0.0, 2.0, 0.0, "", snap=3),
            # Page 3: AMP
            Parameter("attack", "ATTACK", 1.0, 2000.0, 0.01, "ms", Curve.EXPONENTIAL),
            Parameter("decay", "DECAY", 1.0, 5000.0, 0.65, "ms", Curve.EXPONENTIAL),
            Parameter("sustain", "SUSTAIN", 0.0, 1.0, 0.7, "%"),
            Parameter("release", "RELEASE", 1.0, 5000.0, 0.55, "ms", Curve.EXPONENTIAL),
            # Page 4: MOD
            Parameter("lfo_rate", "LFO RATE", 0.1, 20.0, 0.2, "Hz", Curve.EXPONENTIAL),
            Parameter("lfo_depth", "LFO DEPTH", 0.0, 1.0, 0.0, "%"),
            Parameter("lfo_dest", "LFO DEST", 0.0, 1.0, 0.0, "", snap=2),
            Parameter("noise", "NOISE", 0.0, 1.0, 0.0, "%"),
        ]

    # ------------------------------------------------------------------
    # Voice interface
    # ------------------------------------------------------------------

    @property
    def params(self) -> list[Parameter]:
        return self._params

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def wave_a_name(self) -> str:
        return WAVE_NAMES[self._wave_a]

    @property
    def wave_b_name(self) -> str:
        return WAVE_NAMES[self._wave_b]

    @property
    def filter_mode_name(self) -> str:
        return FILTER_MODE_NAMES[self._filter_mode]

    def trigger(self, note: int, velocity: float) -> None:
        is_legato = self._active and self._current_note >= 0
        self._current_note = note
        self._velocity = velocity
        self._target_freq = note_to_freq(note)

        if is_legato and self._glide_ms > 1.5:
            self._glide_active = True
        else:
            self._current_freq = self._target_freq
            self._glide_active = False
            self._phase_a = 0.0
            self._phase_b = 0.0
            self._amp_env.trigger()
            self._filter_env.trigger()
            self._filter.reset()
            self._lfo.reset()

        self._active = True

    def release(self, note: int) -> None:
        if note == self._current_note:
            self._amp_env.release()
            self._filter_env.release()
            self._current_note = -1

    def all_notes_off(self) -> None:
        self._active = False
        self._amp_env._stage = "idle"
        self._amp_env._level = 0.0
        self._filter_env._stage = "idle"
        self._filter_env._level = 0.0
        self._current_note = -1

    def process(self, num_frames: int) -> AudioBuffer:
        if not self._active:
            return np.zeros(num_frames, dtype=np.float32)

        self._update_params()

        # --- LFO ---
        lfo_out = np.zeros(num_frames, dtype=np.float32)
        if self._lfo_depth > 0.001:
            lfo_out = self._lfo.process(num_frames) * self._lfo_depth

        # --- Pitch (with glide + optional LFO pitch mod) ---
        freqs = self._compute_pitch(num_frames)
        if self._lfo_dest == 1 and self._lfo_depth > 0.001:
            # LFO → pitch: up to +/- 2 semitones at full depth
            pitch_mod_st = lfo_out * 2.0
            freqs = freqs * np.power(2.0, pitch_mod_st / 12.0)

        # --- Dual oscillators ---
        osc_a = self._generate_waveform(self._wave_a, freqs, num_frames, is_b=False)
        # OSC B: detuned
        if self._osc_mix > 0.001:
            detune_ratio = 2.0 ** (self._detune_cents / 1200.0)
            freqs_b = freqs * detune_ratio
            osc_b = self._generate_waveform(self._wave_b, freqs_b, num_frames, is_b=True)
            signal = osc_a * (1.0 - self._osc_mix) + osc_b * self._osc_mix
        else:
            signal = osc_a

        # --- Noise mix ---
        if self._noise_level > 0.001:
            noise = self._noise_gen.process(num_frames)
            signal = signal * (1.0 - self._noise_level) + noise * self._noise_level

        # --- Filter envelope modulation ---
        filt_env = self._filter_env.process(num_frames)
        env_mod_octaves = self._fenv_amount * filt_env * 7.0
        avg_env_mod = float(np.mean(env_mod_octaves))

        # --- LFO → filter modulation ---
        lfo_filter_mod = 0.0
        if self._lfo_dest == 0 and self._lfo_depth > 0.001:
            # LFO → filter: up to +/- 3 octaves at full depth
            lfo_filter_mod = float(np.mean(lfo_out)) * 3.0

        effective_cutoff = self._cutoff_hz * (2.0 ** (avg_env_mod + lfo_filter_mod))
        effective_cutoff = max(20.0, min(effective_cutoff, self.sample_rate * 0.45))

        # Q from resonance
        q = 0.707 / max(1.0 - self._resonance, 0.05)
        self._filter.set_mode(self._filter_mode, effective_cutoff, q)
        signal = self._filter.process(signal)

        # --- VCA: amplitude envelope ---
        amp_env = self._amp_env.process(num_frames)
        signal *= amp_env * self._velocity

        # --- Drive ---
        if self._drive_amount > 0.001:
            signal = apply_drive(signal, self._drive_amount, "soft")

        # --- Check if done ---
        if not self._amp_env.is_active:
            self._active = False

        return signal

    def get_state(self) -> dict:
        return {
            "active": self._active,
            "wave_a": WAVE_NAMES[self._wave_a],
            "wave_b": WAVE_NAMES[self._wave_b],
            "osc_mix": self._osc_mix,
            "filter_mode": FILTER_MODE_NAMES[self._filter_mode],
            "note": self._current_note,
            "freq_hz": self._current_freq,
            "amp_env_stage": self._amp_env.stage,
            "amp_env_level": self._amp_env.level,
            "filter_env_stage": self._filter_env.stage,
            "cutoff_hz": self._cutoff_hz,
            "lfo_dest": "filter" if self._lfo_dest == 0 else "pitch",
            "lfo_depth": self._lfo_depth,
            "gliding": self._glide_active,
            "velocity": self._velocity,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_params(self) -> None:
        """Read current parameter values into cached fields."""
        # OSC
        self._wave_a = max(0, min(3, int(round(self._params[0].mapped))))
        self._wave_b = max(0, min(3, int(round(self._params[1].mapped))))
        self._osc_mix = self._params[2].mapped
        self._detune_cents = self._params[3].mapped

        # FILTER
        self._cutoff_hz = self._params[4].mapped
        self._resonance = self._params[5].mapped
        self._fenv_amount = self._params[6].mapped
        self._filter_mode = max(0, min(2, int(round(self._params[7].mapped))))

        # AMP envelope ADSR
        attack_s = self._params[8].mapped / 1000.0
        decay_s = self._params[9].mapped / 1000.0
        sustain = self._params[10].mapped
        release_s = self._params[11].mapped / 1000.0

        self._amp_env.attack_time = attack_s
        self._amp_env.decay_time = decay_s
        self._amp_env.sustain_level = sustain
        self._amp_env.release_time = release_s

        # Filter envelope: independent ADSR with same times but decays to 0
        self._filter_env.attack_time = attack_s * 0.5  # Filter opens faster
        self._filter_env.decay_time = decay_s
        self._filter_env.sustain_level = 0.0
        self._filter_env.release_time = release_s

        # MOD
        self._lfo_rate = self._params[12].mapped
        self._lfo_depth = self._params[13].mapped
        self._lfo_dest = max(0, min(1, int(round(self._params[14].mapped))))
        self._noise_level = self._params[15].mapped

        # Apply LFO settings
        self._lfo.rate_hz = self._lfo_rate
        # LFO wave is fixed to sine for now — can expose later
        self._lfo.wave = 0

        # Drive and glide are accessible but not on the main 4 pages
        # They keep their defaults (0 drive, 0 glide) unless set externally
        self._drive_amount = 0.0
        self._glide_ms = 0.0

    def _compute_pitch(self, num_frames: int) -> np.ndarray:
        """Per-sample frequency array with exponential glide."""
        if not self._glide_active:
            self._current_freq = self._target_freq
            return np.full(num_frames, self._current_freq, dtype=np.float64)

        glide_samples = max(1.0, self._glide_ms / 1000.0 * self.sample_rate)
        decay = np.exp(-5.0 / glide_samples)

        n = np.arange(num_frames, dtype=np.float64)
        freqs = self._target_freq + (self._current_freq - self._target_freq) * (decay ** n)

        self._current_freq = float(freqs[-1])
        if abs(self._current_freq - self._target_freq) / max(self._target_freq, 1.0) < 0.001:
            self._current_freq = self._target_freq
            self._glide_active = False

        return freqs

    def _generate_waveform(
        self, wave_type: int, freqs: np.ndarray, num_frames: int, is_b: bool
    ) -> AudioBuffer:
        """Generate waveform from phase accumulation.

        Args:
            wave_type: WAVE_SAW, WAVE_SINE, WAVE_SQUARE, or WAVE_TRI.
            freqs: Per-sample frequency array.
            num_frames: Buffer length.
            is_b: If True, uses OSC B phase state.
        """
        phase_inc = freqs / self.sample_rate
        cum_inc = np.cumsum(phase_inc)

        if is_b:
            phase = (self._phase_b + cum_inc) % 1.0
            self._phase_b = float(phase[-1]) if num_frames > 0 else self._phase_b
        else:
            phase = (self._phase_a + cum_inc) % 1.0
            self._phase_a = float(phase[-1]) if num_frames > 0 else self._phase_a

        if wave_type == WAVE_SAW:
            return (2.0 * phase - 1.0).astype(np.float32)
        elif wave_type == WAVE_SINE:
            return np.sin(2.0 * np.pi * phase).astype(np.float32)
        elif wave_type == WAVE_SQUARE:
            return np.where(phase < 0.5, np.float32(1.0), np.float32(-1.0))
        elif wave_type == WAVE_TRI:
            return (4.0 * np.abs(phase - 0.5) - 1.0).astype(np.float32)
        else:
            return np.zeros(num_frames, dtype=np.float32)
