"""FXEngine — global send-bus effects.

Topology:
    send_bus ──→ [reverb (100% wet)] ──→ [HPF] ──→ [LPF] ──┐
                                                           ├──→ output
    dry_bus  ─────────────────────────────────────────────┘

MODE presets choose reverb character (Plate, Chamber, Hall, Room, Ambience).
Each mode tweaks internal delay scaling, modulation, and damping defaults.

Parameters (8, 2 pages):
    VERB:  MIX | PREDELAY | DECAY | MODE
    TONE:  HIGHCUT | LOWCUT | DAMPING | SIZE

C++ portability notes:
    - Composes DattorroPlateReverb + two BiquadFilters, no new DSP.
    - Mode is a snap-int parameter, maps to a table lookup.
    - No allocation in process() path.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.filters import BiquadFilter
from ..core.parameter import Curve, Parameter
from ..core.reverb import DattorroPlateReverb
from ..core.types import AudioBuffer, DEFAULT_SAMPLE_RATE


# ── Mode presets ──────────────────────────────────────────────────────
MODE_PLATE = 0
MODE_CHAMBER = 1
MODE_HALL = 2
MODE_ROOM = 3
MODE_AMBIENCE = 4

MODE_NAMES = ["PLATE", "CHAMBER", "HALL", "ROOM", "AMBIENCE"]


def _seconds_to_feedback(rt60_seconds: float, sample_rate: int) -> float:
    """Convert an RT60 decay time (seconds) to a feedback coefficient.

    The Dattorro tank has roughly an 8000-sample average round-trip at its
    reference rate (29761 Hz). Scale to actual sample rate and solve:
        feedback^(rt60 / round_trip) = 0.001   (-60dB)
    """
    avg_round_trip_sec = 8000.0 / 29761.0
    n = max(0.5, rt60_seconds / avg_round_trip_sec)
    fb = math.pow(0.001, 1.0 / n)
    return min(0.99, max(0.1, fb))

# Each mode tweaks reverb internals:
#   decay_scale : multiplies user DECAY (shorter/longer than the knob)
#   size_scale  : multiplies user SIZE  (tighter/looser room)
#   mod_depth   : LFO excursion for chorus-like smearing
#   mod_rate    : LFO speed
#   damp_bias   : added to user DAMPING (darker/brighter baseline)
_MODE_PROFILES: dict[int, dict[str, float]] = {
    MODE_PLATE:    {"decay_scale": 1.00, "size_scale": 1.00, "mod_depth": 0.50, "mod_rate": 0.80, "damp_bias": 0.00},
    MODE_CHAMBER:  {"decay_scale": 0.75, "size_scale": 0.80, "mod_depth": 0.30, "mod_rate": 0.55, "damp_bias": 0.15},
    MODE_HALL:     {"decay_scale": 1.30, "size_scale": 1.25, "mod_depth": 0.70, "mod_rate": 0.40, "damp_bias": 0.20},
    MODE_ROOM:     {"decay_scale": 0.55, "size_scale": 0.60, "mod_depth": 0.25, "mod_rate": 0.90, "damp_bias": 0.10},
    MODE_AMBIENCE: {"decay_scale": 0.35, "size_scale": 0.45, "mod_depth": 0.45, "mod_rate": 1.20, "damp_bias": 0.30},
}


class FXEngine:
    """Global send-bus FX — reverb + output filters + mode presets.

    Takes send_bus + dry_bus, returns the final mixed output.
    MIX parameter controls how much wet signal is added to the dry.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

        # Reverb — always 100% wet; mix handled by FXEngine.
        self._reverb = DattorroPlateReverb(sample_rate)
        self._reverb.mix = 1.0

        # Output filters on wet path
        self._hp = BiquadFilter(sample_rate)
        self._lp = BiquadFilter(sample_rate)

        # Cached param values (refreshed each process block)
        self._mix: float = 0.3
        self._predelay_ms: float = 20.0
        self._decay_sec: float = 2.0  # RT60 seconds
        self._mode: int = 0
        self._highcut_hz: float = 18000.0
        self._lowcut_hz: float = 20.0
        self._damping: float = 0.3
        self._size: float = 0.5
        self._last_mode: int = -1

        # 8 params, 2 pages of 4
        self._params: list[Parameter] = [
            # Page 1: VERB
            Parameter("mix",      "MIX",      0.0,   1.0,    0.3,  "%"),
            Parameter("predelay", "PREDELAY", 0.0,   200.0,  0.1,  "ms"),
            Parameter("decay",    "DECAY",    0.3,   15.0,   0.3,  "s",  Curve.EXPONENTIAL),
            Parameter("mode",     "MODE",     0.0,   4.0,    0.0,  "", snap=5),
            # Page 2: TONE
            Parameter("highcut",  "HIGHCUT",  500.0, 20000.0, 1.0, "Hz", Curve.EXPONENTIAL),
            Parameter("lowcut",   "LOWCUT",   20.0,  2000.0,  0.0, "Hz", Curve.EXPONENTIAL),
            Parameter("damping",  "DAMPING",  0.0,   1.0,     0.3, "%"),
            Parameter("size",     "SIZE",     0.3,   1.5,     0.5, "%"),
        ]

    # ──────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────

    @property
    def params(self) -> list[Parameter]:
        return self._params

    @property
    def mode_name(self) -> str:
        return MODE_NAMES[self._mode]

    def _update_params(self) -> None:
        """Read param values into cached fields, push to underlying DSP."""
        self._mix = self._params[0].mapped
        self._predelay_ms = self._params[1].mapped
        self._decay_sec = self._params[2].mapped
        self._mode = max(0, min(4, int(round(self._params[3].mapped))))
        self._highcut_hz = self._params[4].mapped
        self._lowcut_hz = self._params[5].mapped
        self._damping = self._params[6].mapped
        self._size = self._params[7].mapped

        # Apply mode profile. DECAY is in seconds — convert to feedback coeff.
        profile = _MODE_PROFILES[self._mode]
        self._reverb.predelay_ms = self._predelay_ms
        self._reverb.decay = _seconds_to_feedback(
            self._decay_sec * profile["decay_scale"], self.sample_rate
        )
        self._reverb.damping = min(0.95, max(0.0, self._damping + profile["damp_bias"]))
        self._reverb.mod_depth = profile["mod_depth"] * self._size
        self._reverb.mod_rate = profile["mod_rate"]

        if self._mode != self._last_mode:
            # Flush the tank so tail from previous mode doesn't leak
            self._reverb.clear()
            self._last_mode = self._mode

        # Output filters
        self._lp.set_lowpass(self._highcut_hz, 0.707)
        self._hp.set_highpass(max(20.0, self._lowcut_hz), 0.707)

    def process(self, send_buf: AudioBuffer, dry_buf: AudioBuffer) -> AudioBuffer:
        """Mix dry + wet(send).

        send_buf : voices that are routed through FX
        dry_buf  : all voices (FX is additive, not replacing dry signal)

        Returns: dry_buf + wet * mix
        """
        self._update_params()

        # Reverb on the send bus
        wet = self._reverb.process(send_buf)

        # Output filters on the wet signal
        if self._lowcut_hz > 20.5:
            wet = self._hp.process(wet)
        wet = self._lp.process(wet)

        return dry_buf + wet * self._mix

    def clear(self) -> None:
        """Reset all internal state — call when disabling or changing modes."""
        self._reverb.clear()
        self._hp.reset()
        self._lp.reset()

    def get_state(self) -> dict:
        return {
            "active": False,  # FX has no note state — the UI expects this key
            "mix": self._mix,
            "predelay_ms": self._predelay_ms,
            "decay_sec": self._decay_sec,
            "mode": self.mode_name,
            "highcut_hz": self._highcut_hz,
            "lowcut_hz": self._lowcut_hz,
            "damping": self._damping,
            "size": self._size,
        }

    # ──────────────────────────────────────────────────────────────────
    # Voice-like no-ops (so the UI can treat FX like a "voice" tab)
    # ──────────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return False

    def trigger(self, note: int, velocity: float) -> None:
        pass

    def release(self, note: int) -> None:
        pass

    def all_notes_off(self) -> None:
        pass
