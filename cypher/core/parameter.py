"""Parameter system — bridge between DSP engine, hardware encoders, and UI."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class Curve(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"


@dataclass
class Parameter:
    """A named, ranged, labeled parameter that maps to a hardware encoder.

    Internally stores a normalized 0.0–1.0 value. The `mapped` property
    applies the curve and maps to the real [min_val, max_val] range.

    The encoder always sends the same delta regardless of parameter range.
    The UI always reads `mapped` for the human-readable display value.
    The DSP always reads `mapped` for the musically meaningful value.
    """

    name: str        # Machine name: "pitch", "decay", "drive"
    label: str       # Display label for screen above encoder: "PITCH", "DECAY"
    min_val: float   # Mapped minimum (e.g., 30.0 Hz)
    max_val: float   # Mapped maximum (e.g., 80.0 Hz)
    default: float   # Default normalized value 0.0–1.0
    unit: str = ""   # Display unit: "Hz", "ms", "%"
    curve: Curve = Curve.LINEAR
    snap: int = 0    # 0 = continuous, >0 = N discrete values (e.g. 4 for SAW/SIN/PLS/TRI)

    _value: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._value = max(0.0, min(1.0, self.default))

    @property
    def value(self) -> float:
        """Normalized value 0.0–1.0."""
        return self._value

    @value.setter
    def value(self, v: float) -> None:
        self._value = max(0.0, min(1.0, v))

    @property
    def mapped(self) -> float:
        """Value mapped to real range with curve applied."""
        t = self._value

        if self.curve == Curve.EXPONENTIAL:
            # Exponential: great for frequency and time params.
            # Maps 0–1 to min–max on an exponential scale.
            if self.min_val <= 0:
                # Fallback to linear if min is zero (can't do log of 0)
                return self.min_val + t * (self.max_val - self.min_val)
            return self.min_val * ((self.max_val / self.min_val) ** t)

        elif self.curve == Curve.LOGARITHMIC:
            # Logarithmic: more control at the top of the range.
            if t <= 0:
                return self.min_val
            log_t = math.log10(1 + 9 * t) / math.log10(10)
            return self.min_val + log_t * (self.max_val - self.min_val)

        else:
            # Linear
            return self.min_val + t * (self.max_val - self.min_val)

    @property
    def display_value(self) -> str:
        """Formatted value for screen display."""
        val = self.mapped
        if val >= 1000:
            return f"{val:.0f}{self.unit}"
        elif val >= 100:
            return f"{val:.0f}{self.unit}"
        elif val >= 10:
            return f"{val:.1f}{self.unit}"
        else:
            return f"{val:.2f}{self.unit}"

    def nudge(self, delta: float) -> None:
        """Increment normalized value by delta. Used by hardware encoders.

        For discrete params (snap > 0), steps by exactly one position
        regardless of delta magnitude — direction is all that matters.
        """
        if self.snap > 1:
            step = 1.0 / (self.snap - 1)
            # Snap current to nearest grid point, then step +/- 1
            idx = round(self._value / step)
            idx += 1 if delta > 0 else -1
            self.value = idx * step
        else:
            self.value = self._value + delta

    def reset(self) -> None:
        """Reset to default value."""
        self._value = self.default

    def to_dict(self) -> dict:
        """Snapshot for UI state."""
        return {
            "name": self.name,
            "label": self.label,
            "value": self._value,
            "mapped": self.mapped,
            "display": self.display_value,
            "unit": self.unit,
        }
