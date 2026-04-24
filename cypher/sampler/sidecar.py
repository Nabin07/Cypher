"""Sidecar metadata for samples — `<sample>.cypher.json` beside the WAV.

Stores detected BPM, confidence, and whether the user has corrected it.
Travels with the sample across rename/move/copy and survives on SD card
for the C++ port target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class SampleMeta(TypedDict, total=False):
    bpm: float
    confidence: float
    user_corrected: bool


def sidecar_path(sample_path: str | Path) -> Path:
    p = Path(sample_path)
    return p.with_suffix(p.suffix + ".cypher.json")


def read_sidecar(sample_path: str | Path) -> SampleMeta | None:
    sp = sidecar_path(sample_path)
    if not sp.is_file():
        return None
    try:
        with open(sp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        out: SampleMeta = {}
        if "bpm" in data:
            out["bpm"] = float(data["bpm"])
        if "confidence" in data:
            out["confidence"] = float(data["confidence"])
        if "user_corrected" in data:
            out["user_corrected"] = bool(data["user_corrected"])
        return out
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def write_sidecar(sample_path: str | Path, meta: SampleMeta) -> bool:
    sp = sidecar_path(sample_path)
    try:
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(dict(meta), f, indent=2)
        return True
    except OSError:
        return False
