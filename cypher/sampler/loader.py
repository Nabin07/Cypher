"""Sample loading + folder scanning.

Loads WAVs via soundfile, resamples to project rate if needed, mono sums
stereo for now (sampler is mono for v1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from ..core.types import AudioBuffer, DEFAULT_SAMPLE_RATE
from .sidecar import read_sidecar, write_sidecar, SampleMeta
from .tempo import detect_bpm


WAV_EXTS = {".wav", ".wave", ".aif", ".aiff", ".flac"}


def pick_folder_dialog(initial: str | None = None) -> str | None:
    """Open a native folder-picker dialog. Returns absolute path or None.

    Uses AppleScript on macOS (tkinter + pygame crash when both touch the
    main thread). Falls back to tkinter on Linux/Windows.
    """
    import platform
    import subprocess

    if platform.system() == "Darwin":
        initial_arg = (
            f' default location POSIX file "{initial}"' if initial else ""
        )
        script = (
            'tell application "System Events" to activate\n'
            f'set f to choose folder with prompt "Select samples folder"{initial_arg}\n'
            "return POSIX path of f"
        )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                p = result.stdout.strip()
                return p or None
        except Exception:
            return None
        return None

    # Linux / Windows: tkinter
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(initialdir=initial or str(Path.home()))
    root.destroy()
    return path or None


def scan_folder(folder: str | Path, recursive: bool = True) -> list[Path]:
    """Return all audio files in a folder, sorted alphabetically."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    it = folder.rglob("*") if recursive else folder.iterdir()
    results = [
        p for p in it
        if p.is_file() and p.suffix.lower() in WAV_EXTS
    ]
    results.sort(key=lambda p: p.name.lower())
    return results


def load_sample(
    path: str | Path, target_sample_rate: int = DEFAULT_SAMPLE_RATE
) -> tuple[AudioBuffer, int]:
    """Load an audio file → (mono_float32_buffer, source_sample_rate).

    Stereo is summed to mono. No resampling is done here — the sampler
    voice handles pitch/playback-rate via its own resampling logic.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        # Sum stereo → mono
        data = data.mean(axis=1, dtype=np.float32)
    return data.astype(np.float32, copy=False), int(sr)


def save_file_dialog(
    default_name: str = "export.wav", initial: str | None = None
) -> str | None:
    """Open a native save-file dialog. Returns absolute path or None.

    AppleScript on macOS (same reason as folder picker — avoids tkinter +
    pygame main-thread conflict). tkinter fallback elsewhere.
    """
    import platform
    import subprocess

    if platform.system() == "Darwin":
        initial_arg = (
            f' default location POSIX file "{initial}"' if initial else ""
        )
        script = (
            'tell application "System Events" to activate\n'
            f'set f to choose file name with prompt "Save as"'
            f' default name "{default_name}"{initial_arg}\n'
            "return POSIX path of f"
        )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                p = result.stdout.strip()
                return p or None
        except Exception:
            return None
        return None

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.asksaveasfilename(
        initialdir=initial or str(Path.home()),
        initialfile=default_name,
        defaultextension=".wav",
    )
    root.destroy()
    return path or None


def load_sample_with_meta(
    path: str | Path, target_sample_rate: int = DEFAULT_SAMPLE_RATE
) -> tuple[AudioBuffer, int, SampleMeta]:
    """Load a sample plus its metadata (BPM, confidence, user_corrected).

    If a sidecar JSON exists next to the sample, use it. Otherwise run
    tempo detection once and persist the result. Detection is skipped
    when the sidecar says `user_corrected=True`.
    """
    audio, src_sr = load_sample(path, target_sample_rate)
    existing = read_sidecar(path)
    if existing is not None and "bpm" in existing:
        return audio, src_sr, existing

    bpm, conf = detect_bpm(audio, src_sr)
    meta: SampleMeta = {
        "bpm": float(bpm),
        "confidence": float(conf),
        "user_corrected": False,
    }
    write_sidecar(path, meta)
    return audio, src_sr, meta


def format_duration(frames: int, sample_rate: int) -> str:
    """Pretty-print duration from frame count."""
    total_sec = frames / max(1, sample_rate)
    if total_sec < 1.0:
        return f"{total_sec * 1000:.0f}ms"
    return f"{total_sec:.2f}s"
