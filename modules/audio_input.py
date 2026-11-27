"""Audio input module: load, validate and prepare audio for Whisper.

Accepts a file path or raw bytes (uploaded file, microphone recording) and
returns mono float32 samples resampled to 16 kHz.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

import config

logger = logging.getLogger(__name__)


class AudioValidationError(ValueError):
    """Raised when an audio file cannot be used for analysis."""


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    duration: float
    source_name: str


def _check_extension(name: str) -> None:
    extension = Path(name).suffix.lower()
    if extension not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(config.ALLOWED_EXTENSIONS)
        raise AudioValidationError(
            f"Nepalaikomas failo formatas „{extension or 'be plėtinio'}“. Leidžiami: {allowed}."
        )


def _prepare(path: str, source_name: str) -> AudioData:
    try:
        samples, sample_rate = librosa.load(
            path, sr=config.TARGET_SAMPLE_RATE, mono=True
        )
    except Exception as exc:  # decoder errors differ between backends
        raise AudioValidationError(
            f"Nepavyko nuskaityti garso failo „{source_name}“: {exc}"
        ) from exc

    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise AudioValidationError("Garso failas tuščias.")

    duration = samples.size / sample_rate
    if duration < config.MIN_AUDIO_SECONDS:
        raise AudioValidationError(f"Įrašas per trumpas ({duration:.2f} s).")
    if config.MAX_AUDIO_SECONDS and duration > config.MAX_AUDIO_SECONDS:
        raise AudioValidationError(
            f"Įrašas per ilgas ({duration:.0f} s). Riba: {config.MAX_AUDIO_SECONDS} s."
        )
    if float(np.max(np.abs(samples))) < config.SILENCE_PEAK_THRESHOLD:
        raise AudioValidationError("Įraše girdima tik tyla.")

    logger.info("Loaded %s: %.1f s", source_name, duration)
    return AudioData(samples, sample_rate, duration, source_name)


def load_audio_path(path: str | os.PathLike) -> AudioData:
    """Load audio from a file on disk."""
    path = Path(path)
    if not path.is_file():
        raise AudioValidationError(f"Failas nerastas: {path}")
    _check_extension(path.name)
    return _prepare(str(path), path.name)


def load_audio_bytes(data: bytes, filename: str) -> AudioData:
    """Load audio from bytes (Streamlit upload or microphone recording)."""
    if not data:
        raise AudioValidationError("Gauti tušti duomenys.")
    _check_extension(filename)
    suffix = Path(filename).suffix.lower()
    # A real file lets librosa fall back to ffmpeg for formats such as m4a.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return _prepare(tmp_path, filename)
    finally:
        os.unlink(tmp_path)
