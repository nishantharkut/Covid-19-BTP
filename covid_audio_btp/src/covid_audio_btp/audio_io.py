from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
import zipfile
from typing import Iterator

import librosa
import numpy as np
import soundfile as sf

from covid_audio_btp.config import AUDIO_CONFIG, AudioConfig


def split_archive_member_path(path: str | Path) -> tuple[Path, str] | None:
    """Return `(archive_path, member_name)` for paths written as `archive.zip::member`."""
    text = str(path)
    if "::" not in text:
        return None
    archive_text, member_name = text.split("::", 1)
    archive_path = Path(archive_text)
    if archive_path.suffix.lower() != ".zip" or not member_name:
        return None
    return archive_path, member_name


@contextmanager
def local_audio_path(path: str | Path) -> Iterator[Path]:
    """Yield a normal filesystem path, materializing zip members into a temp file."""
    archive_member = split_archive_member_path(path)
    if archive_member is None:
        yield Path(path)
        return

    archive_path, member_name = archive_member
    suffix = Path(member_name).suffix or ".audio"
    with zipfile.ZipFile(archive_path) as zf:
        data = zf.read(member_name)
    with tempfile.TemporaryDirectory(prefix="covid_audio_btp_audio_") as tmp_dir:
        local_path = Path(tmp_dir) / f"archive_member{suffix}"
        local_path.write_bytes(data)
        yield local_path


def load_audio(path: str | Path, config: AudioConfig = AUDIO_CONFIG) -> tuple[np.ndarray, int, int]:
    """Load audio as mono float32 and resample to the project sample rate.

    Returns:
        y: mono waveform at config.sample_rate
        sample_rate: output sample rate
        original_sample_rate: source sample rate before resampling
    """
    with local_audio_path(path) as resolved_path:
        if resolved_path.name.startswith("._"):
            real_path = resolved_path.parent / resolved_path.name[2:]
            if real_path.exists():
                resolved_path = real_path
            else:
                raise FileNotFoundError(
                    f"Apple Double stub found but real file missing: {real_path}"
                )
        try:
            y, original_sr = sf.read(resolved_path, always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            y = y.astype(np.float32, copy=False)
        except Exception:
            y, original_sr = librosa.load(resolved_path, sr=None, mono=True)
            y = y.astype(np.float32, copy=False)

    if original_sr != config.sample_rate:
        y = librosa.resample(y, orig_sr=original_sr, target_sr=config.sample_rate)

    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return y.astype(np.float32, copy=False), config.sample_rate, int(original_sr)


def peak_normalize(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < eps:
        return y.astype(np.float32, copy=False)
    return (y / peak).astype(np.float32, copy=False)


def fixed_length_samples(modality: str, sample_rate: int, config: AudioConfig = AUDIO_CONFIG) -> int:
    seconds = {
        "cough": config.cough_fixed_seconds,
        "breath": config.breath_fixed_seconds,
        "speech": config.speech_fixed_seconds,
    }.get(modality, config.speech_fixed_seconds)
    return int(round(seconds * sample_rate))


def pad_or_crop_center(y: np.ndarray, target_samples: int) -> np.ndarray:
    if y.size == target_samples:
        return y.astype(np.float32, copy=False)
    if y.size < target_samples:
        left = (target_samples - y.size) // 2
        right = target_samples - y.size - left
        return np.pad(y, (left, right), mode="constant").astype(np.float32, copy=False)
    start = (y.size - target_samples) // 2
    return y[start : start + target_samples].astype(np.float32, copy=False)

