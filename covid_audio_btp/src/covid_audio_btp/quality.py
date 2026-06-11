from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from covid_audio_btp.audio_io import load_audio
from covid_audio_btp.config import QUALITY_CONFIG, QualityConfig
from covid_audio_btp.preprocess import preprocess_for_features, trim_silence
from covid_audio_btp.schemas import QUALITY_COLUMNS


def assign_quality_flag(
    modality: str,
    duration_sec: float,
    silence_ratio: float,
    clipping_ratio: float,
    config: QualityConfig = QUALITY_CONFIG,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    min_duration = {
        "cough": config.min_cough_seconds,
        "breath": config.min_breath_seconds,
        "speech": config.min_speech_seconds,
    }.get(modality, config.min_speech_seconds)

    if duration_sec < min_duration:
        reasons.append("duration")
    if silence_ratio > config.max_silence_ratio:
        reasons.append("silence")
    if clipping_ratio > config.max_clipping_ratio:
        reasons.append("clipping")

    if not reasons:
        return "ok", []
    if "duration" in reasons:
        return "short", reasons
    if "silence" in reasons:
        return "mostly_silence", reasons
    if "clipping" in reasons:
        return "clipped", reasons
    return "low_quality", reasons


def compute_silence_ratio(y: np.ndarray, top_db: float = QUALITY_CONFIG.silence_db_threshold) -> float:
    if y.size == 0:
        return 1.0
    trimmed, index = trim_silence(y, top_db=int(top_db))
    active = max(0, index[1] - index[0])
    return float(1.0 - active / max(1, y.size))


def compute_clipping_ratio(y: np.ndarray, threshold: float = 0.999) -> float:
    if y.size == 0:
        return 0.0
    return float(np.mean(np.abs(y) >= threshold))


def estimate_snr_proxy(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    rms = librosa.feature.rms(y=y)[0]
    if rms.size < 2:
        return 0.0
    signal = float(np.percentile(rms, 90))
    noise = float(np.percentile(rms, 10)) + 1e-8
    return float(20.0 * np.log10(max(signal, 1e-8) / noise))


def quality_for_file(row: pd.Series) -> dict[str, object]:
    path = Path(row["audio_path"])
    modality = str(row.get("modality", "unknown"))
    try:
        y, sample_rate, original_sr = load_audio(path)
        duration_sec = float(y.size / sample_rate)
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)[0]
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        silence_ratio = compute_silence_ratio(y)
        clipping_ratio = compute_clipping_ratio(y)
        _, event_info = preprocess_for_features(y, sample_rate, modality)
        flag, reasons = assign_quality_flag(modality, duration_sec, silence_ratio, clipping_ratio)
        return {
            "recording_id": row["recording_id"],
            "audio_path": row["audio_path"],
            "duration_sec": duration_sec,
            "sample_rate_original": original_sr,
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "zero_crossing_rate_mean": float(np.mean(zcr)),
            "silence_ratio": silence_ratio,
            "clipping_ratio": clipping_ratio,
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_flatness_mean": float(np.mean(flatness)),
            "snr_proxy": estimate_snr_proxy(y),
            "event_start_sec": event_info["event_start_sec"],
            "event_end_sec": event_info["event_end_sec"],
            "event_duration_sec": event_info["event_duration_sec"],
            "active_audio_ratio": event_info["active_audio_ratio"],
            "quality_flag": flag,
            "quality_reasons": ";".join(reasons),
        }
    except Exception as exc:
        return {
            "recording_id": row.get("recording_id", ""),
            "audio_path": row.get("audio_path", ""),
            "duration_sec": 0.0,
            "sample_rate_original": "",
            "rms_mean": 0.0,
            "rms_std": 0.0,
            "zero_crossing_rate_mean": 0.0,
            "silence_ratio": 1.0,
            "clipping_ratio": 0.0,
            "spectral_centroid_mean": 0.0,
            "spectral_flatness_mean": 0.0,
            "snr_proxy": 0.0,
            "event_start_sec": 0.0,
            "event_end_sec": 0.0,
            "event_duration_sec": 0.0,
            "active_audio_ratio": 0.0,
            "quality_flag": "corrupt",
            "quality_reasons": f"{type(exc).__name__}: {exc}",
        }


def run_quality_audit(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = [quality_for_file(row) for _, row in metadata.iterrows()]
    return pd.DataFrame(rows, columns=QUALITY_COLUMNS)


def attach_quality_flags(metadata: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    df = metadata.copy()
    quality_cols = quality[
        ["recording_id", "duration_sec", "sample_rate_original", "quality_flag"]
    ].drop_duplicates("recording_id")
    df = df.drop(columns=["duration_sec", "sample_rate_original", "quality_flag"], errors="ignore")
    return df.merge(quality_cols, on="recording_id", how="left")


def quality_summary(quality: pd.DataFrame) -> pd.DataFrame:
    if quality.empty:
        return pd.DataFrame(columns=["quality_flag", "n_recordings"])
    return (
        quality.groupby("quality_flag", dropna=False)
        .agg(
            n_recordings=("recording_id", "nunique"),
            median_duration_sec=("duration_sec", "median"),
            median_silence_ratio=("silence_ratio", "median"),
            median_clipping_ratio=("clipping_ratio", "median"),
        )
        .reset_index()
        .sort_values("n_recordings", ascending=False)
    )

