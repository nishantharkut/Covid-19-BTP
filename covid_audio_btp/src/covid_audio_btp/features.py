from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from covid_audio_btp.audio_io import load_audio
from covid_audio_btp.preprocess import preprocess_for_features


def _stat_features(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
    }


def extract_mfcc_features(y: np.ndarray, sample_rate: int, n_mfcc: int = 40) -> dict[str, float]:
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features: dict[str, float] = {}
    for matrix_name, matrix in (("mfcc", mfcc), ("delta_mfcc", delta), ("delta2_mfcc", delta2)):
        for idx in range(matrix.shape[0]):
            features.update(_stat_features(f"{matrix_name}_{idx + 1}", matrix[idx]))

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_flatness_mean"] = float(np.mean(flatness))
    features["duration_sec"] = float(y.size / sample_rate)
    return features


def extract_features_for_row(row: pd.Series) -> dict[str, object]:
    y, sample_rate, _ = load_audio(Path(row["audio_path"]))
    processed, event_info = preprocess_for_features(y, sample_rate, str(row.get("modality", "unknown")))
    features = extract_mfcc_features(processed, sample_rate=sample_rate, n_mfcc=40)
    return {
        "recording_id": row["recording_id"],
        "participant_id": row["participant_id"],
        "dataset": row.get("dataset", "coswara"),
        "modality": row.get("modality", "unknown"),
        "submodality": row.get("submodality", "unknown"),
        "label_binary": row.get("label_binary", "unknown"),
        "split": row.get("split", "unused"),
        **event_info,
        **features,
    }


def extract_feature_table(metadata: pd.DataFrame, quality_mode: str = "all_samples") -> pd.DataFrame:
    df = metadata.copy()
    df = df[df["label_binary"].isin(["positive", "negative"])]
    if quality_mode == "quality_ok_only" and "quality_flag" in df.columns:
        df = df[df["quality_flag"] == "ok"]
    elif quality_mode not in {"all_samples", "quality_ok_only"}:
        raise ValueError(f"Unknown quality_mode: {quality_mode}")
    rows = []
    for _, row in df.iterrows():
        try:
            rows.append(extract_features_for_row(row))
        except Exception as e:
            print(f"WARNING: skipping {row.get('audio_path', '?')} — {e}")
    return pd.DataFrame(rows)


def feature_columns(feature_df: pd.DataFrame) -> list[str]:
    excluded = {
        "recording_id",
        "participant_id",
        "dataset",
        "modality",
        "submodality",
        "label_binary",
        "split",
        "segmentation_method",
    }
    return [
        col
        for col in feature_df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(feature_df[col])
    ]

