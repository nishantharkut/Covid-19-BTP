from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from covid_audio_btp.labels import label_to_int


DEFAULT_CALIBRATION_BINS = 10


def _clean_labeled_predictions(
    predictions: pd.DataFrame,
    probability_column: str,
    label_column: str,
) -> pd.DataFrame:
    required = {probability_column, label_column}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()
    frame = predictions[predictions[label_column].isin(["positive", "negative"])].copy()
    if frame.empty:
        return frame
    frame[probability_column] = pd.to_numeric(frame[probability_column], errors="coerce")
    frame = frame[np.isfinite(frame[probability_column])]
    frame[probability_column] = frame[probability_column].clip(0.0, 1.0)
    return frame


def _binary_labels(frame: pd.DataFrame, label_column: str) -> np.ndarray:
    return frame[label_column].map(label_to_int).to_numpy(dtype=int)


def calibration_bin_table(
    predictions: pd.DataFrame,
    probability_column: str = "probability",
    label_column: str = "label_binary",
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> pd.DataFrame:
    frame = _clean_labeled_predictions(predictions, probability_column, label_column)
    if frame.empty:
        return pd.DataFrame()
    n_bins = max(1, int(n_bins))
    y_true = _binary_labels(frame, label_column)
    y_prob = frame[probability_column].to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.searchsorted(edges, y_prob, side="right") - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)

    rows: list[dict[str, object]] = []
    for idx in range(n_bins):
        mask = bin_index == idx
        if not np.any(mask):
            continue
        mean_probability = float(np.mean(y_prob[mask]))
        observed_rate = float(np.mean(y_true[mask]))
        gap = mean_probability - observed_rate
        rows.append(
            {
                "bin_index": idx,
                "probability_lower": float(edges[idx]),
                "probability_upper": float(edges[idx + 1]),
                "n_samples": int(np.sum(mask)),
                "n_positive": int(np.sum(y_true[mask] == 1)),
                "n_negative": int(np.sum(y_true[mask] == 0)),
                "mean_probability": mean_probability,
                "observed_positive_rate": observed_rate,
                "calibration_gap": float(gap),
                "abs_calibration_gap": float(abs(gap)),
            }
        )
    return pd.DataFrame(rows)


def calibration_summary(
    predictions: pd.DataFrame,
    probability_column: str = "probability",
    label_column: str = "label_binary",
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, float]:
    frame = _clean_labeled_predictions(predictions, probability_column, label_column)
    if frame.empty:
        return {}
    y_true = _binary_labels(frame, label_column)
    y_prob = frame[probability_column].to_numpy(dtype=float)
    bins = calibration_bin_table(frame, probability_column=probability_column, label_column=label_column, n_bins=n_bins)
    n_samples = int(len(frame))
    if bins.empty:
        ece = float("nan")
        mce = float("nan")
    else:
        ece = float(np.sum((bins["n_samples"].to_numpy(dtype=float) / n_samples) * bins["abs_calibration_gap"].to_numpy(dtype=float)))
        mce = float(bins["abs_calibration_gap"].max())
    clipped = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    observed_prevalence = float(np.mean(y_true))
    mean_probability = float(np.mean(y_prob))
    return {
        "n_samples": n_samples,
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "observed_prevalence": observed_prevalence,
        "mean_probability": mean_probability,
        "calibration_gap": float(mean_probability - observed_prevalence),
        "ece": ece,
        "mce": mce,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "nll": float(log_loss(y_true, clipped, labels=[0, 1])),
        "n_bins": int(n_bins),
        "non_empty_bins": int(len(bins)),
    }


def build_calibration_shift_report(
    predictions: pd.DataFrame,
    group_columns: list[str] | None = None,
    probability_column: str = "probability",
    label_column: str = "label_binary",
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _clean_labeled_predictions(predictions, probability_column, label_column)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_columns = [col for col in (group_columns or []) if col in frame.columns]
    iterator = frame.groupby(group_columns, dropna=False) if group_columns else [((), frame)]
    summary_rows: list[dict[str, object]] = []
    bin_frames: list[pd.DataFrame] = []
    for group_key, group in iterator:
        group_values: dict[str, object] = {}
        if group_columns:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = dict(zip(group_columns, group_key))
        summary = calibration_summary(group, probability_column=probability_column, label_column=label_column, n_bins=n_bins)
        if summary:
            summary_rows.append({**group_values, **summary})
        bins = calibration_bin_table(group, probability_column=probability_column, label_column=label_column, n_bins=n_bins)
        if not bins.empty:
            for col, value in group_values.items():
                bins[col] = value
            bin_frames.append(bins)
    summary_frame = pd.DataFrame(summary_rows)
    bins_frame = pd.concat(bin_frames, ignore_index=True, sort=False) if bin_frames else pd.DataFrame()
    return summary_frame, bins_frame
