from __future__ import annotations

import numpy as np
import pandas as pd


def _pivot_predictions(predictions: pd.DataFrame, probability_column: str) -> pd.DataFrame:
    required = {"participant_id", "modality", probability_column, "label_binary", "split"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    pivot = predictions.pivot_table(
        index=["participant_id", "label_binary", "split"],
        columns="modality",
        values=probability_column,
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def _available_modalities(row: pd.Series, modality_cols: list[str]) -> str:
    return ",".join(sorted([col for col in modality_cols if pd.notna(row.get(col))]))


def uniform_fusion(predictions: pd.DataFrame, probability_column: str = "probability") -> pd.DataFrame:
    pivot = _pivot_predictions(predictions, probability_column)
    modality_cols = [col for col in ("cough", "breath", "speech") if col in pivot.columns]
    if not modality_cols:
        raise ValueError("No modality columns available for fusion")
    pivot["probability"] = pivot[modality_cols].mean(axis=1, skipna=True)
    pivot["fusion_method"] = "uniform_mean"
    pivot["available_modalities"] = pivot[modality_cols].apply(lambda row: _available_modalities(row, modality_cols), axis=1)
    return pivot[["participant_id", "label_binary", "split", "probability", "fusion_method", "available_modalities"]]


def validation_weighted_fusion(
    predictions: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    probability_column: str = "probability",
    metric_column: str = "auprc",
) -> pd.DataFrame:
    pivot = _pivot_predictions(predictions, probability_column)
    modality_cols = [col for col in ("cough", "breath", "speech") if col in pivot.columns]
    if metric_column in validation_metrics.columns and "modality" in validation_metrics.columns:
        metric_map = validation_metrics.groupby("modality")[metric_column].max().to_dict()
    else:
        metric_map = {}
    raw_weights = np.array([max(float(metric_map.get(col, 0.5)) - 0.5, 0.0) for col in modality_cols])
    if raw_weights.sum() <= 0:
        raw_weights = np.ones(len(modality_cols))
    weights = raw_weights / raw_weights.sum()

    probs = pivot[modality_cols].to_numpy(dtype=float)
    mask = np.isfinite(probs)
    weighted = np.where(mask, probs * weights, 0.0)
    denom = np.where(mask, weights, 0.0).sum(axis=1)
    pivot["probability"] = np.divide(weighted.sum(axis=1), denom, out=np.full(len(pivot), np.nan), where=denom > 0)
    pivot["fusion_method"] = f"validation_weighted_{metric_column}"
    pivot["available_modalities"] = pivot[modality_cols].apply(lambda row: _available_modalities(row, modality_cols), axis=1)
    return pivot[["participant_id", "label_binary", "split", "probability", "fusion_method", "available_modalities"]]


def quality_weight(flag: object) -> float:
    text = str(flag).strip().lower()
    if text in {"ok", "good", "not_audited", "unknown", ""}:
        return 1.0
    if text in {"uncertain", "low_quality"}:
        return 0.50
    if text in {"short", "mostly_silence", "clipped", "bad"}:
        return 0.25
    if text in {"corrupt", "missing", "unreadable"}:
        return 0.05
    return 0.50


def quality_weighted_fusion(
    predictions: pd.DataFrame,
    quality: pd.DataFrame,
    validation_metrics: pd.DataFrame | None = None,
    probability_column: str = "probability",
    metric_column: str = "auprc",
) -> pd.DataFrame:
    """Fuse calibrated branch probabilities with validation and audio-quality weights.

    Weight per branch = max(validation_metric - 0.5, floor) * quality_weight.
    Missing modalities are ignored and weights are renormalized per participant.
    """
    required = {"participant_id", "recording_id", "modality", "label_binary", "split", probability_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    q = quality[[c for c in ["recording_id", "quality_flag"] if c in quality.columns]].drop_duplicates("recording_id")
    merged = predictions.merge(q, on="recording_id", how="left")
    merged["quality_flag"] = merged.get("quality_flag", "unknown")
    metric_map: dict[str, float] = {}
    if validation_metrics is not None and not validation_metrics.empty and metric_column in validation_metrics.columns:
        metric_map = validation_metrics.groupby("modality")[metric_column].max().to_dict()

    rows: list[dict[str, object]] = []
    group_cols = ["participant_id", "label_binary", "split"]
    for key, group in merged.groupby(group_cols, dropna=False):
        participant_id, label_binary, split = key
        weighted_sum = 0.0
        denom = 0.0
        modalities: list[str] = []
        for _, row in group.iterrows():
            modality = str(row["modality"])
            prob = float(row[probability_column])
            validation_weight = max(float(metric_map.get(modality, 0.75)) - 0.5, 0.05)
            branch_weight = validation_weight * quality_weight(row.get("quality_flag", "unknown"))
            if not np.isfinite(prob) or branch_weight <= 0:
                continue
            weighted_sum += prob * branch_weight
            denom += branch_weight
            modalities.append(modality)
        probability = weighted_sum / denom if denom > 0 else float("nan")
        rows.append(
            {
                "participant_id": participant_id,
                "label_binary": label_binary,
                "split": split,
                "probability": probability,
                "fusion_method": f"quality_weighted_{metric_column}",
                "available_modalities": ",".join(sorted(set(modalities))),
            }
        )
    return pd.DataFrame(rows)
