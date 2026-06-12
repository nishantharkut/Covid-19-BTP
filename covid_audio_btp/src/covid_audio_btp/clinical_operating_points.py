from __future__ import annotations

import numpy as np
import pandas as pd

from covid_audio_btp.labels import label_to_int


DEFAULT_TARGET_SPECIFICITIES = [0.80, 0.90, 0.95]
DEFAULT_TARGET_SENSITIVITIES = [0.80, 0.90]


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
    return frame


def _candidate_thresholds(probabilities: pd.Series) -> np.ndarray:
    values = np.unique(np.clip(probabilities.astype(float).to_numpy(), 0.0, 1.0))
    return np.unique(np.concatenate(([0.0, 0.5, 1.0], values)))


def _point_metrics(
    frame: pd.DataFrame,
    threshold: float,
    probability_column: str,
    label_column: str,
) -> dict[str, float]:
    y_true = frame[label_column].map(label_to_int).to_numpy(dtype=int)
    y_prob = frame[probability_column].astype(float).to_numpy()
    y_pred = (y_prob >= threshold).astype(int)
    positive = y_true == 1
    negative = y_true == 0
    predicted_positive = y_pred == 1
    predicted_negative = y_pred == 0

    tp = int(np.sum(positive & predicted_positive))
    tn = int(np.sum(negative & predicted_negative))
    fp = int(np.sum(negative & predicted_positive))
    fn = int(np.sum(positive & predicted_negative))

    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    npv = tn / max(1, tn + fn)
    f1 = 2 * precision * sensitivity / max(1e-12, precision + sensitivity)
    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "f1": float(f1),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_samples": int(len(frame)),
        "n_positive": int(np.sum(positive)),
        "n_negative": int(np.sum(negative)),
    }


def _threshold_metric_table(
    frame: pd.DataFrame,
    probability_column: str,
    label_column: str,
) -> pd.DataFrame:
    y_true = frame[label_column].map(label_to_int).to_numpy(dtype=int)
    y_prob = np.clip(frame[probability_column].astype(float).to_numpy(), 0.0, 1.0)
    thresholds = _candidate_thresholds(pd.Series(y_prob))

    order = np.argsort(y_prob, kind="mergesort")
    sorted_prob = y_prob[order]
    sorted_true = y_true[order]
    sorted_positive = (sorted_true == 1).astype(int)
    sorted_negative = (sorted_true == 0).astype(int)
    cumulative_positive = np.concatenate(([0], np.cumsum(sorted_positive)))
    cumulative_negative = np.concatenate(([0], np.cumsum(sorted_negative)))

    predicted_negative_count = np.searchsorted(sorted_prob, thresholds, side="left")
    fn = cumulative_positive[predicted_negative_count]
    tn = cumulative_negative[predicted_negative_count]
    total_positive = int(np.sum(y_true == 1))
    total_negative = int(np.sum(y_true == 0))
    tp = total_positive - fn
    fp = total_negative - tn

    sensitivity = tp / max(1, total_positive)
    specificity = tn / max(1, total_negative)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    npv = np.divide(tn, tn + fn, out=np.zeros_like(tn, dtype=float), where=(tn + fn) > 0)
    f1 = np.divide(
        2 * precision * sensitivity,
        precision + sensitivity,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + sensitivity) > 0,
    )
    return pd.DataFrame(
        {
            "threshold": thresholds.astype(float),
            "sensitivity": sensitivity.astype(float),
            "specificity": specificity.astype(float),
            "precision": precision.astype(float),
            "npv": npv.astype(float),
            "f1": f1.astype(float),
            "balanced_accuracy": ((sensitivity + specificity) / 2.0).astype(float),
            "tp": tp.astype(int),
            "fp": fp.astype(int),
            "tn": tn.astype(int),
            "fn": fn.astype(int),
            "n_samples": int(len(frame)),
            "n_positive": total_positive,
            "n_negative": total_negative,
        }
    )


def operating_point_at_specificity(
    predictions: pd.DataFrame,
    target_specificity: float,
    probability_column: str = "probability",
    label_column: str = "label_binary",
) -> dict[str, object]:
    frame = _clean_labeled_predictions(predictions, probability_column, label_column)
    if frame.empty or frame[label_column].nunique() < 2:
        return {}
    table = _threshold_metric_table(frame, probability_column, label_column)
    feasible = table[table["specificity"] >= target_specificity]
    if feasible.empty:
        feasible = table
    best = feasible.sort_values(
        ["sensitivity", "specificity", "threshold"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    best["operating_constraint"] = f"specificity>={target_specificity:.3f}"
    best["target_specificity"] = float(target_specificity)
    best["target_sensitivity"] = np.nan
    return best


def operating_point_at_sensitivity(
    predictions: pd.DataFrame,
    target_sensitivity: float,
    probability_column: str = "probability",
    label_column: str = "label_binary",
) -> dict[str, object]:
    frame = _clean_labeled_predictions(predictions, probability_column, label_column)
    if frame.empty or frame[label_column].nunique() < 2:
        return {}
    table = _threshold_metric_table(frame, probability_column, label_column)
    feasible = table[table["sensitivity"] >= target_sensitivity]
    if feasible.empty:
        feasible = table
    best = feasible.sort_values(
        ["specificity", "sensitivity", "threshold"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    best["operating_constraint"] = f"sensitivity>={target_sensitivity:.3f}"
    best["target_specificity"] = np.nan
    best["target_sensitivity"] = float(target_sensitivity)
    return best


def build_clinical_operating_points(
    predictions: pd.DataFrame,
    group_columns: list[str] | None = None,
    probability_column: str = "probability",
    label_column: str = "label_binary",
    target_specificities: list[float] | None = None,
    target_sensitivities: list[float] | None = None,
) -> pd.DataFrame:
    frame = _clean_labeled_predictions(predictions, probability_column, label_column)
    if frame.empty:
        return pd.DataFrame()
    group_columns = [col for col in (group_columns or []) if col in frame.columns]
    target_specificities = target_specificities if target_specificities is not None else DEFAULT_TARGET_SPECIFICITIES
    target_sensitivities = target_sensitivities if target_sensitivities is not None else DEFAULT_TARGET_SENSITIVITIES
    iterator = frame.groupby(group_columns, dropna=False) if group_columns else [((), frame)]
    rows: list[dict[str, object]] = []
    for group_key, group in iterator:
        if group[label_column].nunique() < 2:
            continue
        group_values: dict[str, object] = {}
        if group_columns:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = dict(zip(group_columns, group_key))
        for target in target_specificities:
            row = operating_point_at_specificity(group, float(target), probability_column, label_column)
            if row:
                rows.append({**group_values, **row})
        for target in target_sensitivities:
            row = operating_point_at_sensitivity(group, float(target), probability_column, label_column)
            if row:
                rows.append({**group_values, **row})
    return pd.DataFrame(rows)
