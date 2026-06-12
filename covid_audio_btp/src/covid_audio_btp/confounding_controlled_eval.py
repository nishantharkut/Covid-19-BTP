from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from covid_audio_btp.metrics import labels_to_binary


DEFAULT_CONFOUNDERS = [
    "recording_date",
    "country",
    "age",
    "gender",
    "duration_sec",
    "sample_rate_original",
    "quality_flag",
]

QUALITY_SEVERITY = {
    "ok": 0,
    "good": 0,
    "not_audited": 0,
    "unknown": 1,
    "": 1,
    "uncertain": 2,
    "low_quality": 2,
    "short": 3,
    "mostly_silence": 3,
    "clipped": 3,
    "bad": 3,
    "corrupt": 4,
    "missing": 4,
    "unreadable": 4,
}


@dataclass
class ConfoundingControlledResult:
    metrics: pd.DataFrame
    balance: pd.DataFrame
    weights: pd.DataFrame
    merged_predictions: pd.DataFrame


def _first_non_empty(values: pd.Series) -> object:
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[~cleaned.str.lower().isin({"", "nan", "none", "unknown"})]
    if cleaned.empty:
        return "unknown"
    modes = cleaned.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else cleaned.iloc[0]


def _worst_quality_flag(values: pd.Series) -> str:
    cleaned = values.dropna().astype(str).str.strip().str.lower()
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return "unknown"
    return max(cleaned, key=lambda flag: QUALITY_SEVERITY.get(flag, 2))


def merge_predictions_with_confounders(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "recording_id",
        "participant_id",
        "age",
        "gender",
        "country",
        "recording_date",
        "duration_sec",
        "sample_rate_original",
        "quality_flag",
        "symptoms_json",
        "comorbidities_json",
    ]
    available = [col for col in metadata_columns if col in metadata.columns]
    if "recording_id" in predictions.columns and "recording_id" in metadata.columns:
        right = metadata[available].drop_duplicates("recording_id", keep="first")
        return predictions.merge(right, on="recording_id", how="left")

    if "participant_id" not in predictions.columns or "participant_id" not in metadata.columns:
        return predictions.copy()

    aggregations = {}
    for col in ["age", "gender", "country", "recording_date", "sample_rate_original", "symptoms_json", "comorbidities_json"]:
        if col in metadata.columns:
            aggregations[col] = _first_non_empty
    if "duration_sec" in metadata.columns:
        aggregations["duration_sec"] = "mean"
    if "quality_flag" in metadata.columns:
        aggregations["quality_flag"] = _worst_quality_flag

    if aggregations:
        right = metadata.groupby("participant_id", as_index=False).agg(aggregations)
    else:
        right = metadata[["participant_id"]].drop_duplicates("participant_id")
    return predictions.merge(right, on="participant_id", how="left")


def _parse_json_dict(value: object) -> dict[str, object]:
    if value is None or pd.isna(value):
        return {}
    if isinstance(value, dict):
        return value
    try:
        out = json.loads(str(value))
    except Exception:
        return {}
    return out if isinstance(out, dict) else {}


def _metadata_truthy(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _json_burden(series: pd.Series) -> pd.Series:
    return series.map(lambda value: float(sum(_metadata_truthy(v) for v in _parse_json_dict(value).values())))


def build_confounder_matrix(frame: pd.DataFrame, covariates: list[str] | None = None) -> pd.DataFrame:
    covariates = covariates or DEFAULT_CONFOUNDERS
    base = pd.DataFrame(index=frame.index)
    for covariate in covariates:
        if covariate == "recording_date" and covariate in frame.columns:
            date = pd.to_datetime(frame[covariate], errors="coerce")
            base["recording_year"] = date.dt.year.fillna(0).astype(float)
            base["recording_month"] = date.dt.month.fillna(0).astype(float)
        elif covariate == "symptom_burden" and "symptoms_json" in frame.columns:
            base["symptom_burden"] = _json_burden(frame["symptoms_json"])
        elif covariate == "comorbidity_burden" and "comorbidities_json" in frame.columns:
            base["comorbidity_burden"] = _json_burden(frame["comorbidities_json"])
        elif covariate in frame.columns:
            base[covariate] = frame[covariate]

    if base.empty:
        return pd.DataFrame(index=frame.index)

    numeric_parts: dict[str, pd.Series] = {}
    categorical_cols: list[str] = []
    for col in base.columns:
        series = base[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_parts[col] = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
            continue
        converted = pd.to_numeric(series, errors="coerce")
        present = series.notna() & series.astype(str).str.strip().ne("")
        if bool(present.any()) and bool(converted[present].notna().all()):
            numeric_parts[col] = converted.fillna(0.0).astype(float)
        else:
            categorical_cols.append(col)

    frames: list[pd.DataFrame] = []
    if numeric_parts:
        frames.append(pd.DataFrame(numeric_parts, index=base.index))
    if categorical_cols:
        frames.append(pd.get_dummies(base[categorical_cols].fillna("unknown").astype(str), dummy_na=False))
    if not frames:
        return pd.DataFrame(index=frame.index)
    return pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def _unit_id_column(predictions: pd.DataFrame) -> str:
    if "recording_id" in predictions.columns:
        return "recording_id"
    if "participant_id" in predictions.columns:
        return "participant_id"
    raise KeyError("Predictions must include recording_id or participant_id for controlled evaluation")


def _effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denom = float(np.square(weights).sum())
    if denom <= 0:
        return 0.0
    return float(np.square(weights.sum()) / denom)


def _fit_ipw_weights(
    unit_frame: pd.DataFrame,
    covariates: list[str],
    max_weight: float = 20.0,
    clip_quantile: float = 0.99,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    x = build_confounder_matrix(unit_frame, covariates=covariates)
    y = labels_to_binary(unit_frame["label_binary"])
    if x.empty or len(np.unique(y)) < 2:
        return pd.Series(np.ones(len(unit_frame)), index=unit_frame.index), pd.Series(np.full(len(unit_frame), np.nan), index=unit_frame.index), x

    varying_cols = [col for col in x.columns if x[col].nunique(dropna=False) > 1]
    x = x[varying_cols]
    if x.empty:
        return pd.Series(np.ones(len(unit_frame)), index=unit_frame.index), pd.Series(np.full(len(unit_frame), np.nan), index=unit_frame.index), x

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=random_state)),
        ]
    )
    model.fit(x, y)
    propensity = np.clip(model.predict_proba(x)[:, 1], 0.01, 0.99)
    prevalence = float(np.mean(y))
    weights = np.where(y == 1, prevalence / propensity, (1.0 - prevalence) / (1.0 - propensity))
    if 0 < clip_quantile < 1:
        upper = min(float(max_weight), float(np.quantile(weights, clip_quantile)))
    else:
        upper = float(max_weight)
    weights = np.clip(weights, 0.0, upper)
    mean_weight = float(np.mean(weights))
    if mean_weight > 0:
        weights = weights / mean_weight
    return pd.Series(weights, index=unit_frame.index), pd.Series(propensity, index=unit_frame.index), x


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 0:
        return float("nan")
    return float(np.sum(values * weights) / denom)


def _weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    if not np.isfinite(mean):
        return float("nan")
    denom = float(np.sum(weights))
    if denom <= 0:
        return float("nan")
    return float(np.sum(weights * np.square(values - mean)) / denom)


def _smd(values: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    weights = np.ones(len(values), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    pos = labels == 1
    neg = labels == 0
    if not np.any(pos) or not np.any(neg):
        return float("nan")
    pos_mean = _weighted_mean(values[pos], weights[pos])
    neg_mean = _weighted_mean(values[neg], weights[neg])
    pos_var = _weighted_var(values[pos], weights[pos])
    neg_var = _weighted_var(values[neg], weights[neg])
    pooled = np.sqrt((pos_var + neg_var) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0 if pos_mean == neg_mean else float("inf")
    return float((pos_mean - neg_mean) / pooled)


def balance_diagnostics(unit_frame: pd.DataFrame, x: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    y = labels_to_binary(unit_frame["label_binary"])
    rows: list[dict[str, object]] = []
    for feature in x.columns:
        values = x[feature].astype(float).to_numpy()
        before = _smd(values, y)
        after = _smd(values, y, weights.to_numpy(dtype=float))
        rows.append(
            {
                "feature": feature,
                "before_smd": before,
                "after_smd": after,
                "before_abs_smd": abs(before) if np.isfinite(before) else before,
                "after_abs_smd": abs(after) if np.isfinite(after) else after,
            }
        )
    return pd.DataFrame(rows).sort_values(["before_abs_smd", "feature"], ascending=[False, True]).reset_index(drop=True)


def _weighted_ece(y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return float("nan")
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lower) & (y_prob <= upper)
        if lower == 0.0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        if not np.any(mask):
            continue
        bin_weight = float(np.sum(weights[mask]))
        confidence = _weighted_mean(y_prob[mask], weights[mask])
        accuracy = _weighted_mean(y_true[mask], weights[mask])
        ece += (bin_weight / total_weight) * abs(accuracy - confidence)
    return float(ece)


def weighted_binary_metric_bundle(
    labels: pd.Series,
    probabilities: pd.Series,
    weights: pd.Series | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = labels_to_binary(labels)
    y_prob = probabilities.astype(float).to_numpy()
    sample_weight = np.ones(len(y_true), dtype=float) if weights is None else weights.astype(float).to_numpy()
    y_pred = (y_prob >= threshold).astype(int)
    metrics: dict[str, float] = {}
    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob, sample_weight=sample_weight))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob, sample_weight=sample_weight))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight))
    metrics["f1"] = float(f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1], sample_weight=sample_weight)
    tn, fp, fn, tp = cm.ravel()
    metrics["sensitivity"] = float(tp / max(1e-12, tp + fn))
    metrics["specificity"] = float(tn / max(1e-12, tn + fp))
    metrics["brier"] = float(brier_score_loss(y_true, y_prob, sample_weight=sample_weight))
    metrics["ece"] = _weighted_ece(y_true, y_prob, sample_weight)
    metrics["nll"] = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6), labels=[0, 1], sample_weight=sample_weight))
    metrics["threshold"] = float(threshold)
    metrics["n_samples"] = float(len(y_true))
    metrics["effective_sample_size"] = _effective_sample_size(sample_weight)
    return metrics



def _safe_metric_from_bundle(bundle: dict[str, float], metric: str) -> float:
    value = bundle.get(metric, float("nan"))
    try:
        return float(value)
    except Exception:
        return float("nan")


def _bootstrap_metric_values(
    group: pd.DataFrame,
    metric: str,
    control_method: str,
    n_bootstraps: int,
    threshold: float,
    rng: np.random.Generator,
) -> list[float]:
    values: list[float] = []
    n = len(group)
    if n == 0:
        return values
    for _ in range(n_bootstraps):
        sample = group.iloc[rng.integers(0, n, size=n)]
        weights = sample["ipw_weight"] if control_method == "ipw_label_propensity" else None
        bundle = weighted_binary_metric_bundle(
            sample["label_binary"],
            sample["probability"],
            weights=weights,
            threshold=threshold,
        )
        value = _safe_metric_from_bundle(bundle, metric)
        if np.isfinite(value):
            values.append(value)
    return values


def bootstrap_confounding_controlled_metrics(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    covariates: list[str] | None = None,
    group_columns: list[str] | None = None,
    metrics: list[str] | None = None,
    split: str | None = "test",
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap CIs for unweighted and IPW-controlled prediction metrics.

    Propensity weights are estimated once on the full evaluated split, then held
    fixed inside the nonparametric bootstrap. This keeps the procedure fast and
    makes the interval conditional on the fitted weighting model.
    """
    metrics = metrics or ["auroc", "auprc", "balanced_accuracy", "f1", "brier", "ece"]
    result = evaluate_confounding_controlled_predictions(
        predictions,
        metadata,
        covariates=covariates,
        group_columns=group_columns,
        split=split,
        threshold=threshold,
        random_state=random_state,
    )
    group_columns = [col for col in (group_columns or []) if col in result.merged_predictions.columns]
    rng = np.random.default_rng(random_state)
    alpha = (1.0 - confidence) / 2.0
    rows: list[dict[str, object]] = []
    for group_values, group in _iter_groups(result.merged_predictions, group_columns):
        for control_method in ["unweighted", "ipw_label_propensity"]:
            weights = group["ipw_weight"] if control_method == "ipw_label_propensity" else None
            point_bundle = weighted_binary_metric_bundle(
                group["label_binary"],
                group["probability"],
                weights=weights,
                threshold=threshold,
            )
            for metric in metrics:
                values = np.asarray(
                    _bootstrap_metric_values(
                        group,
                        metric=metric,
                        control_method=control_method,
                        n_bootstraps=n_bootstraps,
                        threshold=threshold,
                        rng=rng,
                    ),
                    dtype=float,
                )
                point = _safe_metric_from_bundle(point_bundle, metric)
                if values.size:
                    ci_low, ci_high = np.quantile(values, [alpha, 1.0 - alpha])
                    mean = float(np.mean(values))
                else:
                    ci_low = ci_high = mean = float("nan")
                row: dict[str, object] = {
                    "metric": metric,
                    "point": point,
                    "mean": mean,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "confidence": float(confidence),
                    "n_bootstraps": int(n_bootstraps),
                    "n_samples": int(len(group)),
                    "effective_sample_size": float(point_bundle["effective_sample_size"]),
                    "control_method": control_method,
                }
                row.update(group_values)
                rows.append(row)
    return pd.DataFrame(rows)


def _iter_groups(frame: pd.DataFrame, group_columns: list[str]):
    if not group_columns:
        yield {}, frame
        return
    for group_key, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        yield dict(zip(group_columns, group_key)), group


def evaluate_confounding_controlled_predictions(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    covariates: list[str] | None = None,
    group_columns: list[str] | None = None,
    split: str | None = "test",
    threshold: float = 0.5,
    random_state: int = 42,
) -> ConfoundingControlledResult:
    covariates = covariates or DEFAULT_CONFOUNDERS
    group_columns = [col for col in (group_columns or []) if col in predictions.columns]
    merged = merge_predictions_with_confounders(predictions, metadata)
    merged = merged[merged["label_binary"].isin(["positive", "negative"])].copy()
    if split is not None and "split" in merged.columns:
        merged = merged[merged["split"].astype(str) == str(split)].copy()
    if merged.empty:
        raise ValueError("No labeled prediction rows available for confounding-controlled evaluation")

    unit_col = _unit_id_column(merged)
    unit_frame = merged.drop_duplicates(unit_col, keep="first").copy()
    weights, propensity, x = _fit_ipw_weights(unit_frame, covariates=covariates, random_state=random_state)
    unit_weights = unit_frame[[unit_col, "label_binary"]].copy()
    unit_weights["propensity_score"] = propensity.to_numpy(dtype=float)
    unit_weights["ipw_weight"] = weights.to_numpy(dtype=float)
    merged = merged.merge(unit_weights[[unit_col, "propensity_score", "ipw_weight"]], on=unit_col, how="left")

    rows: list[dict[str, object]] = []
    for group_values, group in _iter_groups(merged, group_columns):
        unweighted = weighted_binary_metric_bundle(group["label_binary"], group["probability"], threshold=threshold)
        unweighted.update(group_values)
        unweighted["control_method"] = "unweighted"
        rows.append(unweighted)

        weighted = weighted_binary_metric_bundle(
            group["label_binary"],
            group["probability"],
            weights=group["ipw_weight"],
            threshold=threshold,
        )
        weighted.update(group_values)
        weighted["control_method"] = "ipw_label_propensity"
        rows.append(weighted)

    balance = balance_diagnostics(unit_frame, x, weights) if not x.empty else pd.DataFrame()
    return ConfoundingControlledResult(
        metrics=pd.DataFrame(rows),
        balance=balance,
        weights=unit_weights,
        merged_predictions=merged,
    )
