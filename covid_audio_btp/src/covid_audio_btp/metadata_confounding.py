from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from covid_audio_btp.metadata_baseline import _parse_json_dict
from covid_audio_btp.metrics import best_threshold_by_balanced_accuracy, binary_metric_bundle, labels_to_binary


DIRECT_LABEL_COLUMNS = {"label_binary", "label_raw", "label_group", "label", "covid_status", "status"}

FEATURE_SETS: dict[str, list[str]] = {
    "symptoms_only": ["symptoms_json"],
    "demographic_protocol_only": [
        "age",
        "gender",
        "country",
        "recording_date",
        "duration_sec",
        "sample_rate_original",
        "quality_flag",
    ],
    "full_safe_metadata": [
        "age",
        "gender",
        "country",
        "recording_date",
        "symptoms_json",
        "comorbidities_json",
        "duration_sec",
        "sample_rate_original",
        "quality_flag",
    ],
}


@dataclass
class MetadataConfoundingAuditResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    group_summary: pd.DataFrame


def feature_group_for_column(column: str) -> str:
    if column.startswith("symptoms_json_"):
        return "symptom_or_label_proxy"
    if column.startswith("comorbidities_json_"):
        return "comorbidity_proxy"
    if column == "age" or column.startswith("gender_") or column.startswith("country_"):
        return "demographic"
    if (
        column in {"recording_year", "recording_month", "duration_sec", "sample_rate_original"}
        or column.startswith("quality_flag_")
    ):
        return "recording_protocol"
    return "other_metadata"


def _coerce_metadata_value(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.number)):
        if not np.isfinite(float(value)):
            return 0.0
        return float(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return 1.0
    if text in {"false", "0", "no", "n", "nan", "none", ""}:
        return 0.0
    return 1.0


def _selected_raw_columns(feature_set: str, feature_columns: list[str] | None = None) -> list[str]:
    columns = feature_columns if feature_columns is not None else FEATURE_SETS.get(feature_set)
    if columns is None:
        raise ValueError(f"Unknown metadata confounding feature set: {feature_set}")
    return [col for col in columns if col not in DIRECT_LABEL_COLUMNS]


def build_audit_feature_frame(
    metadata: pd.DataFrame,
    feature_set: str = "full_safe_metadata",
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    raw_columns = _selected_raw_columns(feature_set, feature_columns=feature_columns)
    base = pd.DataFrame(index=metadata.index)

    for col in raw_columns:
        if col not in metadata.columns:
            continue
        if col.endswith("_json"):
            parsed = metadata[col].map(_parse_json_dict)
            keys = sorted({str(key) for item in parsed for key in item.keys()})
            for key in keys:
                feature_name = f"{col}_{key}"
                base[feature_name] = parsed.map(lambda item, k=key: _coerce_metadata_value(item.get(k, False)))
            continue
        if col == "recording_date":
            date = pd.to_datetime(metadata[col], errors="coerce")
            base["recording_year"] = date.dt.year.fillna(0).astype(float)
            base["recording_month"] = date.dt.month.fillna(0).astype(float)
            continue
        base[col] = metadata[col]

    if base.empty:
        return pd.DataFrame(index=metadata.index), {}

    frames: list[pd.DataFrame] = []
    groups: dict[str, str] = {}
    numeric_parts: dict[str, pd.Series] = {}
    categorical_cols: list[str] = []

    for col in base.columns:
        series = base[col]
        if pd.api.types.is_bool_dtype(series):
            numeric_parts[col] = series.fillna(False).astype(float)
            groups[col] = feature_group_for_column(col)
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric_parts[col] = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
            groups[col] = feature_group_for_column(col)
            continue
        converted = pd.to_numeric(series, errors="coerce")
        present = series.notna() & series.astype(str).str.strip().ne("")
        if bool(present.any()) and bool(converted[present].notna().all()):
            numeric_parts[col] = converted.fillna(0.0).astype(float)
            groups[col] = feature_group_for_column(col)
        else:
            categorical_cols.append(col)

    if numeric_parts:
        frames.append(pd.DataFrame(numeric_parts, index=base.index))
    if categorical_cols:
        dummies = pd.get_dummies(base[categorical_cols].fillna("unknown").astype(str), dummy_na=False)
        for col in dummies.columns:
            groups[col] = feature_group_for_column(col)
        frames.append(dummies)

    if not frames:
        return pd.DataFrame(index=metadata.index), {}
    features = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return features, groups


def _align_feature_frames(frames: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[str]]:
    columns = sorted(set().union(*(frame.columns for frame in frames)))
    aligned = [frame.reindex(columns=columns, fill_value=0.0) for frame in frames]
    return aligned, columns


def _prediction_frame(source: pd.DataFrame, probabilities: np.ndarray, audit_model: str, split: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recording_id": source["recording_id"].to_numpy(),
            "participant_id": source["participant_id"].to_numpy(),
            "dataset": source.get("dataset", pd.Series([""] * len(source), index=source.index)).to_numpy(),
            "label_binary": source["label_binary"].to_numpy(),
            "split": split,
            "model_name": "metadata_confounding_logistic_regression",
            "audit_model": audit_model,
            "feature_strategy": audit_model,
            "probability": probabilities,
        }
    )


def _fit_one_feature_set(metadata: pd.DataFrame, feature_set: str, random_state: int) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    df = metadata[
        metadata["label_binary"].isin(["positive", "negative"])
        & metadata["split"].isin(["train", "validation", "test"])
    ].copy()
    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Need train/validation/test rows for metadata confounding audit")

    train_raw, groups = build_audit_feature_frame(train, feature_set=feature_set)
    validation_raw, validation_groups = build_audit_feature_frame(validation, feature_set=feature_set)
    test_raw, test_groups = build_audit_feature_frame(test, feature_set=feature_set)
    groups = {**validation_groups, **test_groups, **groups}
    (train_x, validation_x, test_x), columns = _align_feature_frames([train_raw, validation_raw, test_raw])
    varying_columns = [col for col in columns if train_x[col].nunique(dropna=False) > 1]
    if not varying_columns:
        raise ValueError(f"Feature set {feature_set} selected no train-varying columns")
    train_x = train_x[varying_columns]
    validation_x = validation_x[varying_columns]
    test_x = test_x[varying_columns]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=3000, random_state=random_state),
            ),
        ]
    )
    model.fit(train_x, labels_to_binary(train["label_binary"]))
    validation_prob = model.predict_proba(validation_x)[:, 1]
    threshold = best_threshold_by_balanced_accuracy(labels_to_binary(validation["label_binary"]), validation_prob)
    test_prob = model.predict_proba(test_x)[:, 1]
    metrics = binary_metric_bundle(labels_to_binary(test["label_binary"]), test_prob, threshold=threshold)
    dataset = str(test["dataset"].mode().iloc[0]) if "dataset" in test.columns else "coswara"
    metrics.update(
        {
            "model_name": "metadata_confounding_logistic_regression",
            "modality": "metadata",
            "audit_model": feature_set,
            "feature_strategy": feature_set,
            "dataset": dataset,
            "calibration_method": "none",
            "train_rows": float(len(train)),
            "validation_rows": float(len(validation)),
            "test_rows": float(len(test)),
            "n_features": float(len(varying_columns)),
            "test_positive_prevalence": float(labels_to_binary(test["label_binary"]).mean()),
        }
    )

    coefs = model.named_steps["classifier"].coef_[0]
    importance = pd.DataFrame(
        {
            "audit_model": feature_set,
            "feature": varying_columns,
            "feature_group": [groups.get(col, feature_group_for_column(col)) for col in varying_columns],
            "coefficient": coefs,
        }
    )
    importance["importance_abs"] = importance["coefficient"].abs()
    importance["direction"] = np.where(importance["coefficient"] >= 0, "positive_label", "negative_label")
    importance = importance.sort_values(["importance_abs", "feature"], ascending=[False, True]).reset_index(drop=True)
    predictions = _prediction_frame(test, test_prob, audit_model=feature_set, split="test")
    predictions["threshold"] = threshold
    return metrics, predictions, importance


def _summarize_feature_groups(feature_importance: pd.DataFrame) -> pd.DataFrame:
    if feature_importance.empty:
        return pd.DataFrame()
    grouped = (
        feature_importance.groupby(["audit_model", "feature_group"], dropna=False)
        .agg(
            n_features=("feature", "count"),
            importance_abs_sum=("importance_abs", "sum"),
            importance_abs_mean=("importance_abs", "mean"),
            importance_abs_max=("importance_abs", "max"),
        )
        .reset_index()
    )
    totals = grouped.groupby("audit_model")["importance_abs_sum"].transform("sum").replace(0, np.nan)
    grouped["importance_share"] = (grouped["importance_abs_sum"] / totals).fillna(0.0)
    top_rows = (
        feature_importance.sort_values(["audit_model", "feature_group", "importance_abs"], ascending=[True, True, False])
        .groupby(["audit_model", "feature_group"], dropna=False)
        .head(1)[["audit_model", "feature_group", "feature", "coefficient", "importance_abs"]]
        .rename(
            columns={
                "feature": "top_feature",
                "coefficient": "top_coefficient",
                "importance_abs": "top_importance_abs",
            }
        )
    )
    return grouped.merge(top_rows, on=["audit_model", "feature_group"], how="left").sort_values(
        ["audit_model", "importance_abs_sum"], ascending=[True, False]
    )


def run_metadata_confounding_audit(
    metadata: pd.DataFrame,
    feature_sets: list[str] | None = None,
    random_state: int = 42,
) -> MetadataConfoundingAuditResult:
    feature_sets = feature_sets or list(FEATURE_SETS)
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for feature_set in feature_sets:
        metrics, predictions, importance = _fit_one_feature_set(metadata, feature_set=feature_set, random_state=random_state)
        metric_rows.append(metrics)
        prediction_frames.append(predictions)
        importance_frames.append(importance)

    feature_importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return MetadataConfoundingAuditResult(
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
        feature_importance=feature_importance,
        group_summary=_summarize_feature_groups(feature_importance),
    )
