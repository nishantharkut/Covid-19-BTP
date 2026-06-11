from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from covid_audio_btp.metrics import binary_metric_bundle, labels_to_binary


@dataclass
class MetadataBaselineResult:
    model_name: str
    metrics: dict[str, float | str]
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    model: object
    feature_columns: list[str]


def _parse_json_dict(value: object) -> dict[str, object]:
    if value is None or pd.isna(value):
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def build_metadata_feature_frame(
    metadata: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or ["age", "gender", "country", "symptoms_json", "comorbidities_json", "quality_flag"]
    frames: list[pd.DataFrame] = []
    base = pd.DataFrame(index=metadata.index)
    for col in feature_columns:
        if col not in metadata.columns:
            continue
        if col.endswith("_json"):
            parsed = metadata[col].map(_parse_json_dict)
            keys = sorted({str(k) for item in parsed for k in item.keys()})
            for key in keys:
                base[f"{col}_{key}"] = parsed.map(lambda d, k=key: d.get(k, False))
        else:
            base[col] = metadata[col]
    if base.empty:
        return pd.DataFrame(index=metadata.index)
    numeric_parts: dict[str, pd.Series] = {}
    categorical_cols: list[str] = []
    for col in base.columns:
        series = base[col]
        if pd.api.types.is_bool_dtype(series):
            numeric_parts[col] = series.fillna(False).astype(float)
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric_parts[col] = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
            continue
        converted = pd.to_numeric(series, errors="coerce")
        present = series.notna() & series.astype(str).str.strip().ne("")
        if bool(present.any()) and bool(converted[present].notna().all()):
            numeric_parts[col] = converted.fillna(0.0).astype(float)
        else:
            categorical_cols.append(col)
    if numeric_parts:
        frames.append(pd.DataFrame(numeric_parts, index=base.index))
    if categorical_cols:
        frames.append(pd.get_dummies(base[categorical_cols].fillna("unknown").astype(str), dummy_na=False))
    if not frames:
        return pd.DataFrame(index=metadata.index)
    return pd.concat(frames, axis=1).astype(float)


def _align(train_x: pd.DataFrame, other_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    columns = sorted(set(train_x.columns) | set(other_x.columns))
    return train_x.reindex(columns=columns, fill_value=0.0), other_x.reindex(columns=columns, fill_value=0.0), columns


def _prediction_frame(source: pd.DataFrame, probabilities: np.ndarray, split: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recording_id": source["recording_id"].to_numpy(),
            "participant_id": source["participant_id"].to_numpy(),
            "dataset": source.get("dataset", pd.Series([""] * len(source))).to_numpy(),
            "label_binary": source["label_binary"].to_numpy(),
            "split": split,
            "model_name": "metadata_logistic_regression",
            "probability": probabilities,
        }
    )


def train_metadata_baseline(
    metadata: pd.DataFrame,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
) -> MetadataBaselineResult:
    df = metadata[metadata["label_binary"].isin(["positive", "negative"]) & metadata["split"].isin(["train", "validation", "test"])].copy()
    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Need train/validation/test rows for metadata baseline")
    train_x_raw = build_metadata_feature_frame(train, feature_columns=feature_columns)
    val_x_raw = build_metadata_feature_frame(validation, feature_columns=feature_columns)
    test_x_raw = build_metadata_feature_frame(test, feature_columns=feature_columns)
    train_x, val_x, columns = _align(train_x_raw, val_x_raw)
    train_x, test_x, columns = _align(train_x, test_x_raw)
    val_x = val_x.reindex(columns=columns, fill_value=0.0)
    if train_x.empty:
        raise ValueError("No metadata features available")
    y_train = labels_to_binary(train["label_binary"])
    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state)
    model.fit(train_x, y_train)
    val_prob = model.predict_proba(val_x)[:, 1]
    test_prob = model.predict_proba(test_x)[:, 1]
    val_pred = _prediction_frame(validation, val_prob, "validation")
    test_pred = _prediction_frame(test, test_prob, "test")
    metrics = binary_metric_bundle(labels_to_binary(test["label_binary"]), test_prob)
    metrics["model_name"] = "metadata_logistic_regression"
    metrics["modality"] = "metadata"
    return MetadataBaselineResult(
        model_name="metadata_logistic_regression",
        metrics=metrics,
        validation_predictions=val_pred,
        test_predictions=test_pred,
        model=model,
        feature_columns=columns,
    )
