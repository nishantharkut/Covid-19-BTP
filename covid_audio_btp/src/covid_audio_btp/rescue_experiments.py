from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from covid_audio_btp.calibration import PlattCalibrator
from covid_audio_btp.cross_dataset import numeric_feature_columns
from covid_audio_btp.metrics import best_threshold_by_balanced_accuracy, binary_metric_bundle, labels_to_binary


DEFAULT_RESCUE_MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"]
DEFAULT_FEATURE_STRATEGIES = ["all", "drop_high_shift", "top_stable_50", "top_stable_80", "top_stable_120"]


@dataclass(frozen=True)
class ModelRun:
    model_name: str
    feature_strategy: str
    feature_count: int
    predictions: pd.DataFrame
    metrics: dict[str, object]


def scale_pos_weight_from_labels(labels: pd.Series | Iterable[object]) -> float:
    series = pd.Series(list(labels) if not isinstance(labels, pd.Series) else labels)
    positives = int((series == "positive").sum())
    negatives = int((series == "negative").sum())
    if positives <= 0:
        return 1.0
    return float(max(1.0, negatives / positives))


def _require_optional_classifier(model_name: str):
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("xgboost is not installed. Install with: python -m pip install xgboost") from exc
        return XGBClassifier
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("lightgbm is not installed. Install with: python -m pip install lightgbm") from exc
        return LGBMClassifier
    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("catboost is not installed. Install with: python -m pip install catboost") from exc
        return CatBoostClassifier
    raise ValueError(f"No optional classifier registered for {model_name}")


def make_rescue_model(model_name: str, labels: pd.Series, random_state: int = 42) -> object:
    pos_weight = scale_pos_weight_from_labels(labels)
    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=random_state)),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=800,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "xgboost":
        XGBClassifier = _require_optional_classifier(model_name)
        return XGBClassifier(
            n_estimators=700,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2.0,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=pos_weight,
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "lightgbm":
        LGBMClassifier = _require_optional_classifier(model_name)
        return LGBMClassifier(
            n_estimators=900,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            objective="binary",
            scale_pos_weight=pos_weight,
            n_jobs=-1,
            random_state=random_state,
            verbosity=-1,
        )
    if model_name == "catboost":
        CatBoostClassifier = _require_optional_classifier(model_name)
        return CatBoostClassifier(
            iterations=900,
            depth=5,
            learning_rate=0.025,
            loss_function="Logloss",
            eval_metric="AUC",
            scale_pos_weight=pos_weight,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown rescue model: {model_name}")


def _common_numeric_feature_columns(
    source_features: pd.DataFrame,
    external_features: pd.DataFrame,
    id_columns: list[str] | None = None,
) -> list[str]:
    source_cols = set(numeric_feature_columns(source_features, id_columns=id_columns))
    external_cols = set(numeric_feature_columns(external_features, id_columns=id_columns))
    return sorted(source_cols & external_cols)


def _parse_top_stable_k(strategy: str, top_k: int | None) -> int | None:
    if strategy.startswith("top_stable_"):
        try:
            return int(strategy.rsplit("_", 1)[-1])
        except ValueError:
            raise ValueError(f"Invalid top-stable strategy: {strategy}") from None
    return top_k


def _shift_report_map(shift_report: pd.DataFrame) -> pd.Series:
    if shift_report is None or shift_report.empty:
        raise ValueError("A non-empty feature shift report is required for stable-feature strategies")
    required = {"feature", "abs_standardized_mean_difference"}
    missing = required - set(shift_report.columns)
    if missing:
        raise ValueError(f"Shift report missing required columns: {sorted(missing)}")
    report = shift_report[["feature", "abs_standardized_mean_difference"]].copy()
    report["abs_standardized_mean_difference"] = pd.to_numeric(
        report["abs_standardized_mean_difference"], errors="coerce"
    ).fillna(np.inf)
    return report.drop_duplicates("feature").set_index("feature")["abs_standardized_mean_difference"]


def select_feature_columns_for_strategy(
    source_features: pd.DataFrame,
    external_features: pd.DataFrame,
    strategy: str = "all",
    shift_report: pd.DataFrame | None = None,
    smd_threshold: float = 0.5,
    top_k: int | None = None,
    id_columns: list[str] | None = None,
) -> list[str]:
    common_cols = _common_numeric_feature_columns(source_features, external_features, id_columns=id_columns)
    if not common_cols:
        raise ValueError("No common numeric feature columns are available")
    if strategy == "all":
        return common_cols

    shift_values = _shift_report_map(shift_report if shift_report is not None else pd.DataFrame())
    available = pd.DataFrame({"feature": common_cols})
    available["shift"] = available["feature"].map(shift_values).fillna(np.inf)

    if strategy == "drop_high_shift":
        selected = available[available["shift"] < float(smd_threshold)]["feature"].tolist()
    elif strategy == "top_stable" or strategy.startswith("top_stable_"):
        effective_k = _parse_top_stable_k(strategy, top_k)
        if effective_k is None:
            raise ValueError("top_stable requires top_k or a strategy name like top_stable_80")
        selected = (
            available.sort_values(["shift", "feature"], ascending=[True, True])
            .head(max(1, int(effective_k)))["feature"]
            .tolist()
        )
    else:
        raise ValueError(f"Unknown feature strategy: {strategy}")

    if not selected:
        raise ValueError(f"Feature strategy {strategy} selected zero columns")
    return selected


def _predict_probability(model: object, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError("Model does not expose predict_proba or decision_function")


def _fit_platt_or_identity(probabilities: np.ndarray, labels: pd.Series):
    y_true = labels_to_binary(labels)
    if len(np.unique(y_true)) < 2:
        return None, "identity_single_class_validation"
    calibrator = PlattCalibrator().fit(probabilities, y_true)
    return calibrator, "platt"


def _apply_calibrator(probabilities: np.ndarray, calibrator) -> np.ndarray:
    if calibrator is None:
        return np.asarray(probabilities, dtype=float)
    return np.asarray(calibrator.transform(probabilities), dtype=float)


def _prediction_frame(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    model_name: str,
    feature_strategy: str,
    split: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recording_id": source["recording_id"].to_numpy(),
            "participant_id": source["participant_id"].to_numpy(),
            "dataset": source.get("dataset", pd.Series(["external"] * len(source))).to_numpy(),
            "modality": source.get("modality", pd.Series(["cough"] * len(source))).to_numpy(),
            "label_binary": source["label_binary"].to_numpy(),
            "split": split,
            "model_name": model_name,
            "feature_strategy": feature_strategy,
            "probability": probabilities,
        }
    )


def evaluate_source_to_external(
    source_features: pd.DataFrame,
    external_features: pd.DataFrame,
    model_name: str,
    feature_strategy: str = "all",
    shift_report: pd.DataFrame | None = None,
    modality: str = "cough",
    source_train_split: str = "train",
    random_state: int = 42,
    smd_threshold: float = 0.5,
) -> ModelRun:
    source = source_features[
        (source_features["label_binary"].isin(["positive", "negative"]))
        & (source_features["modality"] == modality)
        & (source_features["split"].isin([source_train_split, "validation"]))
    ].copy()
    external = external_features[
        (external_features["label_binary"].isin(["positive", "negative"]))
        & (external_features["modality"] == modality)
    ].copy()
    if source.empty or external.empty:
        raise ValueError("Need non-empty source train rows and external labeled rows")
    train = source[source["split"] == source_train_split].copy()
    validation = source[source["split"] == "validation"].copy()
    if train.empty or validation.empty:
        raise ValueError("Need non-empty source train and validation rows")

    cols = select_feature_columns_for_strategy(
        train,
        external,
        strategy=feature_strategy,
        shift_report=shift_report,
        smd_threshold=smd_threshold,
    )
    x_train = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_varying_cols = [col for col in cols if x_train[col].nunique(dropna=False) > 1]
    if not train_varying_cols:
        raise ValueError(f"Feature strategy {feature_strategy} selected no train-varying columns")
    cols = train_varying_cols
    x_train = x_train[cols]
    x_validation = validation[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_external = external[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model = make_rescue_model(model_name, train["label_binary"], random_state=random_state)
    model.fit(x_train, labels_to_binary(train["label_binary"]))

    validation_raw = _predict_probability(model, x_validation)
    calibrator, calibration_method = _fit_platt_or_identity(validation_raw, validation["label_binary"])
    validation_prob = _apply_calibrator(validation_raw, calibrator)
    threshold = best_threshold_by_balanced_accuracy(labels_to_binary(validation["label_binary"]), validation_prob)

    raw_external = _predict_probability(model, x_external)
    probabilities = _apply_calibrator(raw_external, calibrator)
    predictions = _prediction_frame(external, probabilities, model_name, feature_strategy, split="external")
    predictions["raw_probability"] = raw_external
    predictions["calibration_method"] = calibration_method
    metrics = binary_metric_bundle(labels_to_binary(external["label_binary"]), probabilities, threshold=threshold)
    metrics.update(
        {
            "model_name": model_name,
            "modality": modality,
            "feature_strategy": feature_strategy,
            "calibration_method": calibration_method,
            "source_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "external_rows": int(len(external)),
            "n_features": int(len(cols)),
        }
    )
    return ModelRun(model_name, feature_strategy, len(cols), predictions, metrics)


def make_stratified_external_splits(
    features: pd.DataFrame,
    train_size: float = 0.6,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    df = features[features["label_binary"].isin(["positive", "negative"])].copy()
    if df.empty:
        raise ValueError("Need labeled positive/negative rows to create external splits")
    if not 0 < train_size < 1 or not 0 < validation_size < 1 or train_size + validation_size >= 1:
        raise ValueError("train_size and validation_size must be positive and sum to less than 1")
    train, remaining = train_test_split(
        df,
        train_size=train_size,
        stratify=df["label_binary"],
        random_state=random_state,
    )
    relative_validation = validation_size / (1.0 - train_size)
    validation, test = train_test_split(
        remaining,
        train_size=relative_validation,
        stratify=remaining["label_binary"],
        random_state=random_state,
    )
    out = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )
    return out


def evaluate_internal_splits(
    features_with_split: pd.DataFrame,
    model_name: str,
    modality: str = "cough",
    random_state: int = 42,
) -> ModelRun:
    df = features_with_split[
        (features_with_split["label_binary"].isin(["positive", "negative"]))
        & (features_with_split["modality"] == modality)
        & (features_with_split["split"].isin(["train", "validation", "test"]))
    ].copy()
    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]
    if train.empty or validation.empty or test.empty:
        raise ValueError("Need train, validation, and test rows for internal evaluation")
    cols = numeric_feature_columns(df)
    model = make_rescue_model(model_name, train["label_binary"], random_state=random_state)
    model.fit(train[cols].fillna(0.0), labels_to_binary(train["label_binary"]))
    validation_raw = _predict_probability(model, validation[cols].fillna(0.0))
    calibrator, calibration_method = _fit_platt_or_identity(validation_raw, validation["label_binary"])
    validation_prob = _apply_calibrator(validation_raw, calibrator)
    threshold = best_threshold_by_balanced_accuracy(labels_to_binary(validation["label_binary"]), validation_prob)
    test_raw = _predict_probability(model, test[cols].fillna(0.0))
    test_prob = _apply_calibrator(test_raw, calibrator)
    predictions = _prediction_frame(test, test_prob, model_name, "all", split="test")
    predictions["raw_probability"] = test_raw
    predictions["calibration_method"] = calibration_method
    metrics = binary_metric_bundle(labels_to_binary(test["label_binary"]), test_prob, threshold=threshold)
    metrics.update(
        {
            "model_name": model_name,
            "modality": modality,
            "feature_strategy": "all",
            "calibration_method": calibration_method,
            "dataset": str(test.get("dataset", pd.Series(["external"])).iloc[0]),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "n_features": int(len(cols)),
        }
    )
    return ModelRun(model_name, "all", len(cols), predictions, metrics)
