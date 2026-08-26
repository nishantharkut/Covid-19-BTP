from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


SUPPORTED_BOOTSTRAP_METRICS = ("auroc", "auprc", "brier", "ece")

EVIDENCE_GROUP_COLUMNS = (
    "run_id",
    "track",
    "stage",
    "protocol",
    "fold",
    "seed",
    "dataset",
    "split",
    "model_name",
    "modality",
    "mode",
    "analysis_unit",
    "threshold_source",
)


def _validated_binary_arrays(
    y_true: np.ndarray | Iterable[int],
    y_prob: np.ndarray | Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional")
    if labels.size == 0 or labels.size != probabilities.size:
        raise ValueError("labels and probabilities must have equal nonzero length")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must contain only binary values 0 and 1")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError("probabilities must be finite and within [0, 1]")
    return labels.astype(np.int64, copy=False), probabilities


def calibration_bin_table(
    y_true: np.ndarray | Iterable[int],
    y_prob: np.ndarray | Iterable[float],
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    labels, probabilities = _validated_binary_arrays(y_true, y_prob)
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == n_bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        if count:
            mean_probability = float(probabilities[mask].mean())
            observed_rate = float(labels[mask].mean())
            absolute_gap = abs(mean_probability - observed_rate)
        else:
            mean_probability = observed_rate = absolute_gap = float("nan")
        rows.append(
            {
                "bin": index,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_gap": float(absolute_gap),
            }
        )
    return pd.DataFrame(rows)


def complete_metric_bundle(
    y_true: np.ndarray | Iterable[int],
    y_prob: np.ndarray | Iterable[float],
    *,
    threshold: float,
    n_bins: int = 10,
) -> dict[str, float | int]:
    labels, probabilities = _validated_binary_arrays(y_true, y_prob)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    prevalence = float(labels.mean())
    calibration = calibration_bin_table(labels, probabilities, n_bins=n_bins)
    nonempty = calibration[calibration["count"] > 0]
    ece = float(
        (
            nonempty["count"]
            / labels.size
            * nonempty["absolute_gap"]
        ).sum()
    )
    mce = (
        float(nonempty["absolute_gap"].max())
        if not nonempty.empty
        else float("nan")
    )
    if np.unique(labels).size == 2:
        auroc = float(roc_auc_score(labels, probabilities))
        auprc = float(average_precision_score(labels, probabilities))
    else:
        auroc = auprc = float("nan")
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "auprc_lift": float(auprc / prevalence) if prevalence > 0.0 else float("nan"),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "ppv": float(tp / (tp + fp)) if tp + fp else 0.0,
        "npv": float(tn / (tn + fn)) if tn + fn else 0.0,
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": ece,
        "mce": mce,
        "nll": float(log_loss(labels, clipped, labels=[0, 1])),
        "threshold": float(threshold),
        "prevalence": prevalence,
        "n_samples": int(labels.size),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _participant_table(
    predictions: pd.DataFrame,
    *,
    participant_column: str = "participant_id",
    label_column: str = "label_binary",
    probability_column: str = "probability",
) -> pd.DataFrame:
    required = {participant_column, label_column, probability_column}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"prediction table is missing columns: {missing}")
    frame = predictions.loc[:, [participant_column, label_column, probability_column]].copy()
    if frame.empty or frame[participant_column].isna().any() or frame[participant_column].astype(str).eq("").any():
        raise ValueError("participant identifiers must be present and nonempty")
    label_counts = frame.groupby(participant_column, sort=False)[label_column].nunique(dropna=False)
    if (label_counts != 1).any():
        raise ValueError("each participant must have one stable binary label")
    aggregated = (
        frame.groupby(participant_column, sort=False, as_index=False)
        .agg({label_column: "first", probability_column: "mean"})
    )
    normalized_labels = aggregated[label_column].map(
        lambda value: {
            "negative": 0,
            "positive": 1,
            "0": 0,
            "1": 1,
        }.get(str(value).strip().lower(), value)
    )
    labels, probabilities = _validated_binary_arrays(
        normalized_labels.to_numpy(),
        aggregated[probability_column].to_numpy(),
    )
    aggregated[label_column] = labels
    aggregated[probability_column] = probabilities
    return aggregated


def participant_bootstrap_indices(
    participant_ids: np.ndarray | Iterable[object],
    rng: np.random.Generator,
) -> np.ndarray:
    identifiers = np.asarray(participant_ids)
    if identifiers.ndim != 1 or identifiers.size == 0:
        raise ValueError("participant_ids must be a nonempty one-dimensional array")
    unique_ids = pd.unique(identifiers)
    sampled_ids = rng.choice(unique_ids, size=len(unique_ids), replace=True)
    clusters = [np.flatnonzero(identifiers == participant) for participant in sampled_ids]
    return np.concatenate(clusters).astype(np.int64, copy=False)


def _metric_value(labels: np.ndarray, probabilities: np.ndarray, metric: str) -> float:
    if metric == "auroc":
        return (
            float(roc_auc_score(labels, probabilities))
            if np.unique(labels).size == 2
            else float("nan")
        )
    if metric == "auprc":
        return (
            float(average_precision_score(labels, probabilities))
            if np.unique(labels).size == 2
            else float("nan")
        )
    if metric == "brier":
        return float(brier_score_loss(labels, probabilities))
    if metric == "ece":
        calibration = calibration_bin_table(labels, probabilities, n_bins=10)
        nonempty = calibration[calibration["count"] > 0]
        return float(
            (nonempty["count"] / labels.size * nonempty["absolute_gap"]).sum()
        )
    raise ValueError(
        f"unsupported bootstrap metric {metric!r}; expected one of "
        f"{SUPPORTED_BOOTSTRAP_METRICS}"
    )


def _percentile_interval(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return float(low), float(high)


def participant_bootstrap_ci(
    predictions: pd.DataFrame,
    *,
    metric: str,
    n_bootstraps: int = 2_000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be positive")
    frame = _participant_table(predictions)
    labels = frame["label_binary"].to_numpy(dtype=np.int64)
    probabilities = frame["probability"].to_numpy(dtype=np.float64)
    participants = frame["participant_id"].to_numpy()
    point = _metric_value(labels, probabilities, metric)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstraps):
        indices = participant_bootstrap_indices(participants, rng)
        value = _metric_value(labels[indices], probabilities[indices], metric)
        if np.isfinite(value):
            values.append(float(value))
    ci_low, ci_high = _percentile_interval(values)
    return {
        "metric": metric,
        "point": float(point),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "valid_replicates": len(values),
        "requested_replicates": int(n_bootstraps),
        "seed": int(seed),
        "n_participants": int(len(frame)),
        "design": "participant_cluster",
    }


def bootstrap_metric_delta(
    left_predictions: pd.DataFrame,
    right_predictions: pd.DataFrame,
    *,
    metric: str,
    paired: bool,
    n_bootstraps: int = 2_000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be positive")
    left = _participant_table(left_predictions)
    right = _participant_table(right_predictions)
    point = _metric_value(
        left["label_binary"].to_numpy(), left["probability"].to_numpy(), metric
    ) - _metric_value(
        right["label_binary"].to_numpy(), right["probability"].to_numpy(), metric
    )
    rng = np.random.default_rng(seed)
    values: list[float] = []
    if paired:
        left = left.sort_values("participant_id").reset_index(drop=True)
        right = right.sort_values("participant_id").reset_index(drop=True)
        if left["participant_id"].tolist() != right["participant_id"].tolist():
            raise ValueError("paired comparison requires identical participant sets")
        if not np.array_equal(
            left["label_binary"].to_numpy(), right["label_binary"].to_numpy()
        ):
            raise ValueError("paired comparison requires identical participant labels")
        participant_ids = left["participant_id"].to_numpy()
        for _ in range(n_bootstraps):
            indices = participant_bootstrap_indices(participant_ids, rng)
            left_value = _metric_value(
                left["label_binary"].to_numpy()[indices],
                left["probability"].to_numpy()[indices],
                metric,
            )
            right_value = _metric_value(
                right["label_binary"].to_numpy()[indices],
                right["probability"].to_numpy()[indices],
                metric,
            )
            delta = left_value - right_value
            if np.isfinite(delta):
                values.append(float(delta))
        design = "paired_participant_cluster"
    else:
        left_ids = left["participant_id"].to_numpy()
        right_ids = right["participant_id"].to_numpy()
        for _ in range(n_bootstraps):
            left_indices = participant_bootstrap_indices(left_ids, rng)
            right_indices = participant_bootstrap_indices(right_ids, rng)
            left_value = _metric_value(
                left["label_binary"].to_numpy()[left_indices],
                left["probability"].to_numpy()[left_indices],
                metric,
            )
            right_value = _metric_value(
                right["label_binary"].to_numpy()[right_indices],
                right["probability"].to_numpy()[right_indices],
                metric,
            )
            delta = left_value - right_value
            if np.isfinite(delta):
                values.append(float(delta))
        design = "independent_two_sample_participant_cluster"
    ci_low, ci_high = _percentile_interval(values)
    return {
        "metric": metric,
        "point": float(point),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "valid_replicates": len(values),
        "requested_replicates": int(n_bootstraps),
        "seed": int(seed),
        "left_n_participants": int(len(left)),
        "right_n_participants": int(len(right)),
        "design": design,
    }


def select_sensitivity_threshold(
    y_true: np.ndarray | Iterable[int],
    y_prob: np.ndarray | Iterable[float],
    *,
    minimum_sensitivity: float = 0.90,
) -> float:
    labels, probabilities = _validated_binary_arrays(y_true, y_prob)
    if not 0.0 < minimum_sensitivity <= 1.0:
        raise ValueError("minimum_sensitivity must be within (0, 1]")
    if int(labels.sum()) == 0:
        raise ValueError("sensitivity threshold selection requires positive validation rows")
    candidates = np.unique(np.concatenate(([0.0], probabilities)))[::-1]
    for threshold in candidates:
        predictions = probabilities >= threshold
        sensitivity = float(np.sum(predictions & (labels == 1)) / np.sum(labels == 1))
        if sensitivity >= minimum_sensitivity:
            return float(threshold)
    raise RuntimeError("no threshold satisfies the requested minimum sensitivity")


def decision_curve_analysis(
    predictions: pd.DataFrame,
    *,
    thresholds: np.ndarray | Iterable[float] | None = None,
) -> pd.DataFrame:
    frame = _participant_table(predictions)
    labels = frame["label_binary"].to_numpy(dtype=np.int64)
    probabilities = frame["probability"].to_numpy(dtype=np.float64)
    threshold_values = (
        np.arange(0.01, 0.51, 0.01, dtype=np.float64)
        if thresholds is None
        else np.asarray(list(thresholds), dtype=np.float64)
    )
    if (
        threshold_values.ndim != 1
        or threshold_values.size == 0
        or not np.isfinite(threshold_values).all()
        or (threshold_values <= 0.0).any()
        or (threshold_values >= 1.0).any()
    ):
        raise ValueError("decision thresholds must be finite and strictly within (0, 1)")
    prevalence = float(labels.mean())
    rows: list[dict[str, float | int]] = []
    for threshold in threshold_values:
        predicted = probabilities >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        odds = float(threshold / (1.0 - threshold))
        rows.append(
            {
                "threshold_probability": float(threshold),
                "model_net_benefit": float(tp / len(labels) - fp / len(labels) * odds),
                "treat_all_net_benefit": float(prevalence - (1.0 - prevalence) * odds),
                "treat_none_net_benefit": 0.0,
                "n_participants": int(len(labels)),
                "prevalence": prevalence,
            }
        )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prediction_inputs(run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    track_a = run_dir / "track_a_predictions.csv"
    if track_a.is_file():
        paths.append(track_a)
    track_b = run_dir / "track_b"
    if track_b.is_dir():
        paths.extend(sorted(track_b.glob("*_participant_predictions.csv")))
    if not paths:
        raise ValueError(f"no aggregate prediction files found under {run_dir}")
    return paths


def _load_predictions(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        required = {
            "run_id",
            "track",
            "protocol",
            "seed",
            "dataset",
            "split",
            "model_name",
            "modality",
            "participant_id",
            "label_binary",
            "probability",
            "threshold",
            "threshold_source",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing prediction columns: {missing}")
        frame["source_prediction_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _metric_groups(predictions: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    group_columns = [column for column in EVIDENCE_GROUP_COLUMNS if column in predictions]
    rows: list[dict[str, object]] = []
    for key, group in predictions.groupby(group_columns, dropna=False, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        participant = _participant_table(group)
        thresholds = pd.to_numeric(group["threshold"], errors="coerce").dropna().unique()
        if len(thresholds) != 1:
            raise ValueError("each evidence group must have one frozen threshold")
        metrics = complete_metric_bundle(
            participant["label_binary"].to_numpy(),
            participant["probability"].to_numpy(),
            threshold=float(thresholds[0]),
        )
        rows.append(dict(zip(group_columns, keys)) | metrics)
    return pd.DataFrame(rows), group_columns


def _evaluation_groups(predictions: pd.DataFrame) -> Iterable[tuple[dict[str, object], pd.DataFrame]]:
    evaluation = predictions[predictions["split"].astype(str).isin(("test", "external", "outer_test"))]
    identity = [
        column
        for column in ("run_id", "track", "stage", "protocol", "seed", "dataset", "split", "model_name", "modality", "mode")
        if column in evaluation
    ]
    for key, group in evaluation.groupby(identity, dropna=False, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        yield dict(zip(identity, keys)), group


def _bootstrap_table(
    predictions: pd.DataFrame,
    *,
    n_bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for identity, group in _evaluation_groups(predictions):
        for metric in SUPPORTED_BOOTSTRAP_METRICS:
            rows.append(
                identity
                | participant_bootstrap_ci(
                    group,
                    metric=metric,
                    n_bootstraps=n_bootstraps,
                    seed=seed,
                )
            )
    return pd.DataFrame(rows)


def _paired_model_comparisons(
    predictions: pd.DataFrame,
    *,
    n_bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    evaluation = predictions[
        predictions["track"].astype(str).eq("B")
        & predictions["split"].astype(str).eq("test")
    ]
    context_columns = [
        column
        for column in ("stage", "protocol", "seed", "dataset", "modality")
        if column in evaluation
    ]
    rows: list[dict[str, object]] = []
    for key, context in evaluation.groupby(context_columns, dropna=False, sort=True):
        models = sorted(context["model_name"].astype(str).unique())
        if "dndf" not in models or "dndt" not in models:
            continue
        keys = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(context_columns, keys))
        left = context[context["model_name"].astype(str).eq("dndf")]
        right = context[context["model_name"].astype(str).eq("dndt")]
        for metric in SUPPORTED_BOOTSTRAP_METRICS:
            rows.append(
                identity
                | {"left_model": "dndf", "right_model": "dndt"}
                | bootstrap_metric_delta(
                    left,
                    right,
                    metric=metric,
                    paired=True,
                    n_bootstraps=n_bootstraps,
                    seed=seed,
                )
            )
    return pd.DataFrame(rows)


def _external_deltas(
    predictions: pd.DataFrame,
    *,
    n_bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    external = predictions[
        predictions["track"].astype(str).eq("B")
        & predictions["split"].astype(str).eq("external")
    ]
    source = predictions[
        predictions["track"].astype(str).eq("B")
        & predictions["stage"].astype(str).eq("final")
        & predictions["protocol"].astype(str).eq("existing")
        & predictions["split"].astype(str).eq("test")
    ]
    rows: list[dict[str, object]] = []
    match_columns = ("seed", "model_name", "modality")
    for key, target_group in external.groupby(list(match_columns), sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        matched = source.copy()
        for column, value in zip(match_columns, key_tuple):
            matched = matched[matched[column].astype(str).eq(str(value))]
        if matched.empty:
            continue
        identity = dict(zip(match_columns, key_tuple)) | {
            "source_dataset": str(matched["dataset"].iloc[0]),
            "target_dataset": str(target_group["dataset"].iloc[0]),
        }
        for metric in SUPPORTED_BOOTSTRAP_METRICS:
            rows.append(
                identity
                | bootstrap_metric_delta(
                    matched,
                    target_group,
                    metric=metric,
                    paired=False,
                    n_bootstraps=n_bootstraps,
                    seed=seed,
                )
            )
    return pd.DataFrame(rows)


def _fixed_sensitivity_table(predictions: pd.DataFrame) -> pd.DataFrame:
    track_b = predictions[predictions["track"].astype(str).eq("B")]
    context_columns = [
        column
        for column in ("stage", "protocol", "seed", "model_name", "modality")
        if column in track_b
    ]
    rows: list[dict[str, object]] = []
    for key, context in track_b.groupby(context_columns, dropna=False, sort=True):
        validation = context[context["split"].astype(str).eq("validation")]
        evaluation = context[context["split"].astype(str).isin(("test", "external"))]
        if validation.empty or evaluation.empty:
            continue
        validation_participant = _participant_table(validation)
        threshold = select_sensitivity_threshold(
            validation_participant["label_binary"].to_numpy(),
            validation_participant["probability"].to_numpy(),
            minimum_sensitivity=0.90,
        )
        keys = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(context_columns, keys))
        for (dataset, split), group in evaluation.groupby(["dataset", "split"], sort=True):
            participant = _participant_table(group)
            rows.append(
                identity
                | {
                    "dataset": dataset,
                    "split": split,
                    "threshold_source": "source_validation_sensitivity_at_least_0.90",
                }
                | complete_metric_bundle(
                    participant["label_binary"].to_numpy(),
                    participant["probability"].to_numpy(),
                    threshold=threshold,
                )
            )
    return pd.DataFrame(rows)


def _decision_curve_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for identity, group in _evaluation_groups(predictions):
        rows.append(decision_curve_analysis(group).assign(**identity))
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _model_selection_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["modality", "selection_json"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("selected configurations must be a JSON object")
    return pd.DataFrame(
        [
            {"modality": modality, "selection_json": json.dumps(value, sort_keys=True)}
            for modality, value in sorted(payload.items())
        ]
    )


def generate_evidence(
    run_dir: str | Path,
    *,
    n_bootstraps: int = 2_000,
    seed: int = 42,
) -> dict[str, object]:
    run_path = Path(run_dir).resolve()
    if not run_path.is_dir():
        raise ValueError(f"run directory does not exist: {run_path}")
    prediction_paths = _prediction_inputs(run_path)
    predictions = _load_predictions(prediction_paths)
    run_ids = predictions["run_id"].astype(str).unique()
    if len(run_ids) != 1 or run_ids[0] != run_path.name:
        raise ValueError("prediction run_id must match the run directory name")
    evidence_dir = run_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    metrics, _ = _metric_groups(predictions)
    final_summary = metrics[metrics["split"].astype(str).isin(("test", "external", "outer_test"))].copy()
    bootstrap = _bootstrap_table(
        predictions, n_bootstraps=n_bootstraps, seed=seed
    )
    paired = _paired_model_comparisons(
        predictions, n_bootstraps=n_bootstraps, seed=seed
    )
    external = _external_deltas(
        predictions, n_bootstraps=n_bootstraps, seed=seed
    )
    operating = _fixed_sensitivity_table(predictions)
    decision = _decision_curve_table(predictions)
    selected_path = run_path / "track_b" / "selected_configurations.json"
    model_selection = _model_selection_table(selected_path)
    fusion_source = run_path / "track_b" / "fusion_weights.csv"
    fusion = (
        pd.read_csv(fusion_source)
        if fusion_source.is_file()
        else pd.DataFrame(columns=["protocol", "seed", "fusion_method", "modality", "weight"])
    )
    tables = {
        "metrics.csv": metrics,
        "final_summary.csv": final_summary,
        "bootstrap_ci.csv": bootstrap,
        "paired_comparisons.csv": paired,
        "external_deltas.csv": external,
        "fixed_sensitivity_operating_points.csv": operating,
        "decision_curve.csv": decision,
        "model_selection.csv": model_selection,
        "fusion_weights.csv": fusion,
    }
    for name, table in tables.items():
        _atomic_csv(table, evidence_dir / name)
    input_paths = prediction_paths + ([selected_path] if selected_path.is_file() else [])
    if fusion_source.is_file():
        input_paths.append(fusion_source)
    inputs = {
        path.name: {"path": str(path), "sha256": _sha256(path)}
        for path in input_paths
    }
    outputs = {
        name: {
            "path": str(evidence_dir / name),
            "sha256": _sha256(evidence_dir / name),
            "rows": int(len(table)),
        }
        for name, table in tables.items()
    }
    manifest: dict[str, object] = {
        "format_version": 1,
        "run_id": str(run_ids[0]),
        "n_bootstraps": int(n_bootstraps),
        "bootstrap_seed": int(seed),
        "bootstrap_unit": "participant",
        "inputs": inputs,
        "outputs": outputs,
    }
    _atomic_json(manifest, evidence_dir / "run_manifest.json")
    return manifest
