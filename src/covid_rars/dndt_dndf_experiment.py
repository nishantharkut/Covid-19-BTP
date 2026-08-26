from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE, SVMSMOTE
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from covid_rars.dndt_dndf_models import ModelConfig, NeuralDecisionClassifier
from covid_rars.metrics import binary_metric_bundle, best_threshold_by_balanced_accuracy


BalanceMethod = Literal["svm_smote", "smote", "class_weight"]
CheckpointRole = Literal["latest_recovery", "best_inference"]
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_CHECKPOINT_ROLES: dict[str, str] = {
    "latest_recovery": "recovery",
    "best_inference": "inference",
}
_MAX_DEFERRED_CLEANUP_FILENAMES = 32
SMOTE_K_NEIGHBORS = 5
SVMSMOTE_K_NEIGHBORS = 5
SVMSMOTE_M_NEIGHBORS = 10
TRACK_A_AUTHOR_COMMIT = "feb0e63c790c042eaa21e9f3fc83ed64bdc8a24e"
TRACK_A_FEATURE_RELATIVE = "Extracted Features/Coswara/cough_X_features_np.npy"
TRACK_A_LABEL_RELATIVE = "Extracted Features/Coswara/cough_y_features_np.npy"
TRACK_A_EXPECTED_SHAPE = (1319, 193)
TRACK_A_EXPECTED_CLASS_COUNTS = {"C": 185, "N": 1134}
EXPECTED_TRACK_A_SELECTED_INDICES = (
    0, 1, 2, 5, 6, 7, 9, 10, 15, 20, 21, 29, 33, 35, 36, 39, 42,
    47, 49, 65, 66, 69, 108, 120, 121, 136, 137, 138, 180, 182, 185,
    187, 188,
)
TRACK_A_PINNED_SHA256: dict[str, str] = {
    TRACK_A_FEATURE_RELATIVE: "e61f43b6b5082003725bcf785cc97d934a14a83cda70d8ff32eefb3b21c4a38e",
    TRACK_A_LABEL_RELATIVE: "d78ec9af0c8d0a59b6634558d044e007c293ac05850da05c5d9a5bd15958cf97",
    "Train-Test Split/coswaradataset/train/0.csv": "223f5d5ed4248ab44a19d4cbcbd20f552de82984750eda75392fd6c578d83a8b",
    "Train-Test Split/coswaradataset/test/0.csv": "87a77c3dbac8b8c33aad277d6128dd230d761adb4fe089a5ecb2529b3b817534",
    "Train-Test Split/coswaradataset/train/1.csv": "ee1d6c963752565e642484c65ea593cbe681efb8657a7ac98e4d097c2f1490bc",
    "Train-Test Split/coswaradataset/test/1.csv": "54f131c907df3b0806562424e249c6e6a6db6e5938e0eb8353c68b5cd32888c7",
    "Train-Test Split/coswaradataset/train/2.csv": "0935b4f7da81b9714bbb5990b61af1590d4468900c1cfbf738df9b0134ede9d7",
    "Train-Test Split/coswaradataset/test/2.csv": "9d8faa0859f64c1a79389797278f5418a3f6a795dd65e2a2a330ed3af79d3b5e",
    "Train-Test Split/coswaradataset/train/3.csv": "6d3112bf9d4db258a4daca711f60b5ea760f59dc4c5a62e846f9d617d62bf3da",
    "Train-Test Split/coswaradataset/test/3.csv": "38fdca51beca90cb0dbad1743ddd938f3dbb45e0ff0d5e24e321b76872df0171",
    "Train-Test Split/coswaradataset/train/4.csv": "132c15fde68eb04cb3db06e3d5509b2d5cf3b998042f05f47e039e0d8d8eb611",
    "Train-Test Split/coswaradataset/test/4.csv": "e942a34634e675759e15582a8951f0beacaafcc3616ee52c4c11d2f98ad3f5a7",
    "Train-Test Split/coswaradataset/train/5.csv": "54ce547ddb55f2a8b11e00760460e461ac98516024a1206de88c35dadc1e2c64",
    "Train-Test Split/coswaradataset/test/5.csv": "df13aa5474228f750d5c07ba4f9ff56e612d106f7d887ad5ec39dcb2022484a3",
    "Train-Test Split/coswaradataset/train/6.csv": "4467a0e7712eeec0c0d9df3dd2420a21befbe669a18b04a0e9a2fbecaf16fe38",
    "Train-Test Split/coswaradataset/test/6.csv": "5a20f359710c4377a50e6715d7d84a8c7b0b02740c3acf41bd9fd30b4bf99c80",
    "Train-Test Split/coswaradataset/train/7.csv": "f81e63b66a82c372606cbac5ed42a3921fa25bdabc627cd0c4f967e65ac696c7",
    "Train-Test Split/coswaradataset/test/7.csv": "88ba5377bd7ea6fa4a8e32c461341d95a67c1f0eab3b554bfffd6eeb869ca738",
    "Train-Test Split/coswaradataset/train/8.csv": "bd2366a80fc203a49a7cb0df3d0bbae7680f7a38d54e545db56d19e52773dc2d",
    "Train-Test Split/coswaradataset/test/8.csv": "5bb8dffe61133a7c12b9c45f9961922076aeea9b0c8add334203baa6b46ec28e",
    "Train-Test Split/coswaradataset/train/9.csv": "cdee565ea64e4bc414c823af61285236180df87a3dcc7999353bc7410a700f7c",
    "Train-Test Split/coswaradataset/test/9.csv": "e26dbae308b474c8b50542fa8763d4c5e17c1ad536829662f4edeed991bf9b7f",
}

PREDICTION_COLUMNS = (
    "run_id",
    "track",
    "protocol",
    "fold",
    "seed",
    "dataset",
    "split",
    "model_name",
    "modality",
    "participant_id",
    "recording_id",
    "label_binary",
    "probability",
    "threshold",
    "threshold_source",
    "configuration_sha256",
    "split_sha256",
    "feature_sha256",
    "checkpoint_sha256",
)
_TRACK_A_COLUMNS = ("analysis_id", "analysis_unit")


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    seed: int
    balance_method: BalanceMethod

    def __post_init__(self) -> None:
        if not _positive_finite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if not _nonnegative_finite(self.weight_decay):
            raise ValueError("weight_decay must be finite and nonnegative")
        _require_integer("batch_size", self.batch_size, minimum=2)
        _require_integer("max_epochs", self.max_epochs, minimum=1)
        _require_integer("patience", self.patience, minimum=1)
        _require_integer("seed", self.seed, minimum=0)
        if self.balance_method not in {"svm_smote", "smote", "class_weight"}:
            raise ValueError(
                "balance_method must be 'svm_smote', 'smote', or 'class_weight'"
            )


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    validation_auroc: float
    validation_auprc: float
    threshold: float
    checkpoint_path: Path
    validation_probability: np.ndarray


@dataclass(frozen=True)
class FittedPreprocessor:
    imputer: SimpleImputer
    scaler: StandardScaler
    n_features: int


@dataclass(frozen=True)
class BalanceResult:
    features: np.ndarray
    labels: np.ndarray
    sample_weights: np.ndarray | None
    audit: dict[str, object]


@dataclass(frozen=True)
class ResolvedCheckpoint:
    path: Path
    descriptor: dict[str, object]
    payload: dict[str, object]
    used_fallback: bool
    deferred_cleanup: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuthorTrackAArtifacts:
    features: np.ndarray
    author_labels: np.ndarray
    covid_labels: np.ndarray
    train_folds: tuple[np.ndarray, ...]
    test_folds: tuple[np.ndarray, ...]
    source_hashes: dict[str, str]
    fold_hashes: dict[str, str]
    author_commit: str


@dataclass(frozen=True)
class TrackAFeatureCache:
    selected_features: np.ndarray
    selected_indices: np.ndarray
    cache_key: str
    selected_features_sha256: str
    selected_indices_sha256: str
    manifest_path: Path


class PlannedInterruption(RuntimeError):
    """Raised by tests or controllers only after an epoch checkpoint is durable."""


def _git_output(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed for {repository}: {detail}")
    return process.stdout.strip()


def _read_track_a_fold_indices(path: Path, expected_first_column: str) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"fold CSV has no header: {path}") from exc
        if not header or header[0].strip() != expected_first_column:
            raise ValueError(
                f"fold CSV {path} must have {expected_first_column!r} as its first column"
            )
        values: list[int] = []
        for line_number, row in enumerate(reader, start=2):
            if not row or not row[0].strip():
                continue
            raw = row[0].strip()
            try:
                numeric = float(raw)
            except ValueError as exc:
                raise ValueError(f"invalid fold index at {path}:{line_number}") from exc
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"non-integer fold index at {path}:{line_number}")
            values.append(int(numeric))
    return np.asarray(values, dtype=np.int64)


def load_track_a_author_artifacts(
    author_repo: str | Path,
    expected_commit: str = TRACK_A_AUTHOR_COMMIT,
    *,
    expected_hashes: Mapping[str, str] | None = None,
    require_clean_tracked_tree: bool = True,
) -> AuthorTrackAArtifacts:
    repository = Path(author_repo)
    if not repository.is_dir():
        raise FileNotFoundError(f"author repository does not exist: {repository}")
    head = _git_output(repository, "rev-parse", "HEAD")
    if head != expected_commit:
        raise RuntimeError(
            f"author repository HEAD mismatch: expected {expected_commit}, found {head}"
        )
    if require_clean_tracked_tree:
        status = _git_output(
            repository, "status", "--porcelain", "--untracked-files=no"
        )
        if status:
            raise RuntimeError("author repository has tracked worktree changes")

    expected = dict(expected_hashes or TRACK_A_PINNED_SHA256)
    required_relative = [TRACK_A_FEATURE_RELATIVE, TRACK_A_LABEL_RELATIVE]
    for fold in range(10):
        required_relative.extend(
            [
                f"Train-Test Split/coswaradataset/train/{fold}.csv",
                f"Train-Test Split/coswaradataset/test/{fold}.csv",
            ]
        )
    if set(expected) != set(required_relative):
        raise ValueError("expected_hashes must cover exactly the 22 Track A artifacts")
    actual_hashes: dict[str, str] = {}
    for relative in required_relative:
        path = repository / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"required Track A artifact is missing: {path}")
        digest = checkpoint_sha256(path)
        actual_hashes[relative] = digest
        if digest != expected[relative].lower():
            raise RuntimeError(
                f"SHA256 mismatch for {relative}: expected {expected[relative]}, found {digest}"
            )

    features = np.load(repository / TRACK_A_FEATURE_RELATIVE, allow_pickle=False)
    labels_raw = np.load(repository / TRACK_A_LABEL_RELATIVE, allow_pickle=False)
    if features.shape != TRACK_A_EXPECTED_SHAPE:
        raise ValueError(
            f"author X must have shape {TRACK_A_EXPECTED_SHAPE}, found {features.shape}"
        )
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError("author X must use a floating dtype")
    if not np.isfinite(features).all():
        raise ValueError("author X must contain only finite values")
    if labels_raw.shape != (TRACK_A_EXPECTED_SHAPE[0],):
        raise ValueError("author y must contain exactly 1319 labels")
    labels_text = labels_raw.astype(str)
    counts = {token: int(np.count_nonzero(labels_text == token)) for token in ("C", "N")}
    if counts != TRACK_A_EXPECTED_CLASS_COUNTS or set(np.unique(labels_text)) != {"C", "N"}:
        raise ValueError(
            f"author label contract mismatch: expected {TRACK_A_EXPECTED_CLASS_COUNTS}, found {counts}"
        )
    author_labels = np.where(labels_text == "C", 0, 1).astype(np.int64)
    covid_labels = np.where(labels_text == "C", 1, 0).astype(np.int64)

    all_indices = np.arange(TRACK_A_EXPECTED_SHAPE[0], dtype=np.int64)
    train_folds: list[np.ndarray] = []
    test_folds: list[np.ndarray] = []
    fold_hashes: dict[str, str] = {}
    for fold in range(10):
        train_relative = f"Train-Test Split/coswaradataset/train/{fold}.csv"
        test_relative = f"Train-Test Split/coswaradataset/test/{fold}.csv"
        train = _read_track_a_fold_indices(repository / train_relative, "train_index")
        test = _read_track_a_fold_indices(repository / test_relative, "test_index")
        if len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test):
            raise ValueError(f"fold {fold} contains duplicate indices")
        if (
            np.any(train < 0)
            or np.any(test < 0)
            or np.any(train >= len(all_indices))
            or np.any(test >= len(all_indices))
        ):
            raise ValueError(f"fold {fold} contains out-of-range indices")
        if np.intersect1d(train, test).size:
            raise ValueError(f"fold {fold} train/test indices overlap")
        if not np.array_equal(np.sort(np.concatenate((train, test))), all_indices):
            raise ValueError(f"fold {fold} does not partition all author samples")
        train_folds.append(train)
        test_folds.append(test)
        fold_hashes[train_relative] = actual_hashes[train_relative]
        fold_hashes[test_relative] = actual_hashes[test_relative]
    combined_test = np.concatenate(test_folds)
    if len(combined_test) != len(all_indices) or not np.array_equal(
        np.sort(combined_test), all_indices
    ):
        raise ValueError("ten outer test folds must contain every sample exactly once")
    if len(np.unique(combined_test)) != len(all_indices):
        raise ValueError("outer test folds overlap")

    return AuthorTrackAArtifacts(
        features=np.asarray(features, dtype=np.float64),
        author_labels=author_labels,
        covid_labels=covid_labels,
        train_folds=tuple(train_folds),
        test_folds=tuple(test_folds),
        source_hashes={
            "features": actual_hashes[TRACK_A_FEATURE_RELATIVE],
            "labels": actual_hashes[TRACK_A_LABEL_RELATIVE],
        },
        fold_hashes=fold_hashes,
        author_commit=head,
    )


def _positive_finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _nonnegative_finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _require_integer(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _feature_matrix(features: object, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if np.isinf(matrix).any():
        raise ValueError(f"{name} must not contain infinite values")
    return matrix


def _binary_labels(labels: object, *, expected_rows: int, name: str) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1 or array.shape[0] != expected_rows:
        raise ValueError(f"{name} must contain one label per row")
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain only binary values 0 and 1")
    return array.astype(np.int64, copy=False)


def fit_preprocessor(training_features: object) -> FittedPreprocessor:
    training = _feature_matrix(training_features, name="training_features")
    all_missing = np.isnan(training).all(axis=0)
    if bool(np.any(all_missing)):
        indices = np.flatnonzero(all_missing).tolist()
        raise ValueError(f"training_features contain all-missing columns: {indices}")

    imputer = SimpleImputer(strategy="median", keep_empty_features=False)
    imputed = imputer.fit_transform(training)
    if not np.isfinite(imputed).all():
        raise ValueError("training features are not finite after imputation")
    scaler = StandardScaler()
    transformed = scaler.fit_transform(imputed)
    if not np.isfinite(transformed).all():
        raise ValueError("training features are not finite after scaling")
    return FittedPreprocessor(imputer, scaler, training.shape[1])


def transform_features(
    preprocessor: FittedPreprocessor, features: object
) -> np.ndarray:
    if not isinstance(preprocessor, FittedPreprocessor):
        raise ValueError("preprocessor must be a FittedPreprocessor")
    matrix = _feature_matrix(features, name="features")
    if matrix.shape[1] != preprocessor.n_features:
        raise ValueError(
            f"expected {preprocessor.n_features} feature columns, got {matrix.shape[1]}"
        )
    transformed = preprocessor.scaler.transform(
        preprocessor.imputer.transform(matrix)
    )
    if not np.isfinite(transformed).all():
        raise ValueError("features must be finite after preprocessing")
    return np.asarray(transformed, dtype=np.float32)


def _class_weight_result(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    requested_method: str,
    fallback_reason: str | None,
) -> BalanceResult:
    classes, counts = np.unique(labels, return_counts=True)
    if classes.tolist() != [0, 1]:
        raise ValueError("training labels must contain both binary classes")
    total = int(labels.size)
    class_weights = {
        int(label): float(total / (len(classes) * count))
        for label, count in zip(classes, counts)
    }
    sample_weights = np.asarray(
        [class_weights[int(label)] for label in labels], dtype=np.float32
    )
    audit: dict[str, object] = {
        "requested_method": requested_method,
        "applied_method": "class_weight",
        "fallback_reason": fallback_reason,
        "n_rows_before": total,
        "n_rows_after": total,
        "class_counts_before": {
            str(int(label)): int(count) for label, count in zip(classes, counts)
        },
        "class_counts_after": {
            str(int(label)): int(count) for label, count in zip(classes, counts)
        },
        "class_weights": {
            str(label): weight for label, weight in class_weights.items()
        },
    }
    return BalanceResult(features.copy(), labels.copy(), sample_weights, audit)


def balance_training_rows(
    training_features: object,
    training_labels: object,
    *,
    method: BalanceMethod,
    seed: int,
) -> BalanceResult:
    features = _feature_matrix(training_features, name="training_features")
    if not np.isfinite(features).all():
        raise ValueError("training_features must be finite before balancing")
    labels = _binary_labels(
        training_labels, expected_rows=features.shape[0], name="training_labels"
    )
    _require_integer("seed", seed, minimum=0)
    if method not in {"svm_smote", "smote", "class_weight"}:
        raise ValueError("unknown balance method")
    classes, counts = np.unique(labels, return_counts=True)
    if classes.tolist() != [0, 1]:
        raise ValueError("training labels must contain both binary classes")
    if method == "class_weight":
        return _class_weight_result(
            features, labels, requested_method=method, fallback_reason=None
        )

    minority_count = int(counts.min())
    k_neighbors = (
        SVMSMOTE_K_NEIGHBORS if method == "svm_smote" else SMOTE_K_NEIGHBORS
    )
    if minority_count <= k_neighbors:
        return _class_weight_result(
            features,
            labels,
            requested_method=method,
            fallback_reason=(
                f"minority count {minority_count} must exceed "
                f"k_neighbors={k_neighbors} for {method}"
            ),
        )
    if method == "svm_smote" and features.shape[0] <= SVMSMOTE_M_NEIGHBORS:
        return _class_weight_result(
            features,
            labels,
            requested_method=method,
            fallback_reason=(
                f"total row count {features.shape[0]} must exceed "
                f"m_neighbors={SVMSMOTE_M_NEIGHBORS} for svm_smote"
            ),
        )

    sampler = (
        SVMSMOTE(
            random_state=seed,
            k_neighbors=SVMSMOTE_K_NEIGHBORS,
            m_neighbors=SVMSMOTE_M_NEIGHBORS,
        )
        if method == "svm_smote"
        else SMOTE(random_state=seed, k_neighbors=SMOTE_K_NEIGHBORS)
    )
    try:
        resampled_features, resampled_labels = sampler.fit_resample(features, labels)
    except ValueError as exc:
        return _class_weight_result(
            features,
            labels,
            requested_method=method,
            fallback_reason=(
                f"{'SVMSMOTE' if method == 'svm_smote' else 'SMOTE'} raised "
                f"ValueError and fell back safely: {exc}"
            ),
        )
    after_classes, after_counts = np.unique(resampled_labels, return_counts=True)
    audit = {
        "requested_method": method,
        "applied_method": method,
        "fallback_reason": None,
        "n_rows_before": int(features.shape[0]),
        "n_rows_after": int(len(resampled_labels)),
        "class_counts_before": {
            str(int(label)): int(count) for label, count in zip(classes, counts)
        },
        "class_counts_after": {
            str(int(label)): int(count)
            for label, count in zip(after_classes, after_counts)
        },
        "class_weights": None,
    }
    return BalanceResult(
        np.asarray(resampled_features, dtype=np.float32),
        np.asarray(resampled_labels, dtype=np.int64),
        None,
        audit,
    )


def deterministic_batch_indices(
    n_rows: int, batch_size: int, *, seed: int, epoch: int
) -> list[np.ndarray]:
    _require_integer("n_rows", n_rows, minimum=2)
    _require_integer("batch_size", batch_size, minimum=2)
    _require_integer("seed", seed, minimum=0)
    _require_integer("epoch", epoch, minimum=0)
    order = np.random.default_rng(seed + epoch).permutation(n_rows)
    batches = [order[start : start + batch_size] for start in range(0, n_rows, batch_size)]
    if len(batches) > 1 and batches[-1].size == 1:
        batches[-2] = np.concatenate((batches[-2], batches[-1]))
        batches.pop()
    if any(batch.size < 2 for batch in batches):
        raise RuntimeError("deterministic batching produced a singleton batch")
    return batches


def fixed_order_batch_indices(n_rows: int, batch_size: int) -> list[np.ndarray]:
    _require_integer("n_rows", n_rows, minimum=2)
    _require_integer("batch_size", batch_size, minimum=2)
    order = np.arange(n_rows, dtype=np.int64)
    batches = [order[start : start + batch_size] for start in range(0, n_rows, batch_size)]
    if len(batches) > 1 and batches[-1].size == 1:
        batches[-2] = np.concatenate((batches[-2], batches[-1]))
        batches.pop()
    if any(batch.size < 2 for batch in batches):
        raise RuntimeError("fixed-order batching produced a singleton batch")
    return batches


def _atomic_numpy_save(array: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint_sha256(path)


def _track_a_rfecv_payload(artifacts: AuthorTrackAArtifacts) -> dict[str, object]:
    return {
        "source_hashes": artifacts.source_hashes,
        "algorithm": {
            "split": {"test_size": 0.20, "random_state": 42},
            "estimator": {
                "class": "ExtraTreesClassifier",
                "n_estimators": 50,
                "random_state": 0,
            },
            "rfecv": {
                "step": 1,
                "cv": "StratifiedKFold(10, shuffle=False)",
                "scoring": "roc_auc",
                "min_features_to_select": 1,
            },
        },
        "sklearn_version": sklearn_version,
    }


def _load_valid_track_a_cache(
    cache_dir: Path,
    *,
    cache_key: str,
    expected_rows: int,
    expected_indices: np.ndarray | None,
) -> TrackAFeatureCache | None:
    manifest_path = cache_dir / "manifest.json"
    features_path = cache_dir / "selected_features.npy"
    indices_path = cache_dir / "selected_indices.npy"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("cache_key") != cache_key:
            return None
        if checkpoint_sha256(features_path) != manifest.get("selected_features_sha256"):
            return None
        if checkpoint_sha256(indices_path) != manifest.get("selected_indices_sha256"):
            return None
        selected_features = np.load(features_path, allow_pickle=False)
        selected_indices = np.load(indices_path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if selected_indices.ndim != 1 or not np.issubdtype(selected_indices.dtype, np.integer):
        return None
    if selected_features.shape != (expected_rows, len(selected_indices)):
        return None
    if not np.isfinite(selected_features).all():
        return None
    indices = selected_indices.astype(np.int64, copy=False)
    if expected_indices is not None and not np.array_equal(indices, expected_indices):
        return None
    return TrackAFeatureCache(
        selected_features=np.asarray(selected_features, dtype=np.float64),
        selected_indices=indices,
        cache_key=cache_key,
        selected_features_sha256=str(manifest["selected_features_sha256"]),
        selected_indices_sha256=str(manifest["selected_indices_sha256"]),
        manifest_path=manifest_path,
    )


def prepare_track_a_rfecv_cache(
    artifacts: AuthorTrackAArtifacts,
    run_root: str | Path,
    *,
    expected_indices: Sequence[int] | np.ndarray | None = EXPECTED_TRACK_A_SELECTED_INDICES,
) -> TrackAFeatureCache:
    if not isinstance(artifacts, AuthorTrackAArtifacts):
        raise ValueError("artifacts must be AuthorTrackAArtifacts")
    expected = (
        None
        if expected_indices is None
        else np.asarray(expected_indices, dtype=np.int64)
    )
    payload = _track_a_rfecv_payload(artifacts)
    cache_key = _canonical_sha256(payload)
    cache_dir = Path(run_root) / "cache" / f"track_a_rfecv_{cache_key}"
    cached = _load_valid_track_a_cache(
        cache_dir,
        cache_key=cache_key,
        expected_rows=artifacts.features.shape[0],
        expected_indices=expected,
    )
    if cached is not None:
        return cached

    selection_train, _, labels_train, _ = train_test_split(
        artifacts.features,
        artifacts.author_labels,
        test_size=0.20,
        random_state=42,
    )
    selector = RFECV(
        estimator=ExtraTreesClassifier(n_estimators=50, random_state=0),
        step=1,
        cv=StratifiedKFold(10),
        scoring="roc_auc",
        min_features_to_select=1,
    )
    selector.fit(selection_train, labels_train)
    selected_indices = np.flatnonzero(np.asarray(selector.support_, dtype=bool)).astype(
        np.int64
    )
    if expected is not None and not np.array_equal(selected_indices, expected):
        raise RuntimeError(
            "RFECV selected indices differ from the pinned author reconstruction: "
            f"expected {expected.tolist()}, found {selected_indices.tolist()}"
        )
    selected_features = np.asarray(
        artifacts.features[:, selected_indices], dtype=np.float64
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    selected_features_sha256 = _atomic_numpy_save(
        selected_features, cache_dir / "selected_features.npy"
    )
    selected_indices_sha256 = _atomic_numpy_save(
        selected_indices, cache_dir / "selected_indices.npy"
    )
    manifest = {
        **payload,
        "cache_key": cache_key,
        "rfecv_scope": "single_global_author_pass",
        "selected_count": int(len(selected_indices)),
        "selected_indices": selected_indices.tolist(),
        "selected_features_shape": list(selected_features.shape),
        "selected_features_sha256": selected_features_sha256,
        "selected_indices_sha256": selected_indices_sha256,
    }
    _atomic_json_write(manifest, cache_dir / "manifest.json")
    return TrackAFeatureCache(
        selected_features=selected_features,
        selected_indices=selected_indices,
        cache_key=cache_key,
        selected_features_sha256=selected_features_sha256,
        selected_indices_sha256=selected_indices_sha256,
        manifest_path=cache_dir / "manifest.json",
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _configuration_payload(
    model_config: ModelConfig, train_config: TrainConfig
) -> dict[str, object]:
    return {
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }


def _validate_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character SHA256 hexadecimal digest")


def checkpoint_sha256(path: str | Path) -> str:
    checkpoint = Path(path)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_role(role: str) -> CheckpointRole:
    if role not in _CHECKPOINT_ROLES:
        raise ValueError(f"unknown checkpoint role: {role}")
    return role  # type: ignore[return-value]


def _manifest_path(directory: str | Path, role: str) -> Path:
    checked_role = _checkpoint_role(role)
    return Path(directory) / f"{checked_role}.manifest.json"


def _atomic_text_write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_checkpoint_graph(value: object) -> object:
    """Rebuild a payload with deterministic aliasing before torch serialization."""
    primitive_pool: dict[tuple[type[object], str], object] = {}

    def rebuild(item: object) -> object:
        if item is None or isinstance(item, (bool, int, float, str)):
            key = (type(item), repr(item))
            if key not in primitive_pool:
                if isinstance(item, bool) or item is None:
                    primitive_pool[key] = item
                elif isinstance(item, int):
                    primitive_pool[key] = int(str(item))
                elif isinstance(item, float):
                    primitive_pool[key] = float(repr(item))
                else:
                    primitive_pool[key] = str(item).encode("utf-8").decode("utf-8")
            return primitive_pool[key]
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().clone()
        if isinstance(item, np.ndarray):
            return item.copy()
        if isinstance(item, dict):
            return {rebuild(key): rebuild(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return tuple(rebuild(child) for child in item)
        if isinstance(item, list):
            return [rebuild(child) for child in item]
        return copy.deepcopy(item)

    return rebuild(value)


def _atomic_torch_save(payload: dict[str, object], path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(_canonical_checkpoint_graph(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint_sha256(destination)


def _atomic_json_write(payload: dict[str, object], path: Path) -> None:
    _atomic_text_write(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", path
    )


def _atomic_write_manifest(payload: dict[str, object], path: Path) -> None:
    _atomic_json_write(payload, path)


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix" or not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_manifest(directory: Path, role: CheckpointRole) -> dict[str, object] | None:
    path = _manifest_path(directory, role)
    if path.is_symlink() or not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("format_version") != 1
        or manifest.get("role") != role
    ):
        return None
    return manifest


def _validated_descriptor(
    value: object, *, role: CheckpointRole
) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    filename = value.get("filename")
    digest = value.get("sha256")
    epoch = value.get("epoch")
    generation = value.get("generation")
    kind = value.get("kind")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or not isinstance(digest, str)
        or _SHA256_PATTERN.fullmatch(digest) is None
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
        or kind != _CHECKPOINT_ROLES[role]
    ):
        return None
    normalized = {
        "filename": filename,
        "sha256": digest.lower(),
        "epoch": epoch,
        "generation": generation,
        "kind": kind,
    }
    expected_filename = (
        f"{role}-g{generation:06d}-e{epoch:06d}-{digest.lower()}.pt"
    )
    return normalized if filename == expected_filename else None


def _descriptor_has_valid_bytes(directory: Path, descriptor: object) -> bool:
    if not isinstance(descriptor, dict):
        return False
    filename = descriptor.get("filename")
    expected = descriptor.get("sha256")
    if not isinstance(filename, str) or not isinstance(expected, str):
        return False
    path = directory / filename
    if path.is_symlink() or not path.is_file():
        return False
    try:
        return checkpoint_sha256(path) == expected.lower()
    except OSError:
        return False


def _next_generation(directory: Path, role: CheckpointRole) -> int:
    pattern = re.compile(
        rf"^{re.escape(role)}-g(?P<generation>\d+)-e\d+-[0-9a-f]{{64}}\.pt$"
    )
    generations = [
        int(match.group("generation"))
        for path in directory.glob(f"{role}-g*-e*-*.pt")
        if (match := pattern.fullmatch(path.name)) is not None
    ]
    return max(generations, default=0) + 1


def _cleanup_checkpoint_generations(
    directory: Path,
    *,
    role: CheckpointRole,
    retained: tuple[dict[str, object] | None, ...],
) -> tuple[str, ...]:
    keep = {
        str(descriptor["filename"])
        for descriptor in retained
        if isinstance(descriptor, dict) and isinstance(descriptor.get("filename"), str)
    }
    pattern = re.compile(
        rf"^{re.escape(role)}-g\d+-e\d+-[0-9a-f]{{64}}\.pt$"
    )
    deferred: list[str] = []
    for path in sorted(directory.glob(f"{role}-g*-e*-*.pt")):
        if pattern.fullmatch(path.name) and path.name not in keep:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                if len(deferred) < _MAX_DEFERRED_CLEANUP_FILENAMES:
                    deferred.append(path.name)
    return tuple(deferred)


def _publish_checkpoint_generation(
    payload: dict[str, object],
    directory: str | Path,
    *,
    role: str,
    epoch: int,
) -> ResolvedCheckpoint:
    checked_role = _checkpoint_role(role)
    _require_integer("epoch", epoch, minimum=0)
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    generation = _next_generation(output_dir, checked_role)
    temporary = output_dir / f".{checked_role}-g{generation:06d}.pending.tmp"
    checkpoint_payload = dict(payload)
    existing_role = checkpoint_payload.get("checkpoint_role")
    if existing_role not in (None, checked_role):
        raise ValueError("checkpoint payload role conflicts with publication role")
    checkpoint_payload["checkpoint_role"] = checked_role
    try:
        with temporary.open("wb") as handle:
            torch.save(_canonical_checkpoint_graph(checkpoint_payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        provisional_digest = checkpoint_sha256(temporary)
        filename = (
            f"{checked_role}-g{generation:06d}-e{epoch:06d}-"
            f"{provisional_digest}.pt"
        )
        destination = output_dir / filename
        if destination.exists():
            if checkpoint_sha256(destination) != provisional_digest:
                raise RuntimeError("immutable checkpoint generation collision")
            temporary.unlink()
        else:
            os.replace(temporary, destination)
        with destination.open("rb+") as handle:
            os.fsync(handle.fileno())
        _fsync_directory(output_dir)
        final_digest = checkpoint_sha256(destination)
        if final_digest != provisional_digest:
            raise RuntimeError("checkpoint changed while publishing immutable generation")

        descriptor: dict[str, object] = {
            "filename": filename,
            "sha256": final_digest,
            "epoch": epoch,
            "generation": generation,
            "kind": _CHECKPOINT_ROLES[checked_role],
        }
        old_manifest = _read_manifest(output_dir, checked_role)
        previous: dict[str, object] | None = None
        if old_manifest is not None:
            for candidate_name in ("current", "previous"):
                candidate = _validated_descriptor(
                    old_manifest.get(candidate_name), role=checked_role
                )
                if candidate is not None and _descriptor_has_valid_bytes(
                    output_dir, candidate
                ):
                    previous = candidate
                    break
        manifest: dict[str, object] = {
            "format_version": 1,
            "role": checked_role,
            "current": descriptor,
            "previous": previous,
        }
        _atomic_write_manifest(manifest, _manifest_path(output_dir, checked_role))
        _fsync_directory(output_dir)
        deferred_cleanup = _cleanup_checkpoint_generations(
            output_dir, role=checked_role, retained=(descriptor, previous)
        )
        return ResolvedCheckpoint(
            path=destination,
            descriptor=descriptor,
            payload=checkpoint_payload,
            used_fallback=False,
            deferred_cleanup=deferred_cleanup,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_checkpoint_manifest(
    directory: str | Path,
    *,
    role: str,
    map_location: str | torch.device,
) -> ResolvedCheckpoint:
    checked_role = _checkpoint_role(role)
    output_dir = Path(directory)
    manifest = _read_manifest(output_dir, checked_role)
    if manifest is None:
        raise ValueError(f"checkpoint manifest is unavailable for {checked_role}")
    failures: list[str] = []
    for position, candidate_name in enumerate(("current", "previous")):
        descriptor = _validated_descriptor(
            manifest.get(candidate_name), role=checked_role
        )
        if descriptor is None:
            failures.append(f"{candidate_name}: invalid descriptor")
            continue
        path = output_dir / str(descriptor["filename"])
        if not _descriptor_has_valid_bytes(output_dir, descriptor):
            failures.append(f"{candidate_name}: missing or SHA256 mismatch")
            continue
        try:
            # Pickle is enabled only after validating immutable bytes against the
            # atomic manifest produced by this module. External checkpoints are
            # never accepted by this loader.
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except Exception as exc:
            failures.append(f"{candidate_name}: deserialization failed ({exc})")
            continue
        if not isinstance(payload, dict) or payload.get("checkpoint_role") != checked_role:
            failures.append(f"{candidate_name}: payload role/type mismatch")
            continue
        return ResolvedCheckpoint(
            path=path,
            descriptor=descriptor,
            payload=payload,
            used_fallback=position == 1,
        )
    raise ValueError(
        f"no valid {checked_role} checkpoint generation: {'; '.join(failures)}"
    )


def _preprocessor_state(preprocessor: FittedPreprocessor) -> dict[str, object]:
    imputer = preprocessor.imputer
    scaler = preprocessor.scaler
    return {
        "n_features": preprocessor.n_features,
        "imputer_strategy": str(imputer.strategy),
        "imputer_statistics": np.asarray(imputer.statistics_).tolist(),
        "scaler_mean": np.asarray(scaler.mean_).tolist(),
        "scaler_scale": np.asarray(scaler.scale_).tolist(),
        "scaler_var": np.asarray(scaler.var_).tolist(),
        "scaler_n_samples_seen": np.asarray(scaler.n_samples_seen_).tolist(),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_execution_backend(device: str | torch.device) -> str:
    try:
        requested = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid execution backend: {device}") from exc
    if requested.type == "cpu":
        return "cpu"
    if requested.type != "cuda":
        raise ValueError("execution backend must be CPU or CUDA")
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    index = requested.index
    if index is None:
        index = 0
    count = int(torch.cuda.device_count())
    if index < 0 or index >= count:
        raise ValueError(f"CUDA device index {index} is unavailable (device_count={count})")
    return f"cuda:{index}"


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: object) -> None:
    if not isinstance(state, dict):
        raise ValueError("checkpoint RNG state is invalid")
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]
    cuda_state = state.get("torch_cuda", [])
    if torch.cuda.is_available() and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)  # type: ignore[arg-type]


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _safe_validation_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    if np.unique(labels).size != 2 or not np.isfinite(probabilities).all():
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(labels, probabilities)),
        float(average_precision_score(labels, probabilities)),
    )


def author_threshold_sweep(
    author_labels: object, author_class_one_probability: object
) -> dict[str, object]:
    labels = _binary_labels(
        author_labels,
        expected_rows=len(np.asarray(author_class_one_probability)),
        name="author_labels",
    )
    probabilities = np.asarray(author_class_one_probability, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.shape[0] != labels.shape[0]:
        raise ValueError("author probabilities must be a one-dimensional aligned array")
    if not np.isfinite(probabilities).all() or bool(
        ((probabilities < 0.0) | (probabilities > 1.0)).any()
    ):
        raise ValueError("author probabilities must be finite and within [0, 1]")
    if np.unique(labels).size != 2:
        raise ValueError("author threshold sweep requires both classes")
    thresholds = np.arange(0.0, 1.0, 0.001)
    scores = np.asarray(
        [
            roc_auc_score(labels, (probabilities >= threshold).astype(np.int64))
            for threshold in thresholds
        ],
        dtype=np.float64,
    )
    index = int(np.argmax(scores))
    author_threshold = float(thresholds[index])
    author_prediction = (probabilities >= author_threshold).astype(np.int64)
    covid_labels = 1 - labels
    covid_prediction = 1 - author_prediction
    tn, fp, fn, tp = confusion_matrix(
        covid_labels, covid_prediction, labels=[0, 1]
    ).ravel()
    return {
        "author_threshold": author_threshold,
        "paper_thresholded_auc": float(scores[index]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def author_threshold_to_covid_threshold(author_threshold: float) -> float:
    if not math.isfinite(author_threshold) or not 0.0 <= author_threshold < 1.0:
        raise ValueError("author_threshold must be finite and within [0, 1)")
    boundary = 1.0 - float(author_threshold)
    if boundary >= 1.0:
        return 1.0
    return float(np.nextafter(boundary, 1.0))


def _selection_key(auroc: float, auprc: float) -> tuple[float, float]:
    return (
        auroc if math.isfinite(auroc) else -math.inf,
        auprc if math.isfinite(auprc) else -math.inf,
    )


def _predict_probabilities(
    model: NeuralDecisionClassifier,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            batch = torch.as_tensor(
                features[start : start + batch_size], dtype=torch.float32, device=device
            )
            probabilities.append(model(batch)[:, 1].detach().cpu().numpy())
    result = np.concatenate(probabilities).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError("model produced non-finite validation probabilities")
    return result


def _recovery_checkpoint_payload(
    *,
    model: NeuralDecisionClassifier,
    optimizer: torch.optim.Optimizer,
    completed_epoch: int,
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_auroc: float,
    best_auprc: float,
    early_stop_counter: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    preprocessor_state: dict[str, object],
    balancing_audit: dict[str, object],
    input_feature_hash: str,
    split_hash: str,
    code_revision: str,
    execution_backend: str,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "checkpoint_role": "latest_recovery",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "completed_epoch": completed_epoch,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_metrics": {"auroc": best_auroc, "auprc": best_auprc},
        "early_stop_counter": early_stop_counter,
        "rng_state": _rng_state(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "preprocessor_state": preprocessor_state,
        "balancing_audit": balancing_audit,
        "input_feature_hash": input_feature_hash,
        "split_hash": split_hash,
        "configuration_sha256": _canonical_sha256(
            _configuration_payload(model_config, train_config)
        ),
        "code_revision": code_revision,
        "execution_backend": execution_backend,
    }


def _best_inference_checkpoint_payload(
    *,
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_auroc: float,
    best_auprc: float,
    model_config: ModelConfig,
    train_config: TrainConfig,
    preprocessor_state: dict[str, object],
    balancing_audit: dict[str, object],
    input_feature_hash: str,
    split_hash: str,
    code_revision: str,
    execution_backend: str,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "checkpoint_role": "best_inference",
        "model_state": best_state,
        "best_epoch": best_epoch,
        "best_metrics": {"auroc": best_auroc, "auprc": best_auprc},
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "preprocessor_state": preprocessor_state,
        "balancing_audit": balancing_audit,
        "input_feature_hash": input_feature_hash,
        "split_hash": split_hash,
        "configuration_sha256": _canonical_sha256(
            _configuration_payload(model_config, train_config)
        ),
        "code_revision": code_revision,
        "execution_backend": execution_backend,
    }


def _validate_resume_payload(
    payload: dict[str, object],
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
    preprocessor_state: dict[str, object],
    balancing_audit: dict[str, object],
    input_feature_hash: str,
    split_hash: str,
    code_revision: str,
    execution_backend: str,
) -> None:
    expected = {
        "format_version": 1,
        "checkpoint_role": "latest_recovery",
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "preprocessor_state": preprocessor_state,
        "balancing_audit": balancing_audit,
        "input_feature_hash": input_feature_hash,
        "split_hash": split_hash,
        "configuration_sha256": _canonical_sha256(
            _configuration_payload(model_config, train_config)
        ),
        "code_revision": code_revision,
        "execution_backend": execution_backend,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        if "execution_backend" in mismatches:
            raise ValueError(
                "execution_backend mismatch: checkpoint="
                f"{payload.get('execution_backend')!r}, requested={execution_backend!r}"
            )
        raise ValueError(f"resume fingerprint/config/code mismatch: {mismatches}")


def _best_checkpoint_matches(
    actual: dict[str, object], expected: dict[str, object]
) -> bool:
    fingerprint_keys = (
        "format_version",
        "checkpoint_role",
        "best_epoch",
        "model_config",
        "train_config",
        "preprocessor_state",
        "balancing_audit",
        "input_feature_hash",
        "split_hash",
        "configuration_sha256",
        "code_revision",
        "execution_backend",
    )
    if any(actual.get(key) != expected.get(key) for key in fingerprint_keys):
        return False
    actual_metrics = actual.get("best_metrics")
    expected_metrics = expected.get("best_metrics")
    if not isinstance(actual_metrics, dict) or not isinstance(expected_metrics, dict):
        return False
    for metric in ("auroc", "auprc"):
        actual_value = float(actual_metrics.get(metric, float("nan")))
        expected_value = float(expected_metrics.get(metric, float("nan")))
        if math.isnan(actual_value) and math.isnan(expected_value):
            continue
        if actual_value != expected_value:
            return False
    return True


def fit_model(
    training_features: object,
    training_labels: object,
    validation_features: object,
    validation_labels: object,
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_dir: str | Path,
    input_feature_hash: str,
    split_hash: str,
    code_revision: str,
    device: str | torch.device = "cpu",
    resume: bool = False,
    interrupt_after_epoch: int | None = None,
    prediction_batch_size: int = 1024,
) -> FitResult:
    if not isinstance(model_config, ModelConfig):
        raise ValueError("model_config must be a ModelConfig")
    if not isinstance(train_config, TrainConfig):
        raise ValueError("train_config must be a TrainConfig")
    _validate_sha256("input_feature_hash", input_feature_hash)
    _validate_sha256("split_hash", split_hash)
    if not isinstance(code_revision, str) or not code_revision.strip():
        raise ValueError("code_revision must be a nonempty string")
    if not isinstance(resume, bool):
        raise ValueError("resume must be boolean")
    if interrupt_after_epoch is not None:
        _require_integer("interrupt_after_epoch", interrupt_after_epoch, minimum=1)
    _require_integer("prediction_batch_size", prediction_batch_size, minimum=1)

    train_matrix = _feature_matrix(training_features, name="training_features")
    validation_matrix = _feature_matrix(
        validation_features, name="validation_features"
    )
    if validation_matrix.shape[1] != train_matrix.shape[1]:
        raise ValueError("training and validation feature counts differ")
    train_labels = _binary_labels(
        training_labels, expected_rows=train_matrix.shape[0], name="training_labels"
    )
    validation_labels_array = _binary_labels(
        validation_labels,
        expected_rows=validation_matrix.shape[0],
        name="validation_labels",
    )
    if np.unique(train_labels).size != 2:
        raise ValueError("training_labels must contain both classes")

    execution_backend = normalize_execution_backend(device)
    torch_device = torch.device(execution_backend)

    preprocessor = fit_preprocessor(train_matrix)
    transformed_train = transform_features(preprocessor, train_matrix)
    transformed_validation = transform_features(preprocessor, validation_matrix)
    balanced = balance_training_rows(
        transformed_train,
        train_labels,
        method=train_config.balance_method,
        seed=train_config.seed,
    )
    preprocessing_state = _preprocessor_state(preprocessor)

    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_payload: dict[str, object] | None = None
    if resume:
        resolved_recovery = _resolve_checkpoint_manifest(
            output_dir, role="latest_recovery", map_location="cpu"
        )
        resume_payload = resolved_recovery.payload
        _validate_resume_payload(
            resume_payload,
            model_config=model_config,
            train_config=train_config,
            preprocessor_state=preprocessing_state,
            balancing_audit=balanced.audit,
            input_feature_hash=input_feature_hash,
            split_hash=split_hash,
            code_revision=code_revision,
            execution_backend=execution_backend,
        )

    _seed_everything(train_config.seed)
    model = NeuralDecisionClassifier(
        num_features=transformed_train.shape[1],
        model_config=model_config,
        seed=train_config.seed,
    ).to(torch_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    start_epoch = 1
    best_epoch = -1
    best_auroc = float("nan")
    best_auprc = float("nan")
    best_state: dict[str, torch.Tensor] | None = None
    early_stop_counter = 0
    completed_epoch = 0
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])  # type: ignore[arg-type]
        optimizer.load_state_dict(resume_payload["optimizer_state"])  # type: ignore[arg-type]
        _move_optimizer_state(optimizer, torch_device)
        completed_epoch = int(resume_payload["completed_epoch"])
        start_epoch = completed_epoch + 1
        best_epoch = int(resume_payload["best_epoch"])
        metrics = resume_payload["best_metrics"]
        if not isinstance(metrics, dict):
            raise ValueError("checkpoint best metrics are invalid")
        best_auroc = float(metrics["auroc"])
        best_auprc = float(metrics["auprc"])
        best_state = copy.deepcopy(resume_payload["best_state"])  # type: ignore[arg-type]
        early_stop_counter = int(resume_payload["early_stop_counter"])
        _restore_rng_state(resume_payload["rng_state"])
        if early_stop_counter >= train_config.patience:
            start_epoch = train_config.max_epochs + 1

    train_x_tensor = torch.as_tensor(
        balanced.features, dtype=torch.float32, device=torch_device
    )
    train_y_tensor = torch.as_tensor(
        balanced.labels, dtype=torch.long, device=torch_device
    )
    weight_tensor = (
        torch.as_tensor(balanced.sample_weights, dtype=torch.float32, device=torch_device)
        if balanced.sample_weights is not None
        else None
    )

    for epoch in range(start_epoch, train_config.max_epochs + 1):
        model.train()
        for indices in deterministic_batch_indices(
            len(balanced.labels),
            train_config.batch_size,
            seed=train_config.seed,
            epoch=epoch,
        ):
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=torch_device)
            optimizer.zero_grad(set_to_none=True)
            probability = model(train_x_tensor.index_select(0, index_tensor))
            per_row_loss = F.nll_loss(
                torch.log(probability.clamp_min(1e-7)),
                train_y_tensor.index_select(0, index_tensor),
                reduction="none",
            )
            if weight_tensor is None:
                loss = per_row_loss.mean()
            else:
                batch_weights = weight_tensor.index_select(0, index_tensor)
                loss = (per_row_loss * batch_weights).sum() / batch_weights.sum()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("training loss became non-finite")
            loss.backward()
            optimizer.step()

        validation_probability = _predict_probabilities(
            model,
            transformed_validation,
            device=torch_device,
            batch_size=prediction_batch_size,
        )
        epoch_auroc, epoch_auprc = _safe_validation_metrics(
            validation_labels_array, validation_probability
        )
        improved = best_state is None or _selection_key(
            epoch_auroc, epoch_auprc
        ) > _selection_key(best_auroc, best_auprc)
        if improved:
            best_epoch = epoch
            best_auroc = epoch_auroc
            best_auprc = epoch_auprc
            best_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if best_state is None:
            raise RuntimeError("training did not produce a model state")
        completed_epoch = epoch
        recovery_payload = _recovery_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            completed_epoch=epoch,
            best_state=best_state,
            best_epoch=best_epoch,
            best_auroc=best_auroc,
            best_auprc=best_auprc,
            early_stop_counter=early_stop_counter,
            model_config=model_config,
            train_config=train_config,
            preprocessor_state=preprocessing_state,
            balancing_audit=balanced.audit,
            input_feature_hash=input_feature_hash,
            split_hash=split_hash,
            code_revision=code_revision,
            execution_backend=execution_backend,
        )
        if improved:
            inference_payload = _best_inference_checkpoint_payload(
                best_state=best_state,
                best_epoch=best_epoch,
                best_auroc=best_auroc,
                best_auprc=best_auprc,
                model_config=model_config,
                train_config=train_config,
                preprocessor_state=preprocessing_state,
                balancing_audit=balanced.audit,
                input_feature_hash=input_feature_hash,
                split_hash=split_hash,
                code_revision=code_revision,
                execution_backend=execution_backend,
            )
            _publish_checkpoint_generation(
                inference_payload,
                output_dir,
                role="best_inference",
                epoch=best_epoch,
            )
        _publish_checkpoint_generation(
            recovery_payload,
            output_dir,
            role="latest_recovery",
            epoch=epoch,
        )

        if interrupt_after_epoch == epoch:
            raise PlannedInterruption(f"planned interruption after epoch {epoch}")
        if early_stop_counter >= train_config.patience:
            break

    if best_state is None or best_epoch < 1 or completed_epoch < 1:
        raise RuntimeError("no validation-selected checkpoint is available")
    model.load_state_dict(best_state)
    final_best_payload = _best_inference_checkpoint_payload(
        best_state=best_state,
        best_epoch=best_epoch,
        best_auroc=best_auroc,
        best_auprc=best_auprc,
        model_config=model_config,
        train_config=train_config,
        preprocessor_state=preprocessing_state,
        balancing_audit=balanced.audit,
        input_feature_hash=input_feature_hash,
        split_hash=split_hash,
        code_revision=code_revision,
        execution_backend=execution_backend,
    )
    try:
        best_checkpoint = _resolve_checkpoint_manifest(
            output_dir, role="best_inference", map_location="cpu"
        )
        if not _best_checkpoint_matches(best_checkpoint.payload, final_best_payload):
            raise ValueError("best inference checkpoint does not match selected epoch")
    except ValueError:
        best_checkpoint = _publish_checkpoint_generation(
            final_best_payload,
            output_dir,
            role="best_inference",
            epoch=best_epoch,
        )
    validation_probability = _predict_probabilities(
        model,
        transformed_validation,
        device=torch_device,
        batch_size=prediction_batch_size,
    )
    validation_auroc, validation_auprc = _safe_validation_metrics(
        validation_labels_array, validation_probability
    )
    threshold = best_threshold_by_balanced_accuracy(
        validation_labels_array, validation_probability
    )
    latest_checkpoint = _resolve_checkpoint_manifest(
        output_dir, role="latest_recovery", map_location="cpu"
    )
    best_hash = str(best_checkpoint.descriptor["sha256"])
    latest_hash = str(latest_checkpoint.descriptor["sha256"])
    _atomic_json_write(
        {
            "status": "complete",
            "best_epoch": best_epoch,
            "validation_auroc": (
                validation_auroc if math.isfinite(validation_auroc) else None
            ),
            "validation_auprc": (
                validation_auprc if math.isfinite(validation_auprc) else None
            ),
            "threshold": threshold,
            "threshold_source": "validation_balanced_accuracy",
            "best_checkpoint_sha256": best_hash,
            "latest_checkpoint_sha256": latest_hash,
            "configuration_sha256": _canonical_sha256(
                _configuration_payload(model_config, train_config)
            ),
            "input_feature_hash": input_feature_hash,
            "split_hash": split_hash,
            "code_revision": code_revision,
            "execution_backend": execution_backend,
            "balancing_audit": balanced.audit,
            "checkpoints": {
                "latest_recovery": {
                    **latest_checkpoint.descriptor,
                    "path": str(latest_checkpoint.path),
                    "role": "latest_recovery",
                },
                "best_inference": {
                    **best_checkpoint.descriptor,
                    "path": str(best_checkpoint.path),
                    "role": "best_inference",
                },
            },
        },
        output_dir / "completion.json",
    )
    return FitResult(
        best_epoch=best_epoch,
        validation_auroc=validation_auroc,
        validation_auprc=validation_auprc,
        threshold=float(threshold),
        checkpoint_path=best_checkpoint.path,
        validation_probability=validation_probability,
    )


@dataclass(frozen=True)
class _TrackACorrectedFit:
    model: NeuralDecisionClassifier
    threshold: float
    best_epoch: int
    validation_auroc: float
    validation_auprc: float
    checkpoint_sha256: str
    balancing_audit: dict[str, object]


def _create_track_a_model_optimizer(
    *,
    num_features: int,
    model_config: ModelConfig,
    learning_rate: float,
    seed: int,
    device: torch.device,
    mode: str,
) -> tuple[NeuralDecisionClassifier, torch.optim.Optimizer]:
    del mode
    _seed_everything(seed)
    model = NeuralDecisionClassifier(
        num_features=num_features,
        model_config=model_config,
        seed=seed,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def _track_a_train_epoch(
    model: NeuralDecisionClassifier,
    optimizer: torch.optim.Optimizer,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batches: Sequence[np.ndarray],
    device: torch.device,
) -> None:
    model.train()
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    label_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
    for indices in batches:
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        probability = model(feature_tensor.index_select(0, index_tensor))
        loss = F.nll_loss(
            torch.log(probability.clamp_min(1e-7)),
            label_tensor.index_select(0, index_tensor),
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Track A training loss became non-finite")
        loss.backward()
        optimizer.step()


def _track_a_corrected_fit_no_scaler(
    training_features: np.ndarray,
    training_author_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_covid_labels: np.ndarray,
    *,
    model_config: ModelConfig,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    reconstruction_seed: int,
    checkpoint_dir: Path,
    feature_sha256: str,
    split_sha256: str,
    code_revision: str,
    device: torch.device,
    mode: str,
    fold: int,
    resume: bool,
    interrupt_after_epoch: int | None,
) -> _TrackACorrectedFit:
    balanced = balance_training_rows(
        training_features,
        training_author_labels,
        method="svm_smote",
        seed=reconstruction_seed,
    )
    model, optimizer = _create_track_a_model_optimizer(
        num_features=training_features.shape[1],
        model_config=model_config,
        learning_rate=learning_rate,
        seed=reconstruction_seed,
        device=device,
        mode=mode,
    )
    fingerprint = {
        "mode": mode,
        "fold": fold,
        "model_config": asdict(model_config),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "reconstruction_seed": reconstruction_seed,
        "feature_sha256": feature_sha256,
        "split_sha256": split_sha256,
        "code_revision": code_revision,
        "execution_backend": str(device),
        "external_scaler": False,
    }
    start_epoch = 1
    best_epoch = -1
    best_auroc = float("nan")
    best_auprc = float("nan")
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    if resume and _manifest_path(checkpoint_dir, "latest_recovery").is_file():
        recovered = _resolve_checkpoint_manifest(
            checkpoint_dir, role="latest_recovery", map_location="cpu"
        )
        payload = recovered.payload
        if payload.get("track_a_fingerprint") != fingerprint:
            raise ValueError("corrected Track A resume fingerprint mismatch")
        model.load_state_dict(payload["model_state"])  # type: ignore[arg-type]
        optimizer.load_state_dict(payload["optimizer_state"])  # type: ignore[arg-type]
        _move_optimizer_state(optimizer, device)
        start_epoch = int(payload["completed_epoch"]) + 1
        best_epoch = int(payload["best_epoch"])
        best_auroc = float(payload["best_auroc"])
        best_auprc = float(payload["best_auprc"])
        best_state = copy.deepcopy(payload["best_state"])  # type: ignore[arg-type]
        no_improvement = int(payload["no_improvement"])
        _restore_rng_state(payload["rng_state"])

    for epoch in range(start_epoch, max_epochs + 1):
        batches = deterministic_batch_indices(
            len(balanced.labels),
            batch_size,
            seed=reconstruction_seed,
            epoch=epoch,
        )
        _track_a_train_epoch(
            model,
            optimizer,
            balanced.features,
            balanced.labels,
            batches=batches,
            device=device,
        )
        probability_n = _predict_probabilities(
            model, validation_features, device=device, batch_size=1024
        )
        probability_covid = 1.0 - probability_n
        epoch_auroc, epoch_auprc = _safe_validation_metrics(
            validation_covid_labels, probability_covid
        )
        improved = best_state is None or _selection_key(
            epoch_auroc, epoch_auprc
        ) > _selection_key(best_auroc, best_auprc)
        if improved:
            best_epoch = epoch
            best_auroc = epoch_auroc
            best_auprc = epoch_auprc
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        if best_state is None:
            raise RuntimeError("corrected Track A training produced no state")
        payload = {
            "format_version": 1,
            "checkpoint_role": "latest_recovery",
            "track_a_fingerprint": fingerprint,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "completed_epoch": epoch,
            "best_epoch": best_epoch,
            "best_auroc": best_auroc,
            "best_auprc": best_auprc,
            "best_state": best_state,
            "no_improvement": no_improvement,
            "rng_state": _rng_state(),
            "balancing_audit": balanced.audit,
        }
        _publish_checkpoint_generation(
            payload,
            checkpoint_dir,
            role="latest_recovery",
            epoch=epoch,
        )
        if interrupt_after_epoch == epoch:
            raise PlannedInterruption(
                f"planned Track A interruption at {mode}/{fold}/epoch {epoch}"
            )
        if no_improvement >= patience:
            break

    if best_state is None or best_epoch < 1:
        raise RuntimeError("corrected Track A has no validation-selected state")
    model.load_state_dict(best_state)
    inference = _publish_checkpoint_generation(
        {
            "format_version": 1,
            "checkpoint_role": "best_inference",
            "track_a_fingerprint": fingerprint,
            "model_state": best_state,
            "best_epoch": best_epoch,
            "best_auroc": best_auroc,
            "best_auprc": best_auprc,
            "balancing_audit": balanced.audit,
        },
        checkpoint_dir,
        role="best_inference",
        epoch=best_epoch,
    )
    validation_probability_covid = 1.0 - _predict_probabilities(
        model, validation_features, device=device, batch_size=1024
    )
    threshold = best_threshold_by_balanced_accuracy(
        validation_covid_labels, validation_probability_covid
    )
    return _TrackACorrectedFit(
        model=model,
        threshold=float(threshold),
        best_epoch=best_epoch,
        validation_auroc=best_auroc,
        validation_auprc=best_auprc,
        checkpoint_sha256=str(inference.descriptor["sha256"]),
        balancing_audit=balanced.audit,
    )


def _track_a_model_config(
    model_name: str, published: Mapping[str, object]
) -> ModelConfig:
    if model_name not in {"dndt", "dndf"}:
        raise ValueError(f"unknown Track A model: {model_name}")
    return ModelConfig(
        model_name,
        num_trees=(
            int(published["dndt_trees"])
            if model_name == "dndt"
            else int(published["dndf_trees"])
        ),
        depth=int(published["depth"]),
        used_features_rate=float(published["used_features_rate"]),
    )


def _atomic_dataframe_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _track_a_prediction_frame(
    *,
    run_id: str,
    mode: str,
    fold: int,
    seed: int,
    model_name: str,
    test_indices: np.ndarray,
    covid_labels: np.ndarray,
    covid_probability: np.ndarray,
    threshold: float,
    threshold_source: str,
    configuration_sha256: str,
    split_sha256: str,
    feature_sha256: str,
    checkpoint_sha: str,
) -> pd.DataFrame:
    analysis_ids = [f"author-sample-{index:04d}" for index in test_indices]
    rows = {
        "run_id": run_id,
        "track": "A",
        "protocol": f"author_released_10fold_{mode}",
        "fold": fold,
        "seed": seed,
        "dataset": "coswara_author_released_array",
        "split": "outer_test",
        "model_name": model_name,
        "modality": "cough",
        "participant_id": analysis_ids,
        "recording_id": analysis_ids,
        "label_binary": covid_labels.astype(np.int64),
        "probability": covid_probability.astype(np.float64),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "configuration_sha256": configuration_sha256,
        "split_sha256": split_sha256,
        "feature_sha256": feature_sha256,
        "checkpoint_sha256": checkpoint_sha,
        "analysis_id": analysis_ids,
        "analysis_unit": "author_sample",
    }
    return pd.DataFrame(rows, columns=list(PREDICTION_COLUMNS) + list(_TRACK_A_COLUMNS))


def _predict_track_a_outer_test(
    model: NeuralDecisionClassifier,
    selected_features: np.ndarray,
    test_indices: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    return _predict_probabilities(
        model,
        selected_features[test_indices],
        device=device,
        batch_size=1024,
    )


def _track_a_metric_row(
    *,
    frame: pd.DataFrame,
    mode: str,
    paper_thresholded_auc: float | None,
    author_threshold: float | None,
    confusion_override: Mapping[str, int] | None,
    best_epoch: int,
    validation_auroc: float | None,
    validation_auprc: float | None,
    reconstruction_seed: int,
    model_reinitialized: bool,
    checkpoint_sha: str,
    author_commit: str,
    feature_cache: TrackAFeatureCache,
    fold_hash: str,
    balancing_audit: Mapping[str, object],
    inner_split_seed: int | None = None,
    inner_validation_fraction: float | None = None,
) -> dict[str, object]:
    labels = frame["label_binary"].to_numpy(dtype=np.int64)
    probabilities = frame["probability"].to_numpy(dtype=np.float64)
    threshold = float(frame["threshold"].iloc[0])
    metrics = binary_metric_bundle(labels, probabilities, threshold=threshold)
    predicted = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    if confusion_override is not None:
        tn = int(confusion_override["tn"])
        fp = int(confusion_override["fp"])
        fn = int(confusion_override["fn"])
        tp = int(confusion_override["tp"])
        metrics["balanced_accuracy"] = 0.5 * (
            tp / max(1, tp + fn) + tn / max(1, tn + fp)
        )
        metrics["f1"] = 2.0 * tp / max(1, 2 * tp + fp + fn)
        metrics["sensitivity"] = tp / max(1, tp + fn)
        metrics["specificity"] = tn / max(1, tn + fp)
    return {
        **metrics,
        "run_id": str(frame["run_id"].iloc[0]),
        "track": "A",
        "analysis_id": f"{mode}:{frame['model_name'].iloc[0]}:fold-{int(frame['fold'].iloc[0])}",
        "analysis_unit": "author_sample",
        "mode": mode,
        "model_name": str(frame["model_name"].iloc[0]),
        "fold": int(frame["fold"].iloc[0]),
        "paper_thresholded_auc": paper_thresholded_auc,
        "author_threshold": author_threshold,
        "threshold_source": str(frame["threshold_source"].iloc[0]),
        "threshold_selected_on_outer_test": mode == "author_behaviour_audit",
        "model_reinitialized_per_fold": model_reinitialized,
        "optimizer_reinitialized_per_fold": model_reinitialized,
        "author_training_order": (
            "no_shuffle_repeated_dataset"
            if mode == "author_behaviour_audit"
            else "deterministic_per_epoch_shuffle"
        ),
        "author_randomness_unseeded": True,
        "reconstruction_seed": reconstruction_seed,
        "outer_test_evaluation_count": 1,
        "inner_split_seed": inner_split_seed,
        "inner_validation_fraction": inner_validation_fraction,
        "best_epoch": best_epoch,
        "validation_auroc": validation_auroc,
        "validation_auprc": validation_auprc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "checkpoint_sha256": checkpoint_sha,
        "feature_sha256": feature_cache.selected_features_sha256,
        "feature_indices_sha256": feature_cache.selected_indices_sha256,
        "fold_sha256": fold_hash,
        "author_commit": author_commit,
        "global_feature_selection_retained": True,
        "released_preprocessed_array_retained": True,
        "external_standard_scaler": False,
        "balancing_audit": dict(balancing_audit),
    }


def _write_track_a_fold_outputs(
    fold_dir: Path,
    predictions: pd.DataFrame,
    metric: dict[str, object],
    *,
    feature_cache: TrackAFeatureCache,
    author_commit: str,
    configuration_sha256: str,
    split_sha256: str,
    code_revision: str,
) -> None:
    predictions_path = fold_dir / "predictions.csv"
    metrics_path = fold_dir / "metrics.json"
    write_predictions(predictions, predictions_path)
    serializable_metric = {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in metric.items()
    }
    _atomic_json_write(serializable_metric, metrics_path)
    _atomic_json_write(
        {
            "status": "complete",
            "mode": metric["mode"],
            "model_name": metric["model_name"],
            "fold": metric["fold"],
            "feature_sha256": feature_cache.selected_features_sha256,
            "feature_indices_sha256": feature_cache.selected_indices_sha256,
            "fold_sha256": split_sha256,
            "configuration_sha256": configuration_sha256,
            "checkpoint_sha256": metric["checkpoint_sha256"],
            "predictions_sha256": checkpoint_sha256(predictions_path),
            "metrics_sha256": checkpoint_sha256(metrics_path),
            "code_revision": code_revision,
            "author_commit": author_commit,
            "global_feature_selection_retained": True,
            "released_preprocessed_array_retained": True,
            "author_training_order": metric["author_training_order"],
            "author_randomness_unseeded": True,
            "reconstruction_seed": metric["reconstruction_seed"],
        },
        fold_dir / "receipt.json",
    )


def _load_completed_track_a_fold(
    fold_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]] | None:
    receipt_path = fold_dir / "receipt.json"
    predictions_path = fold_dir / "predictions.csv"
    metrics_path = fold_dir / "metrics.json"
    if not (receipt_path.is_file() and predictions_path.is_file() and metrics_path.is_file()):
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if checkpoint_sha256(predictions_path) != receipt.get("predictions_sha256"):
            return None
        if checkpoint_sha256(metrics_path) != receipt.get("metrics_sha256"):
            return None
        metric = json.loads(metrics_path.read_text(encoding="utf-8"))
        predictions = validate_prediction_frame(pd.read_csv(predictions_path))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if receipt.get("status") != "complete":
        return None
    return predictions, metric


def _validated_track_a_fold_batch(
    fold_batch: Sequence[int] | None, fold_count: int
) -> tuple[int, ...]:
    selected = tuple(range(fold_count)) if fold_batch is None else tuple(fold_batch)
    if not selected:
        raise ValueError("fold_batch must select at least one fold")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in selected):
        raise ValueError("fold_batch must contain integer folds")
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("fold_batch must be sorted and unique")
    if selected[0] < 0 or selected[-1] >= fold_count:
        raise ValueError(f"fold_batch values must be within 0..{fold_count - 1}")
    return selected


def run_track_a(
    config: Mapping[str, object],
    *,
    run_id: str,
    resume: bool = False,
    smoke: bool = False,
    fold_batch: Sequence[int] | None = None,
    modes: Sequence[str] = ("author_behaviour_audit", "corrected_reference"),
    model_names: Sequence[str] = ("dndt", "dndf"),
    artifacts: AuthorTrackAArtifacts | None = None,
    feature_cache: TrackAFeatureCache | None = None,
    code_revision: str,
    device: str | torch.device | None = None,
    interrupt_after: tuple[str, str, int, int] | None = None,
) -> dict[str, object]:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a nonempty string")
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    run_root_value = config.get("run_root")
    if not isinstance(run_root_value, str) or not run_root_value:
        raise ValueError("config run_root must be a nonempty path")
    run_dir = Path(run_root_value) / run_id
    requested_modes = tuple(modes)
    valid_modes = {"author_behaviour_audit", "corrected_reference"}
    if not requested_modes or set(requested_modes) - valid_modes:
        raise ValueError("Track A modes are invalid")
    requested_models = tuple(model_names)
    if not requested_models or set(requested_models) - {"dndt", "dndf"}:
        raise ValueError("Track A model_names are invalid")

    provisional_folds = tuple(range(10)) if fold_batch is None else tuple(fold_batch)
    if (
        "author_behaviour_audit" in requested_modes
        and provisional_folds
        and provisional_folds[0] > 0
        and not resume
    ):
        raise ValueError(
            "author_behaviour_audit cannot start a later fold batch without valid preceding state"
        )

    if artifacts is None:
        author_repo = config.get("author_repo")
        expected_commit = config.get("author_commit", TRACK_A_AUTHOR_COMMIT)
        if not isinstance(author_repo, str) or not author_repo:
            raise ValueError("config author_repo must be a nonempty path")
        if not isinstance(expected_commit, str) or not expected_commit:
            raise ValueError("config author_commit must be a nonempty string")
        artifacts = load_track_a_author_artifacts(author_repo, expected_commit)
    folds = _validated_track_a_fold_batch(
        (0,) if smoke else fold_batch, len(artifacts.test_folds)
    )
    if feature_cache is None:
        feature_cache = prepare_track_a_rfecv_cache(artifacts, Path(run_root_value))

    published_value = config.get("published")
    if not isinstance(published_value, Mapping):
        raise ValueError("config published must be a mapping")
    published = dict(published_value)
    required_published = {
        "depth",
        "used_features_rate",
        "learning_rate",
        "batch_size",
        "epochs",
        "dndt_trees",
        "dndf_trees",
    }
    if required_published - set(published):
        raise ValueError("config published parameters are incomplete")
    if smoke:
        published.update(
            {
                "depth": 2,
                "dndt_trees": 1,
                "dndf_trees": 2,
                "epochs": 2,
            }
        )
    selection = config.get("selection", {})
    if not isinstance(selection, Mapping):
        raise ValueError("config selection must be a mapping")
    patience = int(selection.get("patience", 3))
    seeds = config.get("seeds", {})
    if not isinstance(seeds, Mapping):
        raise ValueError("config seeds must be a mapping")
    candidate = seeds.get("candidate", [42])
    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)) or not candidate:
        raise ValueError("config candidate seeds must be a nonempty sequence")
    reconstruction_seed = int(candidate[0])
    execution_backend = normalize_execution_backend(device or str(config.get("device", "cuda")))
    torch_device = torch.device(execution_backend)
    run_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict[str, object]] = []
    total_units = len(requested_modes) * len(requested_models) * len(folds)
    completed_units = 0

    for mode in requested_modes:
        for model_name in requested_models:
            model_config = _track_a_model_config(model_name, published)
            mode_config = {
                "mode": mode,
                "model_name": model_name,
                "model_config": asdict(model_config),
                "published": published,
                "patience": patience,
                "reconstruction_seed": reconstruction_seed,
                "author_training_order": (
                    "no_shuffle_repeated_dataset"
                    if mode == "author_behaviour_audit"
                    else "deterministic_per_epoch_shuffle"
                ),
                "author_randomness_unseeded": True,
                "feature_cache_key": feature_cache.cache_key,
                "code_revision": code_revision,
            }
            configuration_sha = _canonical_sha256(mode_config)
            state_dir = run_dir / "track_a" / mode / model_name / "state"

            if mode == "author_behaviour_audit":
                model, optimizer = _create_track_a_model_optimizer(
                    num_features=feature_cache.selected_features.shape[1],
                    model_config=model_config,
                    learning_rate=float(published["learning_rate"]),
                    seed=reconstruction_seed,
                    device=torch_device,
                    mode=mode,
                )
                next_fold = 0
                resume_epoch = 0
                phase = "between_folds"
                if resume and _manifest_path(state_dir, "latest_recovery").is_file():
                    recovered = _resolve_checkpoint_manifest(
                        state_dir, role="latest_recovery", map_location="cpu"
                    )
                    state = recovered.payload
                    if state.get("track_a_configuration_sha256") != configuration_sha:
                        raise ValueError("author Track A resume configuration mismatch")
                    model.load_state_dict(state["model_state"])  # type: ignore[arg-type]
                    optimizer.load_state_dict(state["optimizer_state"])  # type: ignore[arg-type]
                    _move_optimizer_state(optimizer, torch_device)
                    next_fold = int(state["next_fold"])
                    resume_epoch = int(state["completed_epoch"])
                    phase = str(state["phase"])
                    _restore_rng_state(state["rng_state"])
                if next_fold < folds[0] or next_fold > folds[-1] + 1:
                    raise ValueError(
                        "author_behaviour_audit fold batch lacks the exact valid preceding state/receipt: "
                        f"state points to fold {next_fold}, batch is {folds[0]}..{folds[-1]}"
                    )

                for fold in folds:
                    if fold < next_fold:
                        completed = _load_completed_track_a_fold(
                            run_dir / "track_a" / mode / model_name / f"fold_{fold:02d}"
                        )
                        if completed is None:
                            raise ValueError(
                                "author Track A state is ahead of a missing or invalid fold receipt"
                            )
                        predictions, metric = completed
                        all_predictions.append(predictions)
                        all_metrics.append(metric)
                        completed_units += 1
                        continue
                    if fold != next_fold:
                        raise ValueError("author mode folds must continue without gaps")
                    train_indices = artifacts.train_folds[fold]
                    fold_seed = reconstruction_seed + fold
                    balanced = balance_training_rows(
                        feature_cache.selected_features[train_indices],
                        artifacts.author_labels[train_indices],
                        method="svm_smote",
                        seed=fold_seed,
                    )
                    start_epoch = resume_epoch + 1 if phase == "training" else 1
                    if phase == "trained":
                        start_epoch = int(published["epochs"]) + 1
                    for epoch in range(start_epoch, int(published["epochs"]) + 1):
                        _track_a_train_epoch(
                            model,
                            optimizer,
                            balanced.features,
                            balanced.labels,
                            batches=fixed_order_batch_indices(
                                len(balanced.labels), int(published["batch_size"])
                            ),
                            device=torch_device,
                        )
                        phase_after = (
                            "trained" if epoch == int(published["epochs"]) else "training"
                        )
                        recovered = _publish_checkpoint_generation(
                            {
                                "format_version": 1,
                                "checkpoint_role": "latest_recovery",
                                "track_a_configuration_sha256": configuration_sha,
                                "model_state": model.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "next_fold": fold,
                                "completed_epoch": epoch,
                                "phase": phase_after,
                                "rng_state": _rng_state(),
                                "fold_seed": fold_seed,
                                "balancing_audit": balanced.audit,
                            },
                            state_dir,
                            role="latest_recovery",
                            epoch=fold * int(published["epochs"]) + epoch,
                        )
                        if interrupt_after == (mode, model_name, fold, epoch):
                            raise PlannedInterruption(
                                f"planned Track A interruption at {mode}/{model_name}/{fold}/{epoch}"
                            )
                    trained_state = _resolve_checkpoint_manifest(
                        state_dir, role="latest_recovery", map_location="cpu"
                    )
                    test_indices = artifacts.test_folds[fold]
                    probability_n = _predict_track_a_outer_test(
                        model,
                        feature_cache.selected_features,
                        test_indices,
                        device=torch_device,
                    )
                    threshold_audit = author_threshold_sweep(
                        artifacts.author_labels[test_indices], probability_n
                    )
                    covid_probability = 1.0 - probability_n
                    public_threshold = author_threshold_to_covid_threshold(
                        float(threshold_audit["author_threshold"])
                    )
                    split_sha = _canonical_sha256(
                        {
                            "fold": fold,
                            "train_hash": artifacts.fold_hashes.get(
                                f"Train-Test Split/coswaradataset/train/{fold}.csv",
                                _canonical_sha256(train_indices.tolist()),
                            ),
                            "test_hash": artifacts.fold_hashes.get(
                                f"Train-Test Split/coswaradataset/test/{fold}.csv",
                                _canonical_sha256(test_indices.tolist()),
                            ),
                        }
                    )
                    predictions = _track_a_prediction_frame(
                        run_id=run_id,
                        mode=mode,
                        fold=fold,
                        seed=reconstruction_seed,
                        model_name=model_name,
                        test_indices=test_indices,
                        covid_labels=artifacts.covid_labels[test_indices],
                        covid_probability=covid_probability,
                        threshold=public_threshold,
                        threshold_source="test_balanced_accuracy_author_audit",
                        configuration_sha256=configuration_sha,
                        split_sha256=split_sha,
                        feature_sha256=feature_cache.selected_features_sha256,
                        checkpoint_sha=str(trained_state.descriptor["sha256"]),
                    )
                    metric = _track_a_metric_row(
                        frame=predictions,
                        mode=mode,
                        paper_thresholded_auc=float(
                            threshold_audit["paper_thresholded_auc"]
                        ),
                        author_threshold=float(threshold_audit["author_threshold"]),
                        confusion_override=threshold_audit,  # type: ignore[arg-type]
                        best_epoch=int(published["epochs"]),
                        validation_auroc=None,
                        validation_auprc=None,
                        reconstruction_seed=reconstruction_seed,
                        model_reinitialized=False,
                        checkpoint_sha=str(trained_state.descriptor["sha256"]),
                        author_commit=artifacts.author_commit,
                        feature_cache=feature_cache,
                        fold_hash=split_sha,
                        balancing_audit=balanced.audit,
                    )
                    fold_dir = run_dir / "track_a" / mode / model_name / f"fold_{fold:02d}"
                    _write_track_a_fold_outputs(
                        fold_dir,
                        predictions,
                        metric,
                        feature_cache=feature_cache,
                        author_commit=artifacts.author_commit,
                        configuration_sha256=configuration_sha,
                        split_sha256=split_sha,
                        code_revision=code_revision,
                    )
                    _publish_checkpoint_generation(
                        {
                            "format_version": 1,
                            "checkpoint_role": "latest_recovery",
                            "track_a_configuration_sha256": configuration_sha,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "next_fold": fold + 1,
                            "completed_epoch": 0,
                            "phase": "between_folds",
                            "rng_state": _rng_state(),
                            "completed_receipt": str(fold_dir / "receipt.json"),
                        },
                        state_dir,
                        role="latest_recovery",
                        epoch=(fold + 1) * int(published["epochs"]),
                    )
                    next_fold = fold + 1
                    resume_epoch = 0
                    phase = "between_folds"
                    all_predictions.append(predictions)
                    all_metrics.append(metric)
                    completed_units += 1
            else:
                for fold in folds:
                    fold_dir = run_dir / "track_a" / mode / model_name / f"fold_{fold:02d}"
                    completed = _load_completed_track_a_fold(fold_dir) if resume else None
                    if completed is not None:
                        predictions, metric = completed
                        all_predictions.append(predictions)
                        all_metrics.append(metric)
                        completed_units += 1
                        continue
                    outer_train = artifacts.train_folds[fold]
                    inner_train, inner_validation = train_test_split(
                        outer_train,
                        test_size=0.125,
                        random_state=reconstruction_seed + fold,
                        stratify=artifacts.author_labels[outer_train],
                    )
                    split_sha = _canonical_sha256(
                        {
                            "fold": fold,
                            "outer_train": outer_train.tolist(),
                            "inner_train": np.asarray(inner_train).tolist(),
                            "inner_validation": np.asarray(inner_validation).tolist(),
                            "outer_test_hash": artifacts.fold_hashes.get(
                                f"Train-Test Split/coswaradataset/test/{fold}.csv",
                                _canonical_sha256(artifacts.test_folds[fold].tolist()),
                            ),
                        }
                    )
                    requested_interrupt = (
                        interrupt_after[3]
                        if interrupt_after is not None
                        and interrupt_after[:3] == (mode, model_name, fold)
                        else None
                    )
                    fitted = _track_a_corrected_fit_no_scaler(
                        feature_cache.selected_features[np.asarray(inner_train)],
                        artifacts.author_labels[np.asarray(inner_train)],
                        feature_cache.selected_features[np.asarray(inner_validation)],
                        artifacts.covid_labels[np.asarray(inner_validation)],
                        model_config=model_config,
                        learning_rate=float(published["learning_rate"]),
                        batch_size=int(published["batch_size"]),
                        max_epochs=int(published["epochs"]),
                        patience=patience,
                        reconstruction_seed=reconstruction_seed + fold,
                        checkpoint_dir=fold_dir / "checkpoints",
                        feature_sha256=feature_cache.selected_features_sha256,
                        split_sha256=split_sha,
                        code_revision=code_revision,
                        device=torch_device,
                        mode=mode,
                        fold=fold,
                        resume=resume,
                        interrupt_after_epoch=requested_interrupt,
                    )
                    test_indices = artifacts.test_folds[fold]
                    probability_n = _predict_track_a_outer_test(
                        fitted.model,
                        feature_cache.selected_features,
                        test_indices,
                        device=torch_device,
                    )
                    predictions = _track_a_prediction_frame(
                        run_id=run_id,
                        mode=mode,
                        fold=fold,
                        seed=reconstruction_seed + fold,
                        model_name=model_name,
                        test_indices=test_indices,
                        covid_labels=artifacts.covid_labels[test_indices],
                        covid_probability=1.0 - probability_n,
                        threshold=fitted.threshold,
                        threshold_source="inner_validation_balanced_accuracy",
                        configuration_sha256=configuration_sha,
                        split_sha256=split_sha,
                        feature_sha256=feature_cache.selected_features_sha256,
                        checkpoint_sha=fitted.checkpoint_sha256,
                    )
                    metric = _track_a_metric_row(
                        frame=predictions,
                        mode=mode,
                        paper_thresholded_auc=None,
                        author_threshold=None,
                        confusion_override=None,
                        best_epoch=fitted.best_epoch,
                        validation_auroc=fitted.validation_auroc,
                        validation_auprc=fitted.validation_auprc,
                        reconstruction_seed=reconstruction_seed + fold,
                        model_reinitialized=True,
                        checkpoint_sha=fitted.checkpoint_sha256,
                        author_commit=artifacts.author_commit,
                        feature_cache=feature_cache,
                        fold_hash=split_sha,
                        balancing_audit=fitted.balancing_audit,
                        inner_split_seed=reconstruction_seed + fold,
                        inner_validation_fraction=0.125,
                    )
                    _write_track_a_fold_outputs(
                        fold_dir,
                        predictions,
                        metric,
                        feature_cache=feature_cache,
                        author_commit=artifacts.author_commit,
                        configuration_sha256=configuration_sha,
                        split_sha256=split_sha,
                        code_revision=code_revision,
                    )
                    all_predictions.append(predictions)
                    all_metrics.append(metric)
                    completed_units += 1

            progress = {
                "status": "running",
                "run_id": run_id,
                "completed_units": completed_units,
                "total_units": total_units,
                "last_mode": mode,
                "last_model": model_name,
                "fold_batch": list(folds),
            }
            _atomic_json_write(progress, run_dir / "progress.json")

    prediction_frame = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metric_frame = pd.DataFrame(all_metrics)
    if not prediction_frame.empty:
        write_predictions(prediction_frame, run_dir / "track_a_predictions.csv")
    if not metric_frame.empty:
        serializable_metrics = metric_frame.copy()
        if "balancing_audit" in serializable_metrics:
            serializable_metrics["balancing_audit"] = serializable_metrics[
                "balancing_audit"
            ].map(lambda value: json.dumps(value, sort_keys=True))
        _atomic_dataframe_write(serializable_metrics, run_dir / "track_a_metrics.csv")
    final = {
        "status": "complete",
        "run_id": run_id,
        "completed_units": completed_units,
        "total_units": total_units,
        "fold_batch": list(folds),
        "modes": list(requested_modes),
        "model_names": list(requested_models),
        "author_commit": artifacts.author_commit,
        "feature_cache_key": feature_cache.cache_key,
        "author_training_order": "no_shuffle_repeated_dataset",
        "author_randomness_unseeded": True,
        "reconstruction_seed": reconstruction_seed,
        "metrics": metric_frame.to_dict(orient="records"),
        "predictions": prediction_frame.to_dict(orient="records"),
    }
    _atomic_json_write(
        {key: value for key, value in final.items() if key not in {"metrics", "predictions"}},
        run_dir / "progress.json",
    )
    return final


def aggregate_participant_probabilities(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"participant_id", "recording_id", "label_binary", "probability"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"participant aggregation is missing columns: {missing}")
    if predictions.empty:
        return pd.DataFrame(
            columns=["participant_id", "label_binary", "probability", "n_recordings"]
        )
    frame = predictions.copy()
    missing_participant = frame["participant_id"].isna() | (
        frame["participant_id"].astype(str).str.strip().eq("")
    )
    if missing_participant.any():
        raise ValueError("participant_id must be present for participant aggregation")
    missing_recording = frame["recording_id"].isna() | (
        frame["recording_id"].astype(str).str.strip().eq("")
    )
    if missing_recording.any():
        raise ValueError("recording_id must be present for participant aggregation")

    try:
        frame["probability"] = pd.to_numeric(frame["probability"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("probability must be numeric") from exc
    probability = frame["probability"].to_numpy(dtype=float)
    if not np.isfinite(probability).all():
        raise ValueError("probability must be finite")
    if bool(((probability < 0.0) | (probability > 1.0)).any()):
        raise ValueError("probability must be within [0, 1]")

    label_tokens: list[object] = []
    missing_label = np.zeros(len(frame), dtype=bool)
    invalid_label = np.zeros(len(frame), dtype=bool)
    for index, value in enumerate(frame["label_binary"].tolist()):
        is_missing = pd.isna(value)
        if isinstance(is_missing, (bool, np.bool_)) and bool(is_missing):
            missing_label[index] = True
            label_tokens.append("__missing_label__")
        elif value in (0, "negative"):
            label_tokens.append(0)
        elif value in (1, "positive"):
            label_tokens.append(1)
        else:
            invalid_label[index] = True
            label_tokens.append("__invalid_label__")
    frame["_label_token"] = label_tokens

    conflict = frame.groupby("participant_id", dropna=False)["_label_token"].nunique(
        dropna=False
    )
    if bool((conflict > 1).any()):
        ids = conflict[conflict > 1].index.astype(str).tolist()
        raise ValueError(f"conflicting labels within participant: {ids[:10]}")
    if missing_label.any():
        raise ValueError("label_binary must not contain null values")
    if invalid_label.any():
        raise ValueError("label_binary must contain only binary labels")

    duplicate_recording = frame.duplicated(
        subset=["participant_id", "recording_id"], keep=False
    )
    if duplicate_recording.any():
        raise ValueError("duplicate recording rows would bias the participant mean")
    frame = frame.sort_values(["participant_id", "recording_id"], kind="stable")
    result = (
        frame.groupby("participant_id", as_index=False, sort=True)
        .agg(
            label_binary=("label_binary", "first"),
            probability=("probability", "mean"),
            n_recordings=("recording_id", "nunique"),
        )
        .sort_values("participant_id", kind="stable")
        .reset_index(drop=True)
    )
    if bool((result["n_recordings"] <= 0).any()):
        raise RuntimeError("participant aggregation produced zero valid recordings")
    return result


def _is_track_a(frame: pd.DataFrame) -> bool:
    values = frame["track"].astype(str).str.strip().str.upper()
    return bool(values.isin(["A", "TRACK_A"]).any())


def validate_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("predictions must be a pandas DataFrame")
    track_a = "track" in predictions.columns and _is_track_a(predictions)
    expected = list(PREDICTION_COLUMNS) + (list(_TRACK_A_COLUMNS) if track_a else [])
    if predictions.columns.tolist() != expected:
        raise ValueError(f"prediction columns must exactly equal {expected}")
    if predictions.empty:
        return predictions.copy()
    if track_a:
        if not predictions["analysis_unit"].astype(str).eq("author_sample").all():
            raise ValueError("Track A analysis_unit must be author_sample")
        missing_analysis = predictions["analysis_id"].isna() | (
            predictions["analysis_id"].astype(str).str.strip().eq("")
        )
        if missing_analysis.any():
            raise ValueError("Track A analysis_id must be present")
    else:
        missing_participant = predictions["participant_id"].isna() | (
            predictions["participant_id"].astype(str).str.strip().eq("")
        )
        if missing_participant.any():
            raise ValueError("participant_id must be present outside Track A")
    for column in (
        "configuration_sha256",
        "split_sha256",
        "feature_sha256",
        "checkpoint_sha256",
    ):
        if not predictions[column].astype(str).map(
            lambda value: _SHA256_PATTERN.fullmatch(value) is not None
        ).all():
            raise ValueError(f"{column} must contain SHA256 digests")
    probability = pd.to_numeric(predictions["probability"], errors="raise").to_numpy(dtype=float)
    threshold = pd.to_numeric(predictions["threshold"], errors="raise").to_numpy(dtype=float)
    invalid_probability = (probability < 0.0) | (probability > 1.0)
    if not np.isfinite(probability).all() or bool(invalid_probability.any()):
        raise ValueError("probability must be finite and within [0, 1]")
    if not np.isfinite(threshold).all() or bool(((threshold < 0.0) | (threshold > 1.0)).any()):
        raise ValueError("threshold must be finite and within [0, 1]")
    if not predictions["label_binary"].isin([0, 1, "positive", "negative"]).all():
        raise ValueError("label_binary must contain binary labels")
    return predictions.copy()


def write_predictions(predictions: pd.DataFrame, path: str | Path) -> Path:
    validated = validate_prediction_frame(predictions)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            validated.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "AuthorTrackAArtifacts",
    "BalanceResult",
    "EXPECTED_TRACK_A_SELECTED_INDICES",
    "FitResult",
    "FittedPreprocessor",
    "PREDICTION_COLUMNS",
    "PlannedInterruption",
    "TrackAFeatureCache",
    "TrainConfig",
    "aggregate_participant_probabilities",
    "author_threshold_to_covid_threshold",
    "author_threshold_sweep",
    "balance_training_rows",
    "checkpoint_sha256",
    "deterministic_batch_indices",
    "fixed_order_batch_indices",
    "fit_model",
    "fit_preprocessor",
    "load_track_a_author_artifacts",
    "prepare_track_a_rfecv_cache",
    "run_track_a",
    "transform_features",
    "validate_prediction_frame",
    "write_predictions",
]
