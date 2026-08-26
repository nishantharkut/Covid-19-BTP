from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import time
import uuid
from contextlib import contextmanager
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from covid_rars.dndt_dndf_models import ModelConfig, NeuralDecisionClassifier
from covid_rars.compare_is10_rescue import rank_train_features
from covid_rars.features import feature_columns
from covid_rars.fusion import uniform_fusion
from covid_rars.metrics import binary_metric_bundle, best_threshold_by_balanced_accuracy
from covid_rars.dndt_dndf_evidence import complete_metric_bundle
from covid_rars.temporal_holdout import (
    _apply_split_to_features,
    build_temporal_split_assignments,
    build_time_stratified_split_assignments,
)


BalanceMethod = Literal["svm_smote", "smote", "class_weight"]
CheckpointRole = Literal["latest_recovery", "best_inference"]
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_CHECKPOINT_ROLES: dict[str, str] = {
    "latest_recovery": "recovery",
    "best_inference": "inference",
}
_MAX_DEFERRED_CLEANUP_FILENAMES = 32
_RFECV_LOCK_TIMEOUT_SECONDS = 300.0
_RFECV_LOCK_RETRY_SECONDS = 0.05
SMOTE_K_NEIGHBORS = 5
SVMSMOTE_K_NEIGHBORS = 5
SVMSMOTE_M_NEIGHBORS = 10
TRACK_B_MODALITIES = ("breath", "cough", "speech")
TRACK_B_PROTOCOLS = ("existing", "time_stratified", "early_to_late", "external_cough")
TRACK_B_SPLITS = ("train", "validation", "test")
TRACK_B_IDENTITY_COLUMNS = (
    "recording_id",
    "participant_id",
    "dataset",
    "modality",
    "label_binary",
    "split",
)
TRACK_B_OPTIONAL_IDENTITY_COLUMNS = ("submodality",)
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
_TRACK_A_COLUMNS = (
    "analysis_id",
    "analysis_unit",
    "mode",
    "author_commit",
    "feature_indices_sha256",
    "feature_cache_key",
    "code_revision",
    "execution_backend",
    "predicted_label",
    "threshold_comparator",
)


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


@dataclass(frozen=True)
class TrackAFoldContext:
    run_id: str
    mode: str
    fold: int
    model_name: str
    author_commit: str
    feature_sha256: str
    feature_indices_sha256: str
    feature_cache_key: str
    configuration_sha256: str
    split_sha256: str
    code_revision: str
    execution_backend: str
    threshold_source: str
    threshold_comparator: str
    reconstruction_seed: int
    checkpoint_path: Path
    checkpoint_sha256: str

    @property
    def protocol(self) -> str:
        return f"author_released_10fold_{self.mode}"

    @property
    def author_training_order(self) -> str:
        return (
            "no_shuffle_repeated_dataset"
            if self.mode in {"author_behaviour_audit", "fresh_fold_author_protocol"}
            else "deterministic_per_epoch_shuffle"
        )


@dataclass(frozen=True)
class TrackBData:
    project: pd.DataFrame
    external: pd.DataFrame
    metadata: pd.DataFrame
    feature_columns: tuple[str, ...]
    feature_sha256: str
    source_sha256: dict[str, str]


@dataclass(frozen=True)
class TrackBProtocol:
    name: str
    source: pd.DataFrame
    target: pd.DataFrame
    participant_assignments: pd.DataFrame
    split_summary: pd.DataFrame
    split_sha256: str
    source_lineage: str


@dataclass(frozen=True)
class CandidateSelectionResult:
    modality: str
    completed_trials: int
    observed_splits: tuple[str, ...]
    selected_dndf: dict[str, object]
    overall_winner: str
    candidates: tuple[dict[str, object], ...]


def _track_b_checkpoint_input_sha256(
    feature_sha256: str, source_sha256: Mapping[str, str]
) -> str:
    _validate_sha256("feature_sha256", feature_sha256)
    source_names = set(source_sha256)
    if not {"project", "metadata"} <= source_names or not source_names <= {
        "project",
        "external",
        "metadata",
    }:
        raise ValueError(
            "Track B source hashes must identify project and metadata, with "
            "external required only for external-transfer contexts"
        )
    for name, value in source_sha256.items():
        _validate_sha256(f"{name} source SHA256", str(value))
    return _canonical_sha256(
        {
            "feature_sha256": feature_sha256,
            "source_sha256": dict(sorted(source_sha256.items())),
        }
    )


class PlannedInterruption(RuntimeError):
    """Raised by tests or controllers only after an epoch checkpoint is durable."""


def _canonical_dataframe_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].astype(str)
    if len(normalized.columns):
        normalized = normalized.sort_values(list(normalized.columns), kind="stable")
    return _canonical_sha256(normalized.to_dict(orient="records"))


def _track_b_label_token(value: object) -> str | None:
    if value in (1, "1", "positive"):
        return "positive"
    if value in (0, "0", "negative"):
        return "negative"
    return None


def _validate_track_b_participant_labels(frame: pd.DataFrame, name: str) -> None:
    missing_id = frame["participant_id"].isna() | frame["participant_id"].astype(
        str
    ).str.strip().eq("")
    if missing_id.any():
        raise ValueError(f"{name} participant_id must be present")
    conflicts = (
        frame.groupby("participant_id", dropna=False)["label_binary"]
        .nunique(dropna=False)
        .loc[lambda values: values > 1]
    )
    if not conflicts.empty:
        raise ValueError(
            f"conflicting labels within participant in {name}: "
            f"{conflicts.index.astype(str).tolist()[:10]}"
        )


def _validated_track_b_table(
    frame: pd.DataFrame,
    *,
    name: str,
    expected_features: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"{name} contains duplicate columns: {duplicates}")
    missing = sorted(set(TRACK_B_IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required identity columns: {missing}")

    identity = set(TRACK_B_IDENTITY_COLUMNS) | set(TRACK_B_OPTIONAL_IDENTITY_COLUMNS)
    ordered_candidates = tuple(column for column in frame.columns if column not in identity)
    if len(ordered_candidates) != 800:
        raise ValueError(
            f"{name} must contain exactly 800 ordered feature columns; "
            f"found {len(ordered_candidates)}"
        )
    if expected_features is not None and ordered_candidates != tuple(expected_features):
        raise ValueError("project and external ordered 800 feature columns do not match")

    checked = frame.copy()
    for column in ordered_candidates:
        try:
            checked[column] = pd.to_numeric(checked[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} feature column {column!r} contains nonnumeric values"
            ) from exc
    discovered_features = tuple(feature_columns(checked))
    if discovered_features != ordered_candidates:
        raise ValueError(f"{name} ordered feature columns are not exactly numeric")
    matrix = checked.loc[:, ordered_candidates].to_numpy(dtype=np.float64)
    if np.isinf(matrix).any():
        raise ValueError(f"{name} contains nonfinite feature values")
    all_missing = np.flatnonzero(np.isnan(matrix).all(axis=0)).tolist()
    if all_missing:
        names = [ordered_candidates[index] for index in all_missing[:10]]
        raise ValueError(f"{name} contains all-missing feature columns: {names}")

    normalized_labels = checked["label_binary"].map(_track_b_label_token)
    checked = checked.loc[normalized_labels.notna()].copy()
    checked["label_binary"] = normalized_labels.loc[checked.index]
    if checked.empty:
        raise ValueError(f"{name} contains no labelled positive/negative rows")
    _validate_track_b_participant_labels(checked, name)
    return checked.reset_index(drop=True), ordered_candidates


def load_track_b_data(
    project_path: str | Path,
    external_path: str | Path,
    metadata_path: str | Path,
    *,
    read_csv: Callable[..., pd.DataFrame] = pd.read_csv,
) -> TrackBData:
    paths = {
        "project": Path(project_path),
        "external": Path(external_path),
        "metadata": Path(metadata_path),
    }
    loaded = {
        name: read_csv(path, low_memory=False) for name, path in paths.items()
    }
    project, ordered_features = _validated_track_b_table(
        loaded["project"], name="project"
    )
    external, external_features = _validated_track_b_table(
        loaded["external"],
        name="external",
        expected_features=ordered_features,
    )
    if external_features != ordered_features:
        raise RuntimeError("validated external feature order unexpectedly changed")
    if not project["dataset"].astype(str).str.casefold().eq("coswara").all():
        raise ValueError("project rows must all identify the Coswara dataset")
    if not external["dataset"].astype(str).str.casefold().eq("coughvid").all():
        raise ValueError("external rows must all identify the COUGHVID dataset")
    if not external["modality"].astype(str).eq("cough").all():
        raise ValueError("external COUGHVID rows must be cough-only")
    external = external.copy()
    external["split"] = "external"

    metadata = loaded["metadata"].copy()
    required_metadata = {"participant_id", "label_binary", "split", "recording_date"}
    missing_metadata = sorted(required_metadata - set(metadata.columns))
    if missing_metadata:
        raise ValueError(f"metadata is missing required columns: {missing_metadata}")
    metadata_labels = metadata["label_binary"].map(_track_b_label_token)
    metadata = metadata.loc[metadata_labels.notna()].copy()
    metadata["label_binary"] = metadata_labels.loc[metadata.index]
    _validate_track_b_participant_labels(metadata, "metadata")

    project_labels = project.groupby("participant_id")["label_binary"].first()
    metadata_labels_by_participant = metadata.groupby("participant_id")[
        "label_binary"
    ].first()
    shared = project_labels.index.intersection(metadata_labels_by_participant.index)
    disagree = shared[
        project_labels.loc[shared].to_numpy()
        != metadata_labels_by_participant.loc[shared].to_numpy()
    ]
    if len(disagree):
        raise ValueError(
            "conflicting labels between project and metadata participants: "
            f"{disagree.astype(str).tolist()[:10]}"
        )

    project = project[project["split"].isin(TRACK_B_SPLITS)].reset_index(drop=True)
    if project.empty:
        raise ValueError("project contains no labelled rows in required splits")
    source_sha = {name: checkpoint_sha256(path) for name, path in paths.items()}
    return TrackBData(
        project=project,
        external=external.reset_index(drop=True),
        metadata=metadata.reset_index(drop=True),
        feature_columns=ordered_features,
        feature_sha256=_canonical_sha256(list(ordered_features)),
        source_sha256=source_sha,
    )


def _track_b_existing_assignments(project: pd.DataFrame) -> pd.DataFrame:
    participant_splits = project.groupby("participant_id", dropna=False)["split"].agg(
        lambda values: tuple(sorted(set(values.astype(str))))
    )
    conflicts = participant_splits.loc[lambda values: values.map(len) != 1]
    if not conflicts.empty:
        raise ValueError(
            "existing split has participant overlap across train/validation/test: "
            f"{conflicts.index.astype(str).tolist()[:10]}"
        )
    return pd.DataFrame(
        {
            "participant_id": participant_splits.index.astype(str),
            "split": participant_splits.map(lambda values: values[0]).to_numpy(),
        }
    ).sort_values("participant_id", kind="stable").reset_index(drop=True)


def _enforce_track_b_split_isolation(frame: pd.DataFrame, protocol: str) -> None:
    assignments = _track_b_existing_assignments(frame)
    for first_index, first in enumerate(TRACK_B_SPLITS):
        first_ids = set(assignments.loc[assignments["split"].eq(first), "participant_id"])
        for second in TRACK_B_SPLITS[first_index + 1 :]:
            second_ids = set(
                assignments.loc[assignments["split"].eq(second), "participant_id"]
            )
            overlap = sorted(first_ids & second_ids)
            if overlap:
                raise ValueError(
                    f"{protocol} participant overlap between {first} and {second}: "
                    f"{overlap[:10]}"
                )
    for modality in TRACK_B_MODALITIES:
        modality_rows = frame[frame["modality"].astype(str).eq(modality)]
        missing = [
            split
            for split in TRACK_B_SPLITS
            if modality_rows.loc[modality_rows["split"].eq(split)].empty
        ]
        if missing:
            raise ValueError(
                f"{protocol} modality={modality} is missing required splits: {missing}"
            )


def _track_b_split_summary(
    source: pd.DataFrame, target: pd.DataFrame, protocol: str
) -> pd.DataFrame:
    combined = pd.concat([source, target], ignore_index=True, sort=False)
    rows: list[dict[str, object]] = []
    for keys, group in combined.groupby(
        ["dataset", "modality", "split"], sort=True, dropna=False
    ):
        dataset, modality, split = keys
        rows.append(
            {
                "protocol": protocol,
                "dataset": str(dataset),
                "modality": str(modality),
                "split": str(split),
                "n_rows": int(len(group)),
                "n_participants": int(group["participant_id"].nunique()),
                "n_positive": int(group["label_binary"].eq("positive").sum()),
                "n_negative": int(group["label_binary"].eq("negative").sum()),
            }
        )
    return pd.DataFrame(rows)


def build_track_b_protocols(
    data: TrackBData, *, output_dir: str | Path | None = None
) -> dict[str, TrackBProtocol]:
    if not isinstance(data, TrackBData):
        raise ValueError("data must be a TrackBData instance")
    existing = data.project.copy()
    existing_assignments = _track_b_existing_assignments(existing)

    time_assignments, _ = build_time_stratified_split_assignments(
        data.metadata,
        train_fraction=0.6,
        validation_fraction=0.2,
        date_column="recording_date",
        random_state=42,
    )
    time_stratified = _apply_split_to_features(
        data.project, time_assignments, "time_stratified_split"
    )
    time_stratified = time_stratified[
        time_stratified["split"].isin(TRACK_B_SPLITS)
    ].reset_index(drop=True)

    temporal_assignments, _ = build_temporal_split_assignments(
        data.metadata,
        train_fraction=0.6,
        validation_fraction=0.2,
        date_column="recording_date",
    )
    early_to_late = _apply_split_to_features(
        data.project, temporal_assignments, "temporal_split"
    )
    early_to_late = early_to_late[
        early_to_late["split"].isin(TRACK_B_SPLITS)
    ].reset_index(drop=True)

    sources = {
        "existing": existing,
        "time_stratified": time_stratified,
        "early_to_late": early_to_late,
        "external_cough": existing[
            existing["modality"].astype(str).eq("cough")
        ].reset_index(drop=True),
    }
    protocols: dict[str, TrackBProtocol] = {}
    for name, source in sources.items():
        if name != "external_cough":
            _enforce_track_b_split_isolation(source, name)
        target = (
            data.external.copy()
            if name == "external_cough"
            else source.loc[source["split"].eq("test")].copy()
        )
        protocol_source = (
            source.loc[source["split"].isin(("train", "validation"))].copy()
            if name == "external_cough"
            else source
        )
        summary_source = source.loc[
            source["split"].isin(("train", "validation"))
        ].copy()
        assignments = _track_b_existing_assignments(source)
        if name == "external_cough":
            target_assignments = data.external[["participant_id", "split"]].copy()
            assignments = pd.concat(
                [assignments, target_assignments], ignore_index=True, sort=False
            ).sort_values(["split", "participant_id"], kind="stable").reset_index(
                drop=True
            )
        summary = _track_b_split_summary(summary_source, target, name)
        protocol = TrackBProtocol(
            name=name,
            source=protocol_source.reset_index(drop=True),
            target=target.reset_index(drop=True),
            participant_assignments=assignments,
            split_summary=summary,
            split_sha256=_canonical_dataframe_sha256(assignments),
            source_lineage=("coswara_to_coughvid" if name == "external_cough" else "coswara"),
        )
        protocols[name] = protocol

    if output_dir is not None:
        split_dir = Path(output_dir) / "splits"
        manifest: dict[str, object] = {"protocols": {}}
        for name, protocol in protocols.items():
            assignments_path = split_dir / f"{name}_assignments.csv"
            summary_path = split_dir / f"{name}_summary.csv"
            _atomic_dataframe_write(protocol.participant_assignments, assignments_path)
            _atomic_dataframe_write(protocol.split_summary, summary_path)
            manifest["protocols"][name] = {  # type: ignore[index]
                "split_sha256": protocol.split_sha256,
                "assignments_sha256": checkpoint_sha256(assignments_path),
                "summary_sha256": checkpoint_sha256(summary_path),
                "source_lineage": protocol.source_lineage,
            }
        _atomic_json_write(manifest, split_dir / "manifest.json")
    return protocols


def _load_track_b_full_project_features(
    path: str | Path,
    *,
    base_data: TrackBData,
    read_csv: Callable[..., pd.DataFrame] = pd.read_csv,
) -> tuple[pd.DataFrame, tuple[str, ...], str]:
    source_path = Path(path)
    frame = read_csv(source_path, low_memory=False)
    required = set(TRACK_B_IDENTITY_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"full project feature table is missing identity columns: {missing}")
    normalized = frame["label_binary"].map(_track_b_label_token)
    frame = frame.loc[normalized.notna()].copy()
    frame["label_binary"] = normalized.loc[frame.index]
    if not frame["dataset"].astype(str).str.casefold().eq("coswara").all():
        raise ValueError("full project feature rows must identify Coswara")
    _validate_track_b_participant_labels(frame, "full project features")
    full_features = tuple(feature_columns(frame))
    if len(full_features) < 800:
        raise ValueError(
            "full project feature table must contain at least 800 numeric features"
        )
    for column in full_features:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    matrix = frame.loc[:, full_features].to_numpy(dtype=np.float64)
    if np.isinf(matrix).any():
        raise ValueError("full project feature table contains infinite values")
    if np.isnan(matrix).all(axis=0).any():
        raise ValueError("full project feature table contains all-missing features")

    identity_columns = [
        *TRACK_B_IDENTITY_COLUMNS,
        *(
            column
            for column in TRACK_B_OPTIONAL_IDENTITY_COLUMNS
            if column in frame.columns
        ),
    ]
    frame = frame.loc[:, [*identity_columns, *full_features]].copy()
    duplicate_recording = frame.duplicated(["recording_id"], keep=False)
    if duplicate_recording.any():
        raise ValueError("full project feature table contains duplicate recordings")
    base_identity = base_data.project.loc[
        :, ["recording_id", "participant_id", "modality", "label_binary"]
    ].copy()
    full_identity = frame.loc[
        :, ["recording_id", "participant_id", "modality", "label_binary"]
    ].copy()
    if (
        len(base_identity) != len(full_identity)
        or base_identity.astype(str)
        .sort_values(list(base_identity.columns), kind="stable")
        .reset_index(drop=True)
        .to_dict(orient="records")
        != full_identity.astype(str)
        .sort_values(list(full_identity.columns), kind="stable")
        .reset_index(drop=True)
        .to_dict(orient="records")
    ):
        raise ValueError(
            "full and frozen project feature tables do not have identical recording lineage"
        )
    return frame.reset_index(drop=True), full_features, checkpoint_sha256(source_path)


def _default_track_b_protocol_feature_ranker(training: pd.DataFrame) -> pd.DataFrame:
    return rank_train_features(
        training,
        ranker="lightgbm",
        selection_scope="per_modality_mean",
        random_state=42,
    )


def _prepare_track_b_temporal_feature_context(
    *,
    protocol: TrackBProtocol,
    full_project: pd.DataFrame,
    full_feature_columns: tuple[str, ...],
    full_project_sha256: str,
    base_data: TrackBData,
    track_b_dir: Path,
    code_revision: str,
    feature_ranker: Callable[[pd.DataFrame], pd.DataFrame],
) -> tuple[TrackBProtocol, TrackBData, str]:
    if protocol.name not in {"early_to_late", "time_stratified"}:
        raise ValueError("train-only feature selection is only defined for temporal protocols")
    assignments = protocol.participant_assignments.loc[
        :, ["participant_id", "split"]
    ].rename(columns={"split": "protocol_split"})
    assigned = full_project.drop(columns=["split"]).merge(
        assignments,
        on="participant_id",
        how="left",
        validate="many_to_one",
    )
    assigned["split"] = assigned.pop("protocol_split").fillna("unused")
    assigned = assigned[assigned["split"].isin(TRACK_B_SPLITS)].reset_index(drop=True)
    _enforce_track_b_split_isolation(assigned, protocol.name)
    training = assigned[assigned["split"].astype(str).eq("train")].copy()
    if training.empty or set(training["split"].astype(str)) != {"train"}:
        raise ValueError("temporal feature ranking requires train rows only")
    test_participants = set(
        assigned.loc[assigned["split"].astype(str).eq("test"), "participant_id"].astype(str)
    )
    if set(training["participant_id"].astype(str)) & test_participants:
        raise ValueError("temporal feature ranking contains held-out participants")

    feature_dir = track_b_dir / "protocol_features" / protocol.name
    ranking_path = feature_dir / "ranking.csv"
    manifest_path = feature_dir / "selection.json"
    expected_identity = {
        "format_version": 1,
        "protocol": protocol.name,
        "selection_scope": "protocol_train_only",
        "selection_method": "lightgbm_per_modality_mean",
        "top_k": 800,
        "full_project_sha256": full_project_sha256,
        "split_sha256": protocol.split_sha256,
        "code_revision": code_revision,
        "training_participants_sha256": _canonical_sha256(
            sorted(training["participant_id"].astype(str).unique())
        ),
    }
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("temporal feature selection manifest cannot be read") from exc
        mismatches = [
            key for key, value in expected_identity.items() if manifest.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                f"temporal feature selection lineage mismatch: {mismatches}"
            )
        selected_features = manifest.get("selected_features")
        if (
            not isinstance(selected_features, list)
            or len(selected_features) != 800
            or len(set(selected_features)) != 800
            or not set(selected_features) <= set(full_feature_columns)
            or manifest.get("selected_features_sha256")
            != _canonical_sha256(selected_features)
            or not ranking_path.is_file()
            or checkpoint_sha256(ranking_path) != manifest.get("ranking_sha256")
        ):
            raise ValueError("temporal feature selection artifact is invalid")
        selected = tuple(str(value) for value in selected_features)
    else:
        ranking = feature_ranker(training)
        if not isinstance(ranking, pd.DataFrame) or not {
            "feature",
            "importance",
        } <= set(ranking.columns):
            raise ValueError("temporal feature ranker returned an invalid table")
        ranking = ranking.copy()
        ranking["feature"] = ranking["feature"].astype(str)
        if (
            ranking["feature"].duplicated().any()
            or not set(ranking["feature"]) <= set(full_feature_columns)
            or len(ranking) < 800
        ):
            raise ValueError("temporal feature ranking is incomplete or invalid")
        if "rank" in ranking:
            ranking = ranking.sort_values("rank", kind="stable")
        selected = tuple(ranking.head(800)["feature"].tolist())
        _atomic_dataframe_write(ranking.reset_index(drop=True), ranking_path)
        manifest = {
            **expected_identity,
            "selected_features": list(selected),
            "selected_features_sha256": _canonical_sha256(list(selected)),
            "ranking_sha256": checkpoint_sha256(ranking_path),
        }
        _atomic_json_write(manifest, manifest_path)
    selection_sha = checkpoint_sha256(manifest_path)
    identity_columns = [
        *TRACK_B_IDENTITY_COLUMNS,
        *(
            column
            for column in TRACK_B_OPTIONAL_IDENTITY_COLUMNS
            if column in assigned.columns
        ),
    ]
    selected_project = assigned.loc[:, [*identity_columns, *selected]].copy()
    selected_data = TrackBData(
        project=selected_project,
        external=pd.DataFrame(columns=[*identity_columns, *selected]),
        metadata=base_data.metadata,
        feature_columns=selected,
        feature_sha256=_canonical_sha256(list(selected)),
        source_sha256={
            "project": full_project_sha256,
            "metadata": base_data.source_sha256["metadata"],
        },
    )
    selected_protocol = TrackBProtocol(
        name=protocol.name,
        source=selected_project,
        target=selected_project[
            selected_project["split"].astype(str).eq("test")
        ].reset_index(drop=True),
        participant_assignments=protocol.participant_assignments,
        split_summary=protocol.split_summary,
        split_sha256=protocol.split_sha256,
        source_lineage="coswara_protocol_train_only_top800",
    )
    return selected_protocol, selected_data, selection_sha


def _build_track_b_shuffle_protocol(
    existing: TrackBProtocol, *, seed: int
) -> TrackBProtocol:
    if existing.name != "existing":
        raise ValueError("shuffle control must be derived from the existing protocol")
    source = existing.source.copy()
    mapping_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for split in ("train", "validation"):
        participant_labels = (
            source.loc[source["split"].astype(str).eq(split)]
            .groupby("participant_id", sort=True)["label_binary"]
            .first()
        )
        if participant_labels.empty or participant_labels.nunique() != 2:
            raise ValueError(f"shuffle control requires two source classes in {split}")
        original = participant_labels.to_numpy(dtype=object)
        permuted = rng.permutation(original)
        if len(original) > 1 and np.array_equal(original, permuted):
            permuted = np.roll(permuted, 1)
        label_map = dict(zip(participant_labels.index.astype(str), permuted))
        mask = source["split"].astype(str).eq(split)
        source.loc[mask, "label_binary"] = (
            source.loc[mask, "participant_id"].astype(str).map(label_map).to_numpy()
        )
        mapping_rows.extend(
            {
                "participant_id": str(participant_id),
                "split": split,
                "original_label": str(original_label),
                "permuted_label": str(permuted_label),
                "permutation_seed": int(seed),
            }
            for participant_id, original_label, permuted_label in zip(
                participant_labels.index, original, permuted
            )
        )
    _validate_track_b_participant_labels(source, "shuffle source")
    target = existing.target.copy()
    assignments = pd.DataFrame(mapping_rows).sort_values(
        ["split", "participant_id"], kind="stable"
    ).reset_index(drop=True)
    summary = _track_b_split_summary(
        source.loc[source["split"].isin(("train", "validation"))],
        target,
        "shuffle",
    )
    return TrackBProtocol(
        name="shuffle",
        source=source.reset_index(drop=True),
        target=target.reset_index(drop=True),
        participant_assignments=assignments,
        split_summary=summary,
        split_sha256=_canonical_dataframe_sha256(assignments),
        source_lineage="coswara_participant_label_permutation_control",
    )


def _track_b_labels_to_binary(labels: pd.Series) -> np.ndarray:
    normalized = labels.map(_track_b_label_token)
    if normalized.isna().any():
        raise ValueError("Track B labels must contain only positive/negative values")
    return normalized.eq("positive").to_numpy(dtype=np.int64)


def _track_b_candidate_unit_dir(output_dir: Path, modality: str, candidate_id: str) -> Path:
    return output_dir / modality / candidate_id


def _track_b_candidate_receipt_path(
    output_dir: Path, modality: str, candidate_id: str
) -> Path:
    return _track_b_candidate_unit_dir(output_dir, modality, candidate_id) / "completion.json"


def _validated_track_b_candidate_evaluation(
    value: Mapping[str, object], *, candidate_id: str
) -> dict[str, object]:
    try:
        auroc = float(value["auroc"])
        auprc = float(value["auprc"])
        threshold = float(value["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"candidate evaluator returned invalid metrics for {candidate_id}") from exc
    if not math.isfinite(auroc) or not math.isfinite(auprc):
        raise ValueError(f"candidate evaluator returned nonfinite ranking metrics for {candidate_id}")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"candidate evaluator returned an invalid threshold for {candidate_id}")
    artifacts = value.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("candidate artifacts must be a mapping")
    return {
        "auroc": auroc,
        "auprc": auprc,
        "threshold": threshold,
        "artifacts": dict(artifacts),
    }


def _track_b_validation_prediction_frames(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(validation) != len(probabilities):
        raise ValueError("validation probability row count mismatch")
    recording = validation[
        [
            "recording_id",
            "participant_id",
            "dataset",
            "modality",
            "label_binary",
            "split",
        ]
    ].copy()
    recording["probability"] = np.asarray(probabilities, dtype=np.float64)
    recording["threshold"] = float(threshold)
    recording["analysis_unit"] = "recording"
    participant = aggregate_participant_probabilities(recording)
    participant["recording_id"] = participant["participant_id"]
    participant["dataset"] = str(recording["dataset"].iloc[0])
    participant["modality"] = str(recording["modality"].iloc[0])
    participant["split"] = str(recording["split"].iloc[0])
    participant["threshold"] = float(threshold)
    participant["analysis_unit"] = "participant"
    participant = participant[
        [
            "recording_id",
            "participant_id",
            "dataset",
            "modality",
            "label_binary",
            "split",
            "probability",
            "threshold",
            "analysis_unit",
            "n_recordings",
        ]
    ]
    return recording.reset_index(drop=True), participant.reset_index(drop=True)


def _default_track_b_candidate_evaluator(
    *,
    feature_columns_value: tuple[str, ...],
    feature_sha256: str,
    checkpoint_input_sha256: str,
    split_sha256: str,
    code_revision: str,
    device: str,
) -> Callable[
    [ModelConfig, TrainConfig, pd.DataFrame, pd.DataFrame, Path, bool],
    dict[str, object],
]:
    def evaluate(
        model_config: ModelConfig,
        train_config: TrainConfig,
        training: pd.DataFrame,
        validation: pd.DataFrame,
        unit_dir: Path,
        resume: bool,
    ) -> dict[str, object]:
        checkpoint_dir = unit_dir / "checkpoints"
        can_resume = resume and _manifest_path(
            checkpoint_dir, "latest_recovery"
        ).is_file()
        fitted = fit_model(
            training.loc[:, feature_columns_value].to_numpy(dtype=np.float64),
            _track_b_labels_to_binary(training["label_binary"]),
            validation.loc[:, feature_columns_value].to_numpy(dtype=np.float64),
            _track_b_labels_to_binary(validation["label_binary"]),
            model_config=model_config,
            train_config=train_config,
            checkpoint_dir=checkpoint_dir,
            input_feature_hash=checkpoint_input_sha256,
            split_hash=split_sha256,
            code_revision=code_revision,
            device=device,
            resume=can_resume,
            validation_group_ids=validation["participant_id"].astype(str).to_numpy(),
        )
        recording, participant = _track_b_validation_prediction_frames(
            validation,
            fitted.validation_probability,
            threshold=fitted.threshold,
        )
        participant_labels = _track_b_labels_to_binary(participant["label_binary"])
        participant_probability = participant["probability"].to_numpy(dtype=np.float64)
        threshold = best_threshold_by_balanced_accuracy(
            participant_labels, participant_probability
        )
        recording["threshold"] = threshold
        participant["threshold"] = threshold
        metrics = complete_metric_bundle(
            participant_labels, participant_probability, threshold=threshold
        )
        metrics.update(
            {
                "analysis_unit": "participant",
                "best_epoch": fitted.best_epoch,
                "validation_recordings": int(len(recording)),
                "validation_participants": int(len(participant)),
            }
        )
        recording_path = unit_dir / "recording_predictions.csv"
        participant_path = unit_dir / "participant_predictions.csv"
        metrics_path = unit_dir / "metrics.json"
        _atomic_dataframe_write(recording, recording_path)
        _atomic_dataframe_write(participant, participant_path)
        _atomic_json_write(metrics, metrics_path)
        return {
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "threshold": threshold,
            "artifacts": {
                "recording_predictions": {
                    "path": str(recording_path),
                    "sha256": checkpoint_sha256(recording_path),
                },
                "participant_predictions": {
                    "path": str(participant_path),
                    "sha256": checkpoint_sha256(participant_path),
                },
                "metrics": {
                    "path": str(metrics_path),
                    "sha256": checkpoint_sha256(metrics_path),
                },
                "checkpoint": {
                    "path": str(fitted.checkpoint_path),
                    "sha256": checkpoint_sha256(fitted.checkpoint_path),
                },
            },
        }

    return evaluate


def _track_b_candidate_receipt_expected(
    *,
    modality: str,
    candidate_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    feature_columns_value: tuple[str, ...],
    feature_sha256: str,
    split_sha256: str,
    source_sha256: Mapping[str, str],
    code_revision: str,
    execution_backend: str,
) -> dict[str, object]:
    checkpoint_input_sha256 = _track_b_checkpoint_input_sha256(
        feature_sha256, source_sha256
    )
    return {
        "format_version": 1,
        "status": "complete",
        "track": "B",
        "stage": "candidates",
        "protocol": "existing",
        "modality": modality,
        "candidate_id": candidate_id,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "configuration_sha256": _canonical_sha256(
            _configuration_payload(model_config, train_config)
        ),
        "feature_columns": list(feature_columns_value),
        "feature_sha256": feature_sha256,
        "checkpoint_input_sha256": checkpoint_input_sha256,
        "split_sha256": split_sha256,
        "source_sha256": dict(source_sha256),
        "source_lineage_sha256": _canonical_sha256(dict(source_sha256)),
        "code_revision": code_revision,
        "code_sha256": _canonical_sha256(code_revision),
        "execution_backend": execution_backend,
        "observed_splits": ["train", "validation"],
    }


def _validate_track_b_artifacts(artifacts: object) -> dict[str, object]:
    if not isinstance(artifacts, dict):
        raise ValueError("candidate receipt artifacts must be an object")
    for name, descriptor in artifacts.items():
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise ValueError(f"candidate artifact descriptor is invalid: {name}")
        path = Path(str(descriptor["path"]))
        expected_sha = str(descriptor["sha256"])
        _validate_sha256(f"candidate artifact {name} SHA256", expected_sha)
        if not path.is_file() or checkpoint_sha256(path) != expected_sha:
            raise ValueError(f"candidate artifact is missing or tampered: {name}")
    return artifacts


def _load_track_b_candidate_receipt(
    path: Path, *, expected: Mapping[str, object]
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"candidate receipt cannot be read: {path}") from exc
    if not isinstance(receipt, dict):
        raise ValueError("candidate receipt must be a JSON object")
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        if "execution_backend" in mismatches:
            raise ValueError(
                "execution_backend mismatch: receipt="
                f"{receipt.get('execution_backend')!r}, requested="
                f"{expected.get('execution_backend')!r}"
            )
        raise ValueError(f"candidate receipt provenance mismatch: {mismatches}")
    metrics = receipt.get("validation_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("candidate receipt validation_metrics must be an object")
    validated_metrics = _validated_track_b_candidate_evaluation(
        metrics, candidate_id=str(expected["candidate_id"])
    )
    artifacts = _validate_track_b_artifacts(receipt.get("artifacts"))
    if artifacts:
        if set(artifacts) != {
            "recording_predictions",
            "participant_predictions",
            "metrics",
            "checkpoint",
        }:
            raise ValueError("candidate artifact schema is not exact")
        recording = pd.read_csv(
            Path(str(artifacts["recording_predictions"]["path"])), low_memory=False  # type: ignore[index]
        )
        participant = pd.read_csv(
            Path(str(artifacts["participant_predictions"]["path"])), low_memory=False  # type: ignore[index]
        )
        recording_columns = {
            "recording_id",
            "participant_id",
            "dataset",
            "modality",
            "label_binary",
            "split",
            "probability",
            "threshold",
            "analysis_unit",
        }
        participant_columns = recording_columns | {
            "n_recordings",
        }
        if set(recording) != recording_columns or set(participant) != participant_columns:
            raise ValueError("candidate prediction artifact schema is invalid")
        if not recording["analysis_unit"].astype(str).eq("recording").all():
            raise ValueError("candidate recording analysis unit is invalid")
        if not participant["analysis_unit"].astype(str).eq("participant").all():
            raise ValueError("candidate participant analysis unit is invalid")
        if set(recording["split"].astype(str)) != {"validation"} or set(
            participant["split"].astype(str)
        ) != {"validation"}:
            raise ValueError("candidate prediction split is invalid")
        recomputed_participant = aggregate_participant_probabilities(recording)
        recomputed_participant = recomputed_participant.sort_values(
            "participant_id", kind="stable"
        ).reset_index(drop=True)
        stored_participant = participant.sort_values(
            "participant_id", kind="stable"
        ).reset_index(drop=True)
        if (
            stored_participant["participant_id"].astype(str).tolist()
            != recomputed_participant["participant_id"].astype(str).tolist()
            or stored_participant["label_binary"].astype(str).tolist()
            != recomputed_participant["label_binary"].astype(str).tolist()
            or not np.allclose(
                stored_participant["probability"].to_numpy(dtype=np.float64),
                recomputed_participant["probability"].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=1e-15,
            )
        ):
            raise ValueError("candidate participant predictions are not recomputable")
        labels = _track_b_labels_to_binary(stored_participant["label_binary"])
        probabilities = stored_participant["probability"].to_numpy(dtype=np.float64)
        threshold = best_threshold_by_balanced_accuracy(labels, probabilities)
        recomputed_metrics = complete_metric_bundle(
            labels, probabilities, threshold=threshold
        )
        for key, value in {
            "auroc": recomputed_metrics["auroc"],
            "auprc": recomputed_metrics["auprc"],
            "threshold": threshold,
        }.items():
            if not math.isclose(
                float(validated_metrics[key]),
                float(value),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"candidate validation metric is not recomputable: {key}")
        metric_artifact = json.loads(
            Path(str(artifacts["metrics"]["path"])).read_text(encoding="utf-8")  # type: ignore[index]
        )
        if not isinstance(metric_artifact, dict):
            raise ValueError("candidate metric artifact must be an object")
        for key, value in recomputed_metrics.items():
            if not math.isclose(
                float(metric_artifact.get(key, float("nan"))),
                float(value),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"candidate metric artifact is not recomputable: {key}")
    return receipt


def _evaluate_or_load_track_b_candidate(
    *,
    protocol: TrackBProtocol,
    feature_columns_value: tuple[str, ...],
    feature_sha256: str,
    source_sha256: Mapping[str, str],
    modality: str,
    candidate_id: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    output_dir: Path,
    code_revision: str,
    execution_backend: str,
    evaluator: Callable[
        [ModelConfig, TrainConfig, pd.DataFrame, pd.DataFrame, Path, bool],
        Mapping[str, object],
    ],
    resume: bool,
) -> dict[str, object]:
    expected = _track_b_candidate_receipt_expected(
        modality=modality,
        candidate_id=candidate_id,
        model_config=model_config,
        train_config=train_config,
        feature_columns_value=feature_columns_value,
        feature_sha256=feature_sha256,
        split_sha256=protocol.split_sha256,
        source_sha256=source_sha256,
        code_revision=code_revision,
        execution_backend=execution_backend,
    )
    receipt_path = _track_b_candidate_receipt_path(output_dir, modality, candidate_id)
    completed = _load_track_b_candidate_receipt(receipt_path, expected=expected)
    if completed is not None:
        return completed

    modality_rows = protocol.source[
        protocol.source["modality"].astype(str).eq(modality)
    ]
    training = modality_rows.loc[modality_rows["split"].eq("train")].copy()
    validation = modality_rows.loc[modality_rows["split"].eq("validation")].copy()
    if training.empty or validation.empty:
        raise ValueError(f"candidate selection needs train/validation rows for {modality}")
    if not set(training["split"].astype(str)) <= {"train"}:
        raise RuntimeError("candidate evaluator training boundary contains non-train rows")
    if not set(validation["split"].astype(str)) <= {"validation"}:
        raise RuntimeError("candidate evaluator validation boundary contains non-validation rows")
    unit_dir = _track_b_candidate_unit_dir(output_dir, modality, candidate_id)
    evaluation = _validated_track_b_candidate_evaluation(
        evaluator(model_config, train_config, training, validation, unit_dir, resume),
        candidate_id=candidate_id,
    )
    receipt = {
        **expected,
        "validation_metrics": {
            "auroc": evaluation["auroc"],
            "auprc": evaluation["auprc"],
            "threshold": evaluation["threshold"],
        },
        "artifacts": evaluation["artifacts"],
    }
    _atomic_json_write(receipt, receipt_path)
    return receipt


def _track_b_published_configs(
    model_name: str,
    *,
    published: Mapping[str, object],
    selection: Mapping[str, object],
    smoke: bool,
) -> tuple[ModelConfig, TrainConfig]:
    depth = 2 if smoke else int(published.get("depth", 11))
    num_trees = (
        1
        if model_name == "dndt"
        else (2 if smoke else int(published.get("dndf_trees", 25)))
    )
    model_config = ModelConfig(
        model_name,  # type: ignore[arg-type]
        num_trees=num_trees,
        depth=depth,
        used_features_rate=(
            1.0 if smoke else float(published.get("used_features_rate", 0.6))
        ),
    )
    train_config = TrainConfig(
        learning_rate=float(published.get("learning_rate", 0.01)),
        weight_decay=0.0,
        batch_size=(4 if smoke else int(published.get("batch_size", 16))),
        max_epochs=(2 if smoke else int(published.get("epochs", 14))),
        patience=(2 if smoke else int(selection.get("patience", 3))),
        seed=42,
        balance_method="class_weight" if smoke else "smote",
    )
    return model_config, train_config


def _track_b_prespecified_ladder_selection(
    *,
    modality: str,
    protocol: TrackBProtocol,
    data: TrackBData,
    prespecified: Mapping[str, object],
    feature_selection_scope: str,
    feature_selection_artifact_sha256: str,
    smoke: bool,
) -> dict[str, object]:
    required = {
        "source",
        "num_trees",
        "depth",
        "used_features_rate",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "balance_method",
    }
    missing = sorted(required - set(prespecified))
    if missing:
        raise ValueError(
            f"prespecified_ladder_dndf is missing required keys: {missing}"
        )
    if prespecified["source"] != "author_published_configuration":
        raise ValueError(
            "prespecified_ladder_dndf.source must be "
            "author_published_configuration"
        )
    model_config = ModelConfig(
        "dndf",
        num_trees=(2 if smoke else int(prespecified["num_trees"])),
        depth=(2 if smoke else int(prespecified["depth"])),
        used_features_rate=(
            1.0 if smoke else float(prespecified["used_features_rate"])
        ),
    )
    train_config = TrainConfig(
        learning_rate=float(prespecified["learning_rate"]),
        weight_decay=float(prespecified["weight_decay"]),
        batch_size=(4 if smoke else int(prespecified["batch_size"])),
        max_epochs=(2 if smoke else int(prespecified["max_epochs"])),
        patience=(2 if smoke else int(prespecified["patience"])),
        seed=42,
        balance_method=(
            "class_weight" if smoke else str(prespecified["balance_method"])
        ),
    )
    selection_payload = {
        "configuration_selection_source": str(prespecified["source"]),
        "protocol": protocol.name,
        "modality": modality,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "feature_sha256": data.feature_sha256,
        "feature_selection_scope": feature_selection_scope,
        "feature_selection_artifact_sha256": feature_selection_artifact_sha256,
        "split_sha256": protocol.split_sha256,
    }
    return {
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "selected_configuration_sha256": _canonical_sha256(selection_payload),
        "configuration_selection_source": str(prespecified["source"]),
        "feature_selection_scope": feature_selection_scope,
        "feature_selection_artifact_sha256": feature_selection_artifact_sha256,
    }


def _track_b_trial_configs(
    params: Mapping[str, object], *, selection: Mapping[str, object], smoke: bool
) -> tuple[ModelConfig, TrainConfig]:
    model_config = ModelConfig(
        "dndf",
        num_trees=2 if smoke else int(params["num_trees"]),
        depth=2 if smoke else int(params["depth"]),
        used_features_rate=1.0 if smoke else float(params["used_features_rate"]),
    )
    train_config = TrainConfig(
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        batch_size=4 if smoke else 16,
        max_epochs=2 if smoke else 14,
        patience=2 if smoke else int(selection.get("patience", 3)),
        seed=42,
        balance_method="class_weight" if smoke else "smote",
    )
    return model_config, train_config


def _candidate_selection_key(receipt: Mapping[str, object]) -> tuple[float, float]:
    metrics = receipt["validation_metrics"]
    if not isinstance(metrics, Mapping):
        raise ValueError("candidate validation metrics are invalid")
    return _selection_key(float(metrics["auroc"]), float(metrics["auprc"]))


def _dndf_candidate_rank_key(
    receipt: Mapping[str, object],
) -> tuple[float, float, str]:
    return (*_candidate_selection_key(receipt), str(receipt["candidate_id"]))


def _track_b_study_name(
    *,
    modality: str,
    feature_sha256: str,
    split_sha256: str,
    source_sha256: Mapping[str, str],
    code_revision: str,
) -> str:
    return "track_b_" + _canonical_sha256(
        {
            "modality": modality,
            "feature_sha256": feature_sha256,
            "checkpoint_input_sha256": _track_b_checkpoint_input_sha256(
                feature_sha256, source_sha256
            ),
            "split_sha256": split_sha256,
            "code_revision": code_revision,
        }
    )[:24]


def _freeze_track_b_selected_configuration(
    path: Path, modality: str, selected: dict[str, object]
) -> None:
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("selected configuration file cannot be authenticated") from exc
        if not isinstance(payload, dict) or set(payload) != {"format_version", "modalities"}:
            raise ValueError("selected configuration file schema is invalid")
        modalities = payload.get("modalities")
        if not isinstance(modalities, dict):
            raise ValueError("selected configuration modalities must be an object")
    else:
        payload = {"format_version": 1, "modalities": {}}
        modalities = payload["modalities"]
    existing = modalities.get(modality)
    if existing is not None and existing != selected:
        raise ValueError(f"immutable selected DNDF lineage mismatch for modality={modality}")
    modalities[modality] = selected
    _atomic_json_write(payload, path)


def _authenticate_track_b_selected_candidates(
    *,
    selected_path: Path,
    candidates_dir: Path,
    protocol: TrackBProtocol,
    data: TrackBData,
    code_revision: str,
    execution_backend: str,
) -> tuple[dict[str, dict[str, object]], list[dict[str, str]]]:
    try:
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("selected candidate configuration cannot be authenticated") from exc
    if not isinstance(payload, dict) or set(payload) != {"format_version", "modalities"}:
        raise ValueError("selected candidate configuration schema is invalid")
    modalities = payload.get("modalities")
    if not isinstance(modalities, dict):
        raise ValueError("selected candidate modalities are invalid")
    validated: dict[str, dict[str, object]] = {}
    authenticated: dict[Path, dict[str, str]] = {}
    for modality, selected_value in sorted(modalities.items()):
        if not isinstance(selected_value, dict):
            raise ValueError("selected candidate lineage must be an object")
        selected = selected_value
        selected_lineage = {
            "modality": modality,
            "selection_protocol": "existing",
            "feature_columns": list(data.feature_columns),
            "feature_sha256": data.feature_sha256,
            "checkpoint_input_sha256": _track_b_checkpoint_input_sha256(
                data.feature_sha256, data.source_sha256
            ),
            "split_sha256": protocol.split_sha256,
            "source_sha256": data.source_sha256,
            "source_lineage_sha256": _canonical_sha256(data.source_sha256),
            "code_revision": code_revision,
            "code_sha256": _canonical_sha256(code_revision),
            "execution_backend": execution_backend,
        }
        lineage_mismatches = [
            key for key, value in selected_lineage.items() if selected.get(key) != value
        ]
        if lineage_mismatches:
            raise ValueError(
                "selected candidate lineage is not authenticated: "
                f"{lineage_mismatches}"
            )
        model_payload = selected.get("model_config")
        train_payload = selected.get("train_config")
        if not isinstance(model_payload, dict) or not isinstance(train_payload, dict):
            raise ValueError("selected candidate model/train configuration is invalid")
        expected_hyperparameters = {
            "depth": model_payload.get("depth"),
            "num_trees": model_payload.get("num_trees"),
            "used_features_rate": model_payload.get("used_features_rate"),
            "learning_rate": train_payload.get("learning_rate"),
            "weight_decay": train_payload.get("weight_decay"),
            "batch_size": train_payload.get("batch_size"),
            "max_epochs": train_payload.get("max_epochs"),
            "patience": train_payload.get("patience"),
            "balance_method": train_payload.get("balance_method"),
        }
        if selected.get("selected_hyperparameters") != expected_hyperparameters:
            raise ValueError("selected candidate hyperparameters are not authenticated")
        registry_value = selected.get("candidate_registry")
        if not isinstance(registry_value, list) or not registry_value:
            raise ValueError("selected candidate registry is missing")
        registry: dict[str, str] = {}
        for entry in registry_value:
            if not isinstance(entry, dict) or set(entry) != {
                "candidate_id",
                "receipt_sha256",
            }:
                raise ValueError("selected candidate registry entry is invalid")
            candidate_id = str(entry["candidate_id"])
            receipt_sha256 = str(entry["receipt_sha256"])
            _validate_sha256("candidate receipt SHA256", receipt_sha256)
            if not candidate_id or candidate_id in registry:
                raise ValueError("selected candidate registry IDs must be unique")
            registry[candidate_id] = receipt_sha256

        receipt_paths = sorted((candidates_dir / str(modality)).glob("*/completion.json"))
        if {path.parent.name for path in receipt_paths} != set(registry):
            raise ValueError("selected candidate registry does not match completed receipts")
        receipts: dict[str, dict[str, object]] = {}
        for receipt_path in receipt_paths:
            candidate_id = receipt_path.parent.name
            if checkpoint_sha256(receipt_path) != registry[candidate_id]:
                raise ValueError("selected candidate registry receipt hash mismatch")
            try:
                raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("selected candidate receipt cannot be read") from exc
            if not isinstance(raw_receipt, dict):
                raise ValueError("selected candidate receipt must be an object")
            model_value = raw_receipt.get("model_config")
            train_value = raw_receipt.get("train_config")
            if not isinstance(model_value, dict) or not isinstance(train_value, dict):
                raise ValueError("selected candidate model/train configuration is invalid")
            model_config = ModelConfig(**model_value)  # type: ignore[arg-type]
            train_config = TrainConfig(**train_value)  # type: ignore[arg-type]
            expected = _track_b_candidate_receipt_expected(
                modality=str(modality),
                candidate_id=candidate_id,
                model_config=model_config,
                train_config=train_config,
                feature_columns_value=data.feature_columns,
                feature_sha256=data.feature_sha256,
                split_sha256=protocol.split_sha256,
                source_sha256=data.source_sha256,
                code_revision=code_revision,
                execution_backend=execution_backend,
            )
            receipt = _load_track_b_candidate_receipt(receipt_path, expected=expected)
            if receipt is None:
                raise ValueError("registered candidate receipt is missing")
            receipts[candidate_id] = receipt
            authenticated[receipt_path] = {
                "path": str(receipt_path),
                "sha256": checkpoint_sha256(receipt_path),
            }
        if not {"published_dndt", "published_dndf"} <= set(receipts):
            raise ValueError("published candidate receipts are missing from registry")
        dndf_receipts = [
            receipt
            for receipt in receipts.values()
            if receipt["model_config"]["model_name"] == "dndf"  # type: ignore[index]
        ]
        best_dndf = max(dndf_receipts, key=_dndf_candidate_rank_key)
        if (
            selected.get("candidate_id") != best_dndf.get("candidate_id")
            or selected.get("model_config") != best_dndf.get("model_config")
            or selected.get("train_config") != best_dndf.get("train_config")
            or selected.get("validation_metrics") != best_dndf.get("validation_metrics")
            or selected.get("selected_configuration_sha256")
            != best_dndf.get("configuration_sha256")
        ):
            raise ValueError("selected DNDF candidate is not the authenticated best candidate")
        overall = max(
            (receipts["published_dndt"], best_dndf), key=_candidate_selection_key
        )
        overall_winner = str(overall["model_config"]["model_name"])  # type: ignore[index]
        if (
            selected.get("overall_winner") != overall_winner
            or selected.get("overall_winner_model_config") != overall.get("model_config")
            or selected.get("overall_winner_train_config") != overall.get("train_config")
            or selected.get("overall_winner_validation_metrics")
            != overall.get("validation_metrics")
            or selected.get("overall_winner_configuration_sha256")
            != overall.get("configuration_sha256")
        ):
            raise ValueError("overall winner is not the authenticated best candidate")
        validated[str(modality)] = selected
    return validated, [authenticated[path] for path in sorted(authenticated)]


def validate_frozen_track_b_lineage(
    selected_configuration: Mapping[str, object],
    *,
    feature_columns: Sequence[str],
    feature_sha256: str,
    configuration_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if configuration_override is not None:
        raise ValueError("protocol-specific configuration override is forbidden")
    selected_features = selected_configuration.get("feature_columns")
    if selected_features != list(feature_columns) or selected_configuration.get(
        "feature_sha256"
    ) != feature_sha256:
        raise ValueError("feature order/hash does not match frozen Track B lineage")
    hyperparameters = selected_configuration.get("selected_hyperparameters")
    if not isinstance(hyperparameters, dict):
        raise ValueError("frozen selected_hyperparameters are invalid")
    return dict(hyperparameters)


def select_modality_configuration(
    protocol: TrackBProtocol,
    *,
    feature_columns: Sequence[str],
    feature_sha256: str,
    source_sha256: Mapping[str, str],
    modality: str,
    max_trials: int,
    storage: str | Path,
    output_dir: str | Path,
    code_revision: str,
    device: str = "cpu",
    evaluator: Callable[
        [ModelConfig, TrainConfig, pd.DataFrame, pd.DataFrame, Path, bool],
        Mapping[str, object],
    ]
    | None = None,
    published: Mapping[str, object] | None = None,
    selection: Mapping[str, object] | None = None,
    resume: bool = False,
    smoke: bool = False,
) -> CandidateSelectionResult:
    if protocol.name != "existing":
        raise ValueError("candidate selection is allowed only on the existing split")
    if modality not in TRACK_B_MODALITIES:
        raise ValueError(f"unknown Track B modality: {modality}")
    _require_integer("max_trials", max_trials, minimum=0)
    if max_trials > 6:
        raise ValueError("max_trials cannot exceed six")
    feature_columns_value = tuple(feature_columns)
    if len(feature_columns_value) != 800:
        raise ValueError("candidate selection requires exactly 800 feature columns")
    _validate_sha256("feature_sha256", feature_sha256)
    execution_backend = normalize_execution_backend(device)
    output_path = Path(output_dir)
    storage_path = Path(storage).resolve()
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    published_values = dict(published or {})
    selection_values = dict(selection or {})
    evaluate = evaluator or _default_track_b_candidate_evaluator(
        feature_columns_value=feature_columns_value,
        feature_sha256=feature_sha256,
        checkpoint_input_sha256=_track_b_checkpoint_input_sha256(
            feature_sha256, source_sha256
        ),
        split_sha256=protocol.split_sha256,
        code_revision=code_revision,
        device=execution_backend,
    )

    candidates: list[dict[str, object]] = []
    published_receipts: dict[str, dict[str, object]] = {}
    for model_name in ("dndt", "dndf"):
        model_config, train_config = _track_b_published_configs(
            model_name,
            published=published_values,
            selection=selection_values,
            smoke=smoke,
        )
        receipt = _evaluate_or_load_track_b_candidate(
            protocol=protocol,
            feature_columns_value=feature_columns_value,
            feature_sha256=feature_sha256,
            source_sha256=source_sha256,
            modality=modality,
            candidate_id=f"published_{model_name}",
            model_config=model_config,
            train_config=train_config,
            output_dir=output_path,
            code_revision=code_revision,
            execution_backend=execution_backend,
            evaluator=evaluate,
            resume=resume,
        )
        published_receipts[model_name] = receipt
        candidates.append(receipt)

    completed_trials = 0
    if _candidate_selection_key(published_receipts["dndf"]) <= _candidate_selection_key(
        published_receipts["dndt"]
    ) and max_trials:
        import optuna
        from optuna.trial import TrialState

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_name = _track_b_study_name(
            modality=modality,
            feature_sha256=feature_sha256,
            split_sha256=protocol.split_sha256,
            source_sha256=source_sha256,
            code_revision=code_revision,
        )
        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{storage_path.as_posix()}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=True,
        )

        required_params = {
            "depth",
            "num_trees",
            "used_features_rate",
            "learning_rate",
            "weight_decay",
        }
        attempted_trials = len(study.get_trials(deepcopy=False))
        if attempted_trials > max_trials:
            raise ValueError("persistent study exceeds the six-attempt contract")
        for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            if set(trial.params) != required_params:
                study.tell(trial.number, state=TrialState.FAIL)
                continue
            model_config, train_config = _track_b_trial_configs(
                trial.params, selection=selection_values, smoke=smoke
            )
            expected = _track_b_candidate_receipt_expected(
                modality=modality,
                candidate_id=f"optuna_{trial.number:03d}",
                model_config=model_config,
                train_config=train_config,
                feature_columns_value=feature_columns_value,
                feature_sha256=feature_sha256,
                split_sha256=protocol.split_sha256,
                source_sha256=source_sha256,
                code_revision=code_revision,
                execution_backend=execution_backend,
            )
            recovered = _load_track_b_candidate_receipt(
                _track_b_candidate_receipt_path(
                    output_path, modality, f"optuna_{trial.number:03d}"
                ),
                expected=expected,
            )
            if recovered is None:
                if not resume:
                    raise ValueError("unfinished Optuna trial requires --resume")
                recovered = _evaluate_or_load_track_b_candidate(
                    protocol=protocol,
                    feature_columns_value=feature_columns_value,
                    feature_sha256=feature_sha256,
                    source_sha256=source_sha256,
                    modality=modality,
                    candidate_id=f"optuna_{trial.number:03d}",
                    model_config=model_config,
                    train_config=train_config,
                    output_dir=output_path,
                    code_revision=code_revision,
                    execution_backend=execution_backend,
                    evaluator=evaluate,
                    resume=True,
                )
            study.tell(trial.number, float(_candidate_selection_key(recovered)[0]))

        completed_trials = len(
            study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        )
        attempted_trials = len(study.get_trials(deepcopy=False))
        if attempted_trials > max_trials:
            raise ValueError("persistent study exceeds the six-attempt contract")

        def objective(trial: object) -> float:
            params = {
                "depth": trial.suggest_int("depth", 5, 11),  # type: ignore[attr-defined]
                "num_trees": trial.suggest_categorical(  # type: ignore[attr-defined]
                    "num_trees", [5, 10, 15, 25]
                ),
                "used_features_rate": trial.suggest_categorical(  # type: ignore[attr-defined]
                    "used_features_rate", [0.4, 0.6, 0.8]
                ),
                "learning_rate": trial.suggest_float(  # type: ignore[attr-defined]
                    "learning_rate", 1e-4, 1e-2, log=True
                ),
                "weight_decay": trial.suggest_float(  # type: ignore[attr-defined]
                    "weight_decay", 1e-6, 1e-3, log=True
                ),
            }
            model_config, train_config = _track_b_trial_configs(
                params, selection=selection_values, smoke=smoke
            )
            receipt = _evaluate_or_load_track_b_candidate(
                protocol=protocol,
                feature_columns_value=feature_columns_value,
                feature_sha256=feature_sha256,
                source_sha256=source_sha256,
                modality=modality,
                candidate_id=f"optuna_{trial.number:03d}",  # type: ignore[attr-defined]
                model_config=model_config,
                train_config=train_config,
                output_dir=output_path,
                code_revision=code_revision,
                execution_backend=execution_backend,
                evaluator=evaluate,
                resume=resume,
            )
            return float(_candidate_selection_key(receipt)[0])

        remaining = max_trials - attempted_trials
        if remaining:
            study.optimize(objective, n_trials=remaining, catch=())
        complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        completed_trials = len(complete)
        for trial in complete:
            model_config, train_config = _track_b_trial_configs(
                trial.params, selection=selection_values, smoke=smoke
            )
            expected = _track_b_candidate_receipt_expected(
                modality=modality,
                candidate_id=f"optuna_{trial.number:03d}",
                model_config=model_config,
                train_config=train_config,
                feature_columns_value=feature_columns_value,
                feature_sha256=feature_sha256,
                split_sha256=protocol.split_sha256,
                source_sha256=source_sha256,
                code_revision=code_revision,
                execution_backend=execution_backend,
            )
            receipt = _load_track_b_candidate_receipt(
                _track_b_candidate_receipt_path(
                    output_path, modality, f"optuna_{trial.number:03d}"
                ),
                expected=expected,
            )
            if receipt is None or not math.isclose(
                float(trial.value),
                float(_candidate_selection_key(receipt)[0]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("Optuna trial is not bound to its candidate receipt")
            candidates.append(receipt)

    dndf_candidates = [
        candidate
        for candidate in candidates
        if candidate["model_config"]["model_name"] == "dndf"  # type: ignore[index]
    ]
    selected_dndf_receipt = max(dndf_candidates, key=_dndf_candidate_rank_key)
    overall_receipt = max(
        (published_receipts["dndt"], selected_dndf_receipt),
        key=_candidate_selection_key,
    )
    overall_winner = str(overall_receipt["model_config"]["model_name"])  # type: ignore[index]
    model_values = selected_dndf_receipt["model_config"]
    train_values = selected_dndf_receipt["train_config"]
    if not isinstance(model_values, dict) or not isinstance(train_values, dict):
        raise ValueError("selected candidate configuration is invalid")
    selected_hyperparameters = {
        "depth": model_values["depth"],
        "num_trees": model_values["num_trees"],
        "used_features_rate": model_values["used_features_rate"],
        "learning_rate": train_values["learning_rate"],
        "weight_decay": train_values["weight_decay"],
        "batch_size": train_values["batch_size"],
        "max_epochs": train_values["max_epochs"],
        "patience": train_values["patience"],
        "balance_method": train_values["balance_method"],
    }
    candidate_registry = [
        {
            "candidate_id": str(candidate["candidate_id"]),
            "receipt_sha256": checkpoint_sha256(
                _track_b_candidate_receipt_path(
                    output_path, modality, str(candidate["candidate_id"])
                )
            ),
        }
        for candidate in sorted(candidates, key=lambda value: str(value["candidate_id"]))
    ]
    selected = {
        "modality": modality,
        "selection_protocol": "existing",
        "model_config": model_values,
        "train_config": train_values,
        "selected_hyperparameters": selected_hyperparameters,
        "selected_configuration_sha256": selected_dndf_receipt[
            "configuration_sha256"
        ],
        "validation_metrics": selected_dndf_receipt["validation_metrics"],
        "candidate_id": selected_dndf_receipt["candidate_id"],
        "overall_winner": overall_winner,
        "overall_winner_model_config": overall_receipt["model_config"],
        "overall_winner_train_config": overall_receipt["train_config"],
        "overall_winner_configuration_sha256": overall_receipt[
            "configuration_sha256"
        ],
        "overall_winner_validation_metrics": overall_receipt["validation_metrics"],
        "feature_columns": list(feature_columns_value),
        "feature_sha256": feature_sha256,
        "checkpoint_input_sha256": _track_b_checkpoint_input_sha256(
            feature_sha256, source_sha256
        ),
        "split_sha256": protocol.split_sha256,
        "source_sha256": dict(source_sha256),
        "source_lineage_sha256": _canonical_sha256(dict(source_sha256)),
        "code_revision": code_revision,
        "code_sha256": _canonical_sha256(code_revision),
        "execution_backend": execution_backend,
        "candidate_registry": candidate_registry,
    }
    selected_path = output_path.parent / "selected_configurations.json"
    _freeze_track_b_selected_configuration(selected_path, modality, selected)
    candidate_summaries = tuple(
        {
            "candidate_id": candidate["candidate_id"],
            "model_name": candidate["model_config"]["model_name"],  # type: ignore[index]
            "configuration_sha256": candidate["configuration_sha256"],
            "validation_metrics": candidate["validation_metrics"],
        }
        for candidate in sorted(candidates, key=lambda value: str(value["candidate_id"]))
    )
    return CandidateSelectionResult(
        modality=modality,
        completed_trials=completed_trials,
        observed_splits=("train", "validation"),
        selected_dndf=selected,
        overall_winner=overall_winner,
        candidates=candidate_summaries,
    )


_TRACK_B_EXECUTION_PREDICTION_COLUMNS = (
    "run_id",
    "track",
    "stage",
    "protocol",
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
    "analysis_unit",
    "n_recordings",
    "configuration_sha256",
    "selected_configuration_sha256",
    "split_sha256",
    "feature_sha256",
    "checkpoint_sha256",
    "code_revision",
    "execution_backend",
)


def _track_b_inference_probabilities(
    checkpoint_dir: Path,
    features: np.ndarray,
    *,
    checkpoint_input_sha256: str,
    split_sha256: str,
    code_revision: str,
    execution_backend: str,
) -> tuple[np.ndarray, Path, str]:
    resolved = _resolve_checkpoint_manifest(
        checkpoint_dir, role="best_inference", map_location="cpu"
    )
    payload = resolved.payload
    expected = {
        "input_feature_hash": checkpoint_input_sha256,
        "split_hash": split_sha256,
        "code_revision": code_revision,
        "execution_backend": execution_backend,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise ValueError(f"Track B inference checkpoint provenance mismatch: {mismatches}")
    model_payload = payload.get("model_config")
    train_payload = payload.get("train_config")
    preprocessing = payload.get("preprocessor_state")
    if not isinstance(model_payload, dict) or not isinstance(train_payload, dict):
        raise ValueError("Track B inference checkpoint configuration is invalid")
    if not isinstance(preprocessing, dict):
        raise ValueError("Track B inference checkpoint preprocessing is invalid")
    model_config = ModelConfig(**model_payload)  # type: ignore[arg-type]
    train_config = TrainConfig(**train_payload)  # type: ignore[arg-type]
    matrix = _feature_matrix(features, name="inference_features")
    n_features = int(preprocessing.get("n_features", -1))
    if matrix.shape[1] != n_features:
        raise ValueError("Track B inference feature count does not match checkpoint")
    statistics = np.asarray(preprocessing.get("imputer_statistics"), dtype=np.float64)
    mean = np.asarray(preprocessing.get("scaler_mean"), dtype=np.float64)
    scale = np.asarray(preprocessing.get("scaler_scale"), dtype=np.float64)
    if any(value.shape != (n_features,) for value in (statistics, mean, scale)):
        raise ValueError("Track B checkpoint preprocessing vector shape is invalid")
    transformed = matrix.copy()
    missing = np.isnan(transformed)
    transformed[missing] = np.broadcast_to(statistics, transformed.shape)[missing]
    transformed = (transformed - mean) / scale
    if not np.isfinite(transformed).all():
        raise ValueError("Track B inference features are nonfinite after preprocessing")
    torch_device = torch.device(execution_backend)
    model = NeuralDecisionClassifier(
        num_features=n_features,
        model_config=model_config,
        seed=train_config.seed,
    ).to(torch_device)
    model.load_state_dict(payload["model_state"])  # type: ignore[arg-type]
    probability = _predict_probabilities(
        model,
        transformed,
        device=torch_device,
        batch_size=1024,
    )
    return probability, resolved.path, str(resolved.descriptor["sha256"])


def _default_track_b_unit_evaluator(
    *,
    feature_sha256: str,
    checkpoint_input_sha256: str,
    split_sha256: str,
    code_revision: str,
    execution_backend: str,
) -> Callable[..., dict[str, object]]:
    def evaluate(
        model_config: ModelConfig,
        train_config: TrainConfig,
        training: pd.DataFrame,
        validation: pd.DataFrame,
        evaluation: pd.DataFrame,
        feature_names: tuple[str, ...],
        unit_dir: Path,
        resume: bool,
    ) -> dict[str, object]:
        checkpoint_dir = unit_dir / "checkpoints"
        can_resume = resume and _manifest_path(
            checkpoint_dir, "latest_recovery"
        ).is_file()
        fitted = fit_model(
            training.loc[:, feature_names].to_numpy(dtype=np.float64),
            _track_b_labels_to_binary(training["label_binary"]),
            validation.loc[:, feature_names].to_numpy(dtype=np.float64),
            _track_b_labels_to_binary(validation["label_binary"]),
            model_config=model_config,
            train_config=train_config,
            checkpoint_dir=checkpoint_dir,
            input_feature_hash=checkpoint_input_sha256,
            split_hash=split_sha256,
            code_revision=code_revision,
            device=execution_backend,
            resume=can_resume,
            validation_group_ids=validation["participant_id"].astype(str).to_numpy(),
        )
        evaluation_probability, checkpoint_path, checkpoint_sha = (
            _track_b_inference_probabilities(
                checkpoint_dir,
                evaluation.loc[:, feature_names].to_numpy(dtype=np.float64),
                checkpoint_input_sha256=checkpoint_input_sha256,
                split_sha256=split_sha256,
                code_revision=code_revision,
                execution_backend=execution_backend,
            )
        )
        if checkpoint_path != fitted.checkpoint_path:
            raise ValueError("fitted and resolved Track B checkpoints differ")
        return {
            "validation_probability": fitted.validation_probability,
            "evaluation_probability": evaluation_probability,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha,
            "best_epoch": fitted.best_epoch,
        }

    return evaluate


def _validated_track_b_unit_evaluation(
    value: Mapping[str, object],
    *,
    validation_rows: int,
    evaluation_rows: int,
) -> dict[str, object]:
    validation_probability = np.asarray(
        value.get("validation_probability"), dtype=np.float64
    )
    evaluation_probability = np.asarray(
        value.get("evaluation_probability"), dtype=np.float64
    )
    if validation_probability.shape != (validation_rows,):
        raise ValueError("Track B unit validation probability shape is invalid")
    if evaluation_probability.shape != (evaluation_rows,):
        raise ValueError("Track B unit evaluation probability shape is invalid")
    for name, probability in (
        ("validation", validation_probability),
        ("evaluation", evaluation_probability),
    ):
        if not np.isfinite(probability).all() or bool(
            ((probability < 0.0) | (probability > 1.0)).any()
        ):
            raise ValueError(f"Track B {name} probabilities must be finite within [0, 1]")
    checkpoint_path = Path(str(value.get("checkpoint_path", "")))
    if not checkpoint_path.is_file():
        raise ValueError("Track B unit checkpoint is missing")
    actual_sha = checkpoint_sha256(checkpoint_path)
    supplied_sha = value.get("checkpoint_sha256")
    if supplied_sha is not None and str(supplied_sha) != actual_sha:
        raise ValueError("Track B unit checkpoint SHA256 mismatch")
    best_epoch = int(value.get("best_epoch", 0))
    if best_epoch < 1:
        raise ValueError("Track B unit best_epoch must be positive")
    return {
        "validation_probability": validation_probability,
        "evaluation_probability": evaluation_probability,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": actual_sha,
        "best_epoch": best_epoch,
    }


def _track_b_participant_prediction_frame(recording: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in recording.groupby(
        ["dataset", "split", "modality"], sort=True, dropna=False
    ):
        aggregated = aggregate_participant_probabilities(group)
        first = group.iloc[0]
        aggregated["recording_id"] = aggregated["participant_id"]
        for column in (
            "run_id",
            "track",
            "stage",
            "protocol",
            "seed",
            "dataset",
            "split",
            "model_name",
            "modality",
            "threshold",
            "threshold_source",
            "configuration_sha256",
            "selected_configuration_sha256",
            "split_sha256",
            "feature_sha256",
            "checkpoint_sha256",
            "code_revision",
            "execution_backend",
        ):
            aggregated[column] = first[column]
        aggregated["analysis_unit"] = "participant"
        frames.append(
            aggregated.loc[:, _TRACK_B_EXECUTION_PREDICTION_COLUMNS]
        )
    if not frames:
        return pd.DataFrame(columns=_TRACK_B_EXECUTION_PREDICTION_COLUMNS)
    return pd.concat(frames, ignore_index=True, sort=False)


def _validated_fusion_predictions(
    predictions: pd.DataFrame, *, modalities: Sequence[str]
) -> pd.DataFrame:
    required = {
        "participant_id",
        "label_binary",
        "split",
        "modality",
        "probability",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"fusion predictions are missing columns: {missing}")
    selected_modalities = tuple(str(value) for value in modalities)
    if not selected_modalities or len(set(selected_modalities)) != len(
        selected_modalities
    ):
        raise ValueError("fusion modalities must be nonempty and unique")
    frame = predictions[
        predictions["modality"].astype(str).isin(selected_modalities)
    ].copy()
    if frame.empty:
        raise ValueError("fusion predictions contain no requested modalities")
    frame["participant_id"] = frame["participant_id"].astype(str).str.strip()
    if frame["participant_id"].eq("").any():
        raise ValueError("fusion participant identifiers must be nonempty")
    normalized_labels = frame["label_binary"].map(_track_b_label_token)
    if normalized_labels.isna().any():
        raise ValueError("fusion labels must be binary")
    frame["label_binary"] = normalized_labels.map(
        {"negative": 0, "positive": 1}
    ).astype(np.int64)
    probability = pd.to_numeric(frame["probability"], errors="coerce")
    if probability.isna().any() or bool(
        ((probability < 0.0) | (probability > 1.0)).any()
    ):
        raise ValueError("fusion probabilities must be finite within [0, 1]")
    frame["probability"] = probability.astype(np.float64)
    duplicated = frame.duplicated(
        ["participant_id", "split", "modality"], keep=False
    )
    if duplicated.any():
        raise ValueError(
            "fusion requires one participant prediction per split and modality"
        )
    conflicts = frame.groupby(
        ["participant_id", "split"], sort=False
    )["label_binary"].nunique()
    if (conflicts != 1).any():
        raise ValueError("fusion labels conflict across modalities")
    return frame.reset_index(drop=True)


def _fusion_wide_table(
    predictions: pd.DataFrame, *, modalities: Sequence[str]
) -> pd.DataFrame:
    frame = _validated_fusion_predictions(predictions, modalities=modalities)
    labels = (
        frame.groupby(["participant_id", "split"], sort=False, as_index=False)[
            "label_binary"
        ]
        .first()
    )
    wide = frame.pivot(
        index=["participant_id", "split"],
        columns="modality",
        values="probability",
    ).reset_index()
    wide.columns.name = None
    wide = wide.merge(
        labels,
        on=["participant_id", "split"],
        how="inner",
        validate="one_to_one",
    )
    return wide


def fit_validation_logistic_fusion(
    predictions: pd.DataFrame,
    *,
    seed: int,
    modalities: Sequence[str] = ("cough", "speech"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit fusion on validation participants and apply frozen weights to test."""
    selected_modalities = tuple(str(value) for value in modalities)
    wide = _fusion_wide_table(predictions, modalities=selected_modalities)
    complete = wide.dropna(subset=list(selected_modalities)).copy()
    validation = complete[complete["split"].astype(str).eq("validation")].copy()
    test = complete[complete["split"].astype(str).eq("test")].copy()
    if validation.empty or test.empty:
        raise ValueError("logistic fusion requires aligned validation and test rows")
    if validation["label_binary"].nunique() != 2:
        raise ValueError("logistic fusion validation labels must contain both classes")
    classifier = LogisticRegression(random_state=int(seed), max_iter=2000)
    classifier.fit(
        validation.loc[:, selected_modalities].to_numpy(dtype=np.float64),
        validation["label_binary"].to_numpy(dtype=np.int64),
    )
    if classifier.classes_.tolist() != [0, 1]:
        raise RuntimeError("logistic fusion positive-class orientation is invalid")
    output_frames: list[pd.DataFrame] = []
    for split_frame in (validation, test):
        output = split_frame[
            ["participant_id", "split", "label_binary"]
        ].copy()
        output["recording_id"] = output["participant_id"]
        output["probability"] = classifier.predict_proba(
            split_frame.loc[:, selected_modalities].to_numpy(dtype=np.float64)
        )[:, 1]
        output["fusion_method"] = "validation_logistic_stack"
        output["available_modalities"] = ",".join(sorted(selected_modalities))
        output_frames.append(output)
    weights = pd.DataFrame(
        [
            {
                "seed": int(seed),
                "fusion_method": "validation_logistic_stack",
                "modality": modality,
                "weight": float(classifier.coef_[0, index]),
            }
            for index, modality in enumerate(selected_modalities)
        ]
        + [
            {
                "seed": int(seed),
                "fusion_method": "validation_logistic_stack",
                "modality": "intercept",
                "weight": float(classifier.intercept_[0]),
            }
        ]
    )
    return pd.concat(output_frames, ignore_index=True), weights


def uniform_dndf_fusion(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average available breath, cough, and speech participant probabilities."""
    modalities = ("breath", "cough", "speech")
    frame = _validated_fusion_predictions(predictions, modalities=modalities)
    normalized = frame[
        ["participant_id", "label_binary", "split", "modality", "probability"]
    ].copy()
    fused = uniform_fusion(normalized, probability_column="probability")
    fused["recording_id"] = fused["participant_id"]
    return fused


def _track_b_recording_prediction_frame(
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    validation_probability: np.ndarray,
    evaluation_probability: np.ndarray,
    threshold: float,
    run_id: str,
    stage: str,
    protocol: str,
    seed: int,
    model_name: str,
    configuration_sha256: str,
    selected_configuration_sha256: str,
    split_sha256: str,
    feature_sha256: str,
    checkpoint_sha: str,
    code_revision: str,
    execution_backend: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source, probability in (
        (validation, validation_probability),
        (evaluation, evaluation_probability),
    ):
        frame = source[
            [
                "dataset",
                "split",
                "modality",
                "participant_id",
                "recording_id",
                "label_binary",
            ]
        ].copy()
        frame["run_id"] = run_id
        frame["track"] = "B"
        frame["stage"] = stage
        frame["protocol"] = protocol
        frame["seed"] = seed
        frame["model_name"] = model_name
        frame["probability"] = probability
        frame["threshold"] = threshold
        frame["threshold_source"] = "source_validation_balanced_accuracy"
        frame["analysis_unit"] = "recording"
        frame["n_recordings"] = 1
        frame["configuration_sha256"] = configuration_sha256
        frame["selected_configuration_sha256"] = selected_configuration_sha256
        frame["split_sha256"] = split_sha256
        frame["feature_sha256"] = feature_sha256
        frame["checkpoint_sha256"] = checkpoint_sha
        frame["code_revision"] = code_revision
        frame["execution_backend"] = execution_backend
        frames.append(frame.loc[:, _TRACK_B_EXECUTION_PREDICTION_COLUMNS])
    return pd.concat(frames, ignore_index=True, sort=False)


def _track_b_execution_unit_dir(
    track_b_dir: Path,
    *,
    stage: str,
    protocol: str,
    modality: str,
    model_name: str,
    seed: int,
) -> Path:
    return (
        track_b_dir
        / "units"
        / stage
        / protocol
        / modality
        / model_name
        / f"seed_{seed}"
    )


def _track_b_execution_expected(
    *,
    run_id: str,
    stage: str,
    protocol: TrackBProtocol,
    modality: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    selected: Mapping[str, object],
    data: TrackBData,
    code_revision: str,
    execution_backend: str,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "status": "complete",
        "run_id": run_id,
        "track": "B",
        "stage": stage,
        "protocol": protocol.name,
        "modality": modality,
        "model_name": model_config.model_name,
        "seed": train_config.seed,
        "evaluation_split": "external" if protocol.name == "external_cough" else "test",
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "configuration_sha256": _canonical_sha256(
            _configuration_payload(model_config, train_config)
        ),
        "selected_configuration_sha256": selected[
            "selected_configuration_sha256"
        ],
        "configuration_selection_source": selected.get(
            "configuration_selection_source", "validation_selected"
        ),
        "feature_selection_scope": selected.get(
            "feature_selection_scope", "frozen_existing_train_only_top800"
        ),
        "feature_selection_artifact_sha256": selected.get(
            "feature_selection_artifact_sha256", data.source_sha256["project"]
        ),
        "feature_columns": list(data.feature_columns),
        "feature_sha256": data.feature_sha256,
        "split_sha256": protocol.split_sha256,
        "source_sha256": data.source_sha256,
        "source_lineage_sha256": _canonical_sha256(data.source_sha256),
        "code_revision": code_revision,
        "code_sha256": _canonical_sha256(code_revision),
        "execution_backend": execution_backend,
        "fit_splits": ["train", "validation"],
        "threshold_source": "source_validation_balanced_accuracy",
        "analysis_units": ["recording", "participant"],
    }


def _track_b_metric_identity(
    expected: Mapping[str, object], *, threshold: float, best_epoch: int
) -> dict[str, object]:
    return {
        key: expected[key]
        for key in (
            "run_id",
            "track",
            "stage",
            "protocol",
            "modality",
            "model_name",
            "seed",
            "evaluation_split",
            "configuration_sha256",
            "selected_configuration_sha256",
            "configuration_selection_source",
            "feature_selection_scope",
            "feature_selection_artifact_sha256",
            "split_sha256",
            "feature_sha256",
            "code_revision",
            "execution_backend",
            "threshold_source",
        )
    } | {"analysis_unit": "participant", "threshold": threshold, "best_epoch": best_epoch}


def _run_or_load_track_b_execution_unit(
    *,
    track_b_dir: Path,
    run_id: str,
    stage: str,
    protocol: TrackBProtocol,
    modality: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    selected: Mapping[str, object],
    data: TrackBData,
    code_revision: str,
    execution_backend: str,
    resume: bool,
    evaluator: Callable[..., Mapping[str, object]],
) -> dict[str, object]:
    expected = _track_b_execution_expected(
        run_id=run_id,
        stage=stage,
        protocol=protocol,
        modality=modality,
        model_config=model_config,
        train_config=train_config,
        selected=selected,
        data=data,
        code_revision=code_revision,
        execution_backend=execution_backend,
    )
    unit_dir = _track_b_execution_unit_dir(
        track_b_dir,
        stage=stage,
        protocol=protocol.name,
        modality=modality,
        model_name=model_config.model_name,
        seed=train_config.seed,
    )
    receipt_path = unit_dir / "completion.json"
    completed = _load_track_b_execution_receipt(receipt_path, expected=expected)
    if completed is not None:
        return completed

    source = protocol.source[
        protocol.source["modality"].astype(str).eq(modality)
    ]
    training = source.loc[source["split"].eq("train")].copy()
    validation = source.loc[source["split"].eq("validation")].copy()
    evaluation = protocol.target[
        protocol.target["modality"].astype(str).eq(modality)
    ].copy()
    if training.empty or validation.empty or evaluation.empty:
        raise ValueError(
            f"Track B unit requires train/validation/evaluation rows: "
            f"{protocol.name}/{modality}"
        )
    participant_sets = [
        set(frame["participant_id"].astype(str))
        for frame in (training, validation, evaluation)
    ]
    if any(
        participant_sets[first] & participant_sets[second]
        for first in range(3)
        for second in range(first + 1, 3)
    ):
        raise ValueError("Track B execution unit has participant overlap across splits")
    evaluation_result = _validated_track_b_unit_evaluation(
        evaluator(
            model_config,
            train_config,
            training,
            validation,
            evaluation,
            data.feature_columns,
            unit_dir,
            resume,
        ),
        validation_rows=len(validation),
        evaluation_rows=len(evaluation),
    )
    validation_stub = validation[
        ["recording_id", "participant_id", "label_binary"]
    ].copy()
    validation_stub["probability"] = evaluation_result["validation_probability"]
    validation_participant = aggregate_participant_probabilities(validation_stub)
    threshold = best_threshold_by_balanced_accuracy(
        _track_b_labels_to_binary(validation_participant["label_binary"]),
        validation_participant["probability"].to_numpy(dtype=np.float64),
    )
    recording = _track_b_recording_prediction_frame(
        validation,
        evaluation,
        validation_probability=evaluation_result["validation_probability"],  # type: ignore[arg-type]
        evaluation_probability=evaluation_result["evaluation_probability"],  # type: ignore[arg-type]
        threshold=threshold,
        run_id=run_id,
        stage=stage,
        protocol=protocol.name,
        seed=train_config.seed,
        model_name=model_config.model_name,
        configuration_sha256=str(expected["configuration_sha256"]),
        selected_configuration_sha256=str(
            expected["selected_configuration_sha256"]
        ),
        split_sha256=protocol.split_sha256,
        feature_sha256=data.feature_sha256,
        checkpoint_sha=str(evaluation_result["checkpoint_sha256"]),
        code_revision=code_revision,
        execution_backend=execution_backend,
    )
    participant = _track_b_participant_prediction_frame(recording)
    target_participant = participant[
        participant["split"].astype(str).eq(str(expected["evaluation_split"]))
    ]
    metrics = complete_metric_bundle(
        _track_b_labels_to_binary(target_participant["label_binary"]),
        target_participant["probability"].to_numpy(dtype=np.float64),
        threshold=threshold,
    )
    metric_payload = {
        **_track_b_metric_identity(
            expected,
            threshold=threshold,
            best_epoch=int(evaluation_result["best_epoch"]),
        ),
        **metrics,
        "n_recordings": int(
            recording["split"].astype(str).eq(str(expected["evaluation_split"])).sum()
        ),
        "n_participants": int(len(target_participant)),
    }
    recording_path = unit_dir / "recording_predictions.csv"
    participant_path = unit_dir / "participant_predictions.csv"
    metrics_path = unit_dir / "metrics.json"
    _atomic_dataframe_write(recording, recording_path)
    _atomic_dataframe_write(participant, participant_path)
    _atomic_json_write(metric_payload, metrics_path)
    artifacts = {
        "recording_predictions": {
            "path": str(recording_path),
            "sha256": checkpoint_sha256(recording_path),
        },
        "participant_predictions": {
            "path": str(participant_path),
            "sha256": checkpoint_sha256(participant_path),
        },
        "metrics": {
            "path": str(metrics_path),
            "sha256": checkpoint_sha256(metrics_path),
        },
        "checkpoint": {
            "path": str(evaluation_result["checkpoint_path"]),
            "sha256": str(evaluation_result["checkpoint_sha256"]),
        },
    }
    receipt = {
        **expected,
        "threshold": threshold,
        "best_epoch": int(evaluation_result["best_epoch"]),
        "metrics": metric_payload,
        "artifacts": artifacts,
    }
    _atomic_json_write(receipt, receipt_path)
    return receipt


def _load_track_b_execution_receipt(
    path: Path, *, expected: Mapping[str, object]
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Track B execution receipt cannot be read: {path}") from exc
    if not isinstance(receipt, dict):
        raise ValueError("Track B execution receipt must be a JSON object")
    required_extra = {"threshold", "best_epoch", "metrics", "artifacts"}
    if set(receipt) != set(expected) | required_extra:
        raise ValueError("Track B execution receipt schema is not exact")
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        if "execution_backend" in mismatches:
            raise ValueError(
                "execution_backend mismatch: receipt="
                f"{receipt.get('execution_backend')!r}, requested="
                f"{expected.get('execution_backend')!r}"
            )
        raise ValueError(f"Track B execution receipt provenance mismatch: {mismatches}")
    artifacts = _validate_track_b_artifacts(receipt.get("artifacts"))
    recording = pd.read_csv(Path(str(artifacts["recording_predictions"]["path"])))  # type: ignore[index]
    participant = pd.read_csv(Path(str(artifacts["participant_predictions"]["path"])))  # type: ignore[index]
    if recording.columns.tolist() != list(_TRACK_B_EXECUTION_PREDICTION_COLUMNS):
        raise ValueError("Track B recording prediction schema is invalid")
    if participant.columns.tolist() != list(_TRACK_B_EXECUTION_PREDICTION_COLUMNS):
        raise ValueError("Track B participant prediction schema is invalid")
    if not recording["analysis_unit"].astype(str).eq("recording").all():
        raise ValueError("Track B recording analysis_unit is invalid")
    if not participant["analysis_unit"].astype(str).eq("participant").all():
        raise ValueError("Track B participant analysis_unit is invalid")
    prediction_identity = {
        "run_id": expected["run_id"],
        "track": "B",
        "stage": expected["stage"],
        "protocol": expected["protocol"],
        "seed": expected["seed"],
        "model_name": expected["model_name"],
        "modality": expected["modality"],
        "configuration_sha256": expected["configuration_sha256"],
        "selected_configuration_sha256": expected[
            "selected_configuration_sha256"
        ],
        "split_sha256": expected["split_sha256"],
        "feature_sha256": expected["feature_sha256"],
        "checkpoint_sha256": artifacts["checkpoint"]["sha256"],  # type: ignore[index]
        "code_revision": expected["code_revision"],
        "execution_backend": expected["execution_backend"],
        "threshold_source": expected["threshold_source"],
    }
    for frame in (recording, participant):
        mismatched_columns = [
            column
            for column, value in prediction_identity.items()
            if not frame[column].astype(str).eq(str(value)).all()
        ]
        if mismatched_columns:
            raise ValueError(
                "Track B prediction provenance mismatch: "
                f"{mismatched_columns}"
            )
        expected_splits = {"validation", str(expected["evaluation_split"])}
        if set(frame["split"].astype(str)) != expected_splits:
            raise ValueError("Track B prediction split provenance mismatch")
        if not np.allclose(
            frame["threshold"].to_numpy(dtype=np.float64),
            float(receipt["threshold"]),
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("Track B prediction threshold provenance mismatch")
    recomputed_participant = _track_b_participant_prediction_frame(recording)
    compare_columns = [
        "participant_id",
        "dataset",
        "split",
        "modality",
        "label_binary",
        "probability",
        "n_recordings",
    ]
    actual = participant.sort_values(compare_columns[:4], kind="stable").reset_index(
        drop=True
    )
    recomputed = recomputed_participant.sort_values(
        compare_columns[:4], kind="stable"
    ).reset_index(drop=True)
    if actual[compare_columns[:-2] + ["n_recordings"]].astype(str).to_dict(
        orient="records"
    ) != recomputed[compare_columns[:-2] + ["n_recordings"]].astype(str).to_dict(
        orient="records"
    ) or not np.allclose(
        actual["probability"].to_numpy(dtype=np.float64),
        recomputed["probability"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Track B participant predictions are not recomputable")
    validation = participant[participant["split"].astype(str).eq("validation")]
    recomputed_threshold = best_threshold_by_balanced_accuracy(
        _track_b_labels_to_binary(validation["label_binary"]),
        validation["probability"].to_numpy(dtype=np.float64),
    )
    if not math.isclose(
        float(receipt["threshold"]),
        recomputed_threshold,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Track B threshold is not recomputable from validation")
    target = participant[
        participant["split"].astype(str).eq(str(expected["evaluation_split"]))
    ]
    recomputed_metrics = complete_metric_bundle(
        _track_b_labels_to_binary(target["label_binary"]),
        target["probability"].to_numpy(dtype=np.float64),
        threshold=recomputed_threshold,
    )
    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Track B receipt metrics are invalid")
    expected_metric_identity = _track_b_metric_identity(
        expected,
        threshold=float(receipt["threshold"]),
        best_epoch=int(receipt["best_epoch"]),
    )
    metric_identity_mismatches = [
        key for key, value in expected_metric_identity.items() if metrics.get(key) != value
    ]
    if metric_identity_mismatches:
        raise ValueError(
            "Track B metric provenance mismatch: "
            f"{metric_identity_mismatches}"
        )
    metric_file = json.loads(
        Path(str(artifacts["metrics"]["path"])).read_text(encoding="utf-8")  # type: ignore[index]
    )
    if metric_file != metrics:
        raise ValueError("Track B metric artifact differs from receipt")
    for key, value in recomputed_metrics.items():
        if not math.isclose(float(metrics.get(key, float("nan"))), value, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Track B metric is not recomputable: {key}")
    return receipt


def _load_track_b_selected_configurations(
    path: Path,
    *,
    candidates_dir: Path,
    data: TrackBData,
    existing_protocol: TrackBProtocol,
    code_revision: str,
    execution_backend: str,
    expected_modalities: Sequence[str],
) -> dict[str, dict[str, object]]:
    if not path.is_file():
        raise ValueError("selected_configurations.json is required before final/ladder")
    try:
        modalities, _ = _authenticate_track_b_selected_candidates(
            selected_path=path,
            candidates_dir=candidates_dir,
            protocol=existing_protocol,
            data=data,
            code_revision=code_revision,
            execution_backend=execution_backend,
        )
    except ValueError as exc:
        raise ValueError(
            "selected candidate configuration cannot be authenticated"
        ) from exc
    missing = sorted(set(expected_modalities) - set(modalities))
    if missing:
        raise ValueError(f"candidate selection is incomplete for modalities: {missing}")
    validated: dict[str, dict[str, object]] = {}
    for modality in expected_modalities:
        selected = modalities.get(modality)
        if not isinstance(selected, dict):
            raise ValueError(f"selected configuration is invalid for {modality}")
        validate_frozen_track_b_lineage(
            selected,
            feature_columns=data.feature_columns,
            feature_sha256=data.feature_sha256,
        )
        expected = {
            "split_sha256": existing_protocol.split_sha256,
            "source_sha256": data.source_sha256,
            "source_lineage_sha256": _canonical_sha256(data.source_sha256),
            "code_revision": code_revision,
            "code_sha256": _canonical_sha256(code_revision),
            "execution_backend": execution_backend,
        }
        mismatches = [key for key, value in expected.items() if selected.get(key) != value]
        if mismatches:
            if "execution_backend" in mismatches:
                raise ValueError("execution_backend mismatch in frozen Track B lineage")
            raise ValueError(f"frozen Track B lineage mismatch: {mismatches}")
        validated[modality] = selected
    return validated


def _track_b_execution_configs(
    selected: Mapping[str, object], model_name: str, seed: int
) -> tuple[ModelConfig, TrainConfig]:
    if model_name == "dndf":
        model_payload = selected.get("model_config")
        train_payload = selected.get("train_config")
    else:
        model_payload = selected.get("overall_winner_model_config")
        train_payload = selected.get("overall_winner_train_config")
    if not isinstance(model_payload, dict) or not isinstance(train_payload, dict):
        raise ValueError("frozen Track B execution configuration is invalid")
    model_config = ModelConfig(**model_payload)  # type: ignore[arg-type]
    train_values = dict(train_payload)
    train_values["seed"] = seed
    train_config = TrainConfig(**train_values)  # type: ignore[arg-type]
    if model_config.model_name != model_name:
        raise ValueError("frozen Track B model name mismatch")
    return model_config, train_config


def _track_b_expected_execution_specs(
    *,
    stage: str,
    selected: Mapping[str, Mapping[str, object]],
    final_seeds: tuple[int, ...],
) -> list[tuple[str, str, str, int]]:
    specs: list[tuple[str, str, str, int]] = []
    if stage == "final":
        for modality in sorted(selected):
            models = ["dndf"]
            if selected[modality].get("overall_winner") == "dndt":
                models.append("dndt")
            for model_name in models:
                for seed in final_seeds:
                    specs.append(("existing", modality, model_name, seed))
        return specs
    if stage == "shuffle":
        for modality in sorted(selected):
            specs.append(("shuffle", modality, "dndf", final_seeds[0]))
        return specs
    for protocol in ("early_to_late", "existing", "time_stratified"):
        for modality in sorted(selected):
            for seed in final_seeds:
                specs.append((protocol, modality, "dndf", seed))
    if "cough" in selected:
        specs.append(("external_cough", "cough", "dndf", final_seeds[0]))
    return specs


def _aggregate_track_b_execution_receipts(
    *,
    track_b_dir: Path,
    run_id: str,
    stage: str,
    protocols: Mapping[str, TrackBProtocol],
    selected: Mapping[str, Mapping[str, object]],
    data: TrackBData,
    code_revision: str,
    execution_backend: str,
    final_seeds: tuple[int, ...],
    requested_modalities: tuple[str, ...],
    requested_protocols: tuple[str, ...],
    requested_seeds: tuple[int, ...],
    execution_data_by_protocol: Mapping[str, TrackBData] | None = None,
    execution_selected_by_protocol: Mapping[
        str, Mapping[str, Mapping[str, object]]
    ]
    | None = None,
) -> dict[str, object]:
    specs = _track_b_expected_execution_specs(
        stage=stage, selected=selected, final_seeds=final_seeds
    )
    receipts: list[dict[str, object]] = []
    recording_frames: list[pd.DataFrame] = []
    participant_frames: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    authenticated: list[dict[str, str]] = []
    for protocol_name, modality, model_name, seed in specs:
        receipt_path = _track_b_execution_unit_dir(
            track_b_dir,
            stage=stage,
            protocol=protocol_name,
            modality=modality,
            model_name=model_name,
            seed=seed,
        ) / "completion.json"
        context_data = (
            execution_data_by_protocol.get(protocol_name)
            if execution_data_by_protocol is not None
            else data
        )
        context_selected = (
            execution_selected_by_protocol.get(protocol_name)
            if execution_selected_by_protocol is not None
            else selected
        )
        if context_data is None or context_selected is None:
            if receipt_path.is_file():
                raise ValueError(
                    f"cannot authenticate prior Track B unit without protocol context: {protocol_name}"
                )
            continue
        selected_modality = context_selected.get(modality)
        if selected_modality is None:
            if receipt_path.is_file():
                raise ValueError(
                    f"cannot authenticate prior Track B unit without modality context: {protocol_name}/{modality}"
                )
            continue
        model_config, train_config = _track_b_execution_configs(
            selected_modality, model_name, seed
        )
        expected = _track_b_execution_expected(
            run_id=run_id,
            stage=stage,
            protocol=protocols[protocol_name],
            modality=modality,
            model_config=model_config,
            train_config=train_config,
            selected=selected_modality,
            data=context_data,
            code_revision=code_revision,
            execution_backend=execution_backend,
        )
        receipt = _load_track_b_execution_receipt(receipt_path, expected=expected)
        if receipt is None:
            continue
        receipts.append(receipt)
        artifacts = receipt["artifacts"]
        recording_frames.append(
            pd.read_csv(Path(str(artifacts["recording_predictions"]["path"])))  # type: ignore[index]
        )
        participant_frames.append(
            pd.read_csv(Path(str(artifacts["participant_predictions"]["path"])))  # type: ignore[index]
        )
        metrics.append(dict(receipt["metrics"]))  # type: ignore[arg-type]
        authenticated.append(
            {"path": str(receipt_path), "sha256": checkpoint_sha256(receipt_path)}
        )
    recording = (
        pd.concat(recording_frames, ignore_index=True, sort=False)
        if recording_frames
        else pd.DataFrame(columns=_TRACK_B_EXECUTION_PREDICTION_COLUMNS)
    )
    participant = (
        pd.concat(participant_frames, ignore_index=True, sort=False)
        if participant_frames
        else pd.DataFrame(columns=_TRACK_B_EXECUTION_PREDICTION_COLUMNS)
    )
    metric_frame = pd.DataFrame(metrics)
    recording_path = track_b_dir / f"{stage}_recording_predictions.csv"
    participant_path = track_b_dir / f"{stage}_participant_predictions.csv"
    metrics_path = track_b_dir / f"{stage}_metrics.csv"
    _atomic_dataframe_write(recording, recording_path)
    _atomic_dataframe_write(participant, participant_path)
    _atomic_dataframe_write(metric_frame, metrics_path)
    completed_units = len(receipts)
    total_units = len(specs)
    authenticated_modalities = sorted({str(receipt["modality"]) for receipt in receipts})
    authenticated_protocols = sorted({str(receipt["protocol"]) for receipt in receipts})
    authenticated_seeds = sorted({int(receipt["seed"]) for receipt in receipts})
    result = {
        "status": "complete" if completed_units == total_units else "partial",
        "run_id": run_id,
        "track": "B",
        "stage": stage,
        "completed_units": completed_units,
        "total_units": total_units,
        "modalities": authenticated_modalities,
        "protocols": authenticated_protocols,
        "seeds": authenticated_seeds,
        "requested_scope": {
            "modalities": list(requested_modalities),
            "protocols": list(requested_protocols),
            "seeds": list(requested_seeds),
        },
        "execution_backend": execution_backend,
        "authenticated_receipts": authenticated,
        "artifacts": {
            "recording_predictions": {
                "path": str(recording_path),
                "sha256": checkpoint_sha256(recording_path),
            },
            "participant_predictions": {
                "path": str(participant_path),
                "sha256": checkpoint_sha256(participant_path),
            },
            "metrics": {
                "path": str(metrics_path),
                "sha256": checkpoint_sha256(metrics_path),
            },
        },
    }
    if stage == "prespecified_ladder_v2":
        if execution_data_by_protocol is None or execution_selected_by_protocol is None:
            raise ValueError("ladder aggregation requires protocol-specific lineage")
        lineage_payload = {
            "format_version": 1,
            "run_id": run_id,
            "stage": "prespecified_ladder_v2",
            "protocols": {
                protocol_name: {
                    "feature_sha256": context_data.feature_sha256,
                    "source_sha256": context_data.source_sha256,
                    "split_sha256": protocols[protocol_name].split_sha256,
                    "modalities": {
                        modality: {
                            key: value
                            for key, value in context_selected.items()
                            if key
                            in {
                                "selected_configuration_sha256",
                                "configuration_selection_source",
                                "feature_selection_scope",
                                "feature_selection_artifact_sha256",
                            }
                        }
                        for modality, context_selected in sorted(
                            execution_selected_by_protocol.get(
                                protocol_name, {}
                            ).items()
                        )
                    },
                }
                for protocol_name, context_data in sorted(
                    execution_data_by_protocol.items()
                )
            },
        }
        lineage_path = track_b_dir / "prespecified_ladder_v2_feature_lineage.json"
        _atomic_json_write(lineage_payload, lineage_path)
        result["artifacts"]["feature_lineage"] = {
            "path": str(lineage_path),
            "sha256": checkpoint_sha256(lineage_path),
        }
    if stage == "shuffle":
        result["control_scope"] = "single_permutation_fit_stage_sanity_check"
        result["permutation_seed"] = int(final_seeds[0])
    _atomic_json_write(result, track_b_dir / f"{stage}_manifest.json")
    return result


def _single_fusion_lineage_value(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].astype(str).drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"fusion source has inconsistent {column}: {values}")
    return values[0]


def _fusion_prediction_artifact(
    fused: pd.DataFrame,
    *,
    source: pd.DataFrame,
    run_id: str,
    seed: int,
    model_name: str,
    fusion_method: str,
    threshold: float,
    configuration_sha256: str,
    selected_configuration_sha256: str,
    state_sha256: str,
) -> pd.DataFrame:
    output = fused.copy()
    output["run_id"] = run_id
    output["track"] = "B"
    output["stage"] = "fusion"
    output["protocol"] = "existing"
    output["seed"] = int(seed)
    output["dataset"] = "coswara"
    output["model_name"] = model_name
    output["modality"] = "multimodal"
    output["threshold"] = float(threshold)
    output["threshold_source"] = "source_validation_balanced_accuracy"
    output["analysis_unit"] = "participant"
    output["configuration_sha256"] = configuration_sha256
    output["selected_configuration_sha256"] = selected_configuration_sha256
    output["split_sha256"] = _single_fusion_lineage_value(source, "split_sha256")
    output["feature_sha256"] = _single_fusion_lineage_value(source, "feature_sha256")
    output["checkpoint_sha256"] = state_sha256
    output["code_revision"] = _single_fusion_lineage_value(source, "code_revision")
    output["execution_backend"] = _single_fusion_lineage_value(
        source, "execution_backend"
    )
    counts = (
        source.groupby(["participant_id", "split"], as_index=False)["n_recordings"]
        .sum()
        .rename(columns={"n_recordings": "source_n_recordings"})
    )
    output = output.merge(
        counts,
        on=["participant_id", "split"],
        how="left",
        validate="one_to_one",
    )
    if output["source_n_recordings"].isna().any():
        raise ValueError("fusion recording counts cannot be reconstructed")
    output["n_recordings"] = output.pop("source_n_recordings").astype(np.int64)
    output["fusion_method"] = fusion_method
    columns = list(_TRACK_B_EXECUTION_PREDICTION_COLUMNS) + [
        "fusion_method",
        "available_modalities",
    ]
    return output.loc[:, columns]


def _track_b_fusion_metric(
    predictions: pd.DataFrame,
    *,
    run_id: str,
    seed: int,
    model_name: str,
    fusion_method: str,
    threshold: float,
    configuration_sha256: str,
    selected_configuration_sha256: str,
    state_sha256: str,
) -> dict[str, object]:
    test = predictions[predictions["split"].astype(str).eq("test")]
    return {
        "run_id": run_id,
        "track": "B",
        "stage": "fusion",
        "protocol": "existing",
        "seed": seed,
        "dataset": "coswara",
        "split": "test",
        "model_name": model_name,
        "modality": "multimodal",
        "fusion_method": fusion_method,
        "threshold": float(threshold),
        "threshold_source": "source_validation_balanced_accuracy",
        "analysis_unit": "participant",
        "configuration_sha256": configuration_sha256,
        "selected_configuration_sha256": selected_configuration_sha256,
        "checkpoint_sha256": state_sha256,
        **complete_metric_bundle(
            test["label_binary"].to_numpy(dtype=np.int64),
            test["probability"].to_numpy(dtype=np.float64),
            threshold=threshold,
        ),
        "n_participants": int(len(test)),
        "n_recordings": int(test["n_recordings"].sum()),
    }


def _load_track_b_fusion_unit(
    receipt_path: Path,
    *,
    expected_identity: Mapping[str, object],
    expected_source_receipts: Mapping[str, object],
    expected_state: Mapping[str, object],
    fused: pd.DataFrame,
    method_source: pd.DataFrame,
    weights: pd.DataFrame,
    threshold: float,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object], pd.DataFrame] | None:
    if not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"fusion receipt cannot be read: {receipt_path}") from exc
    if not isinstance(receipt, dict):
        raise ValueError("fusion receipt must be a JSON object")
    if set(receipt) != set(expected_identity) | {"source_receipts", "artifacts"}:
        raise ValueError("fusion receipt schema is not exact")
    mismatches = [
        key for key, value in expected_identity.items() if receipt.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"fusion receipt provenance mismatch: {mismatches}")
    if receipt.get("source_receipts") != expected_source_receipts:
        raise ValueError("fusion receipt source lineage mismatch")
    for modality, descriptor_value in expected_source_receipts.items():
        descriptor = dict(descriptor_value)  # type: ignore[arg-type]
        source_path = Path(str(descriptor["path"]))
        if not source_path.is_file() or checkpoint_sha256(source_path) != descriptor["sha256"]:
            raise ValueError(f"fusion source receipt is missing or tampered: {modality}")
        source_receipt = json.loads(source_path.read_text(encoding="utf-8"))
        checkpoint = source_receipt.get("artifacts", {}).get("checkpoint", {})
        if checkpoint.get("sha256") != descriptor["checkpoint_sha256"]:
            raise ValueError(f"fusion source checkpoint lineage mismatch: {modality}")

    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "state",
        "participant_predictions",
        "metrics",
        "weights",
    }:
        raise ValueError("fusion receipt artifact schema is not exact")
    expected_paths = {
        "state": receipt_path.parent / "state.json",
        "participant_predictions": receipt_path.parent / "participant_predictions.csv",
        "metrics": receipt_path.parent / "metrics.json",
        "weights": receipt_path.parent / "weights.csv",
    }
    for name, expected_path in expected_paths.items():
        descriptor = artifacts.get(name)
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise ValueError(f"fusion artifact descriptor is invalid: {name}")
        artifact_path = Path(str(descriptor["path"]))
        if artifact_path.resolve() != expected_path.resolve():
            raise ValueError(f"fusion artifact path mismatch: {name}")
        if (
            not artifact_path.is_file()
            or checkpoint_sha256(artifact_path) != str(descriptor["sha256"])
        ):
            raise ValueError(f"fusion artifact is missing or tampered: {name}")

    state = json.loads(expected_paths["state"].read_text(encoding="utf-8"))
    if state != expected_state:
        raise ValueError("fusion state is not reproducible from frozen inputs")
    state_sha = checkpoint_sha256(expected_paths["state"])
    decorated = _fusion_prediction_artifact(
        fused,
        source=method_source,
        run_id=str(expected_identity["run_id"]),
        seed=int(expected_identity["seed"]),
        model_name=str(expected_identity["model_name"]),
        fusion_method=str(expected_identity["fusion_method"]),
        threshold=threshold,
        configuration_sha256=str(expected_identity["configuration_sha256"]),
        selected_configuration_sha256=str(
            expected_identity["selected_configuration_sha256"]
        ),
        state_sha256=state_sha,
    )
    stored_predictions = pd.read_csv(
        expected_paths["participant_predictions"], low_memory=False
    )
    try:
        pd.testing.assert_frame_equal(
            stored_predictions,
            decorated,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-15,
        )
    except AssertionError as exc:
        raise ValueError("fusion prediction artifact is not reproducible") from exc
    stored_weights = pd.read_csv(expected_paths["weights"], low_memory=False)
    try:
        pd.testing.assert_frame_equal(
            stored_weights,
            weights,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-15,
        )
    except AssertionError as exc:
        raise ValueError("fusion weight artifact is not reproducible") from exc
    metric = _track_b_fusion_metric(
        decorated,
        run_id=str(expected_identity["run_id"]),
        seed=int(expected_identity["seed"]),
        model_name=str(expected_identity["model_name"]),
        fusion_method=str(expected_identity["fusion_method"]),
        threshold=threshold,
        configuration_sha256=str(expected_identity["configuration_sha256"]),
        selected_configuration_sha256=str(
            expected_identity["selected_configuration_sha256"]
        ),
        state_sha256=state_sha,
    )
    stored_metric = json.loads(expected_paths["metrics"].read_text(encoding="utf-8"))
    if stored_metric != metric:
        raise ValueError("fusion metric artifact is not reproducible")
    return receipt, stored_predictions, stored_metric, stored_weights


def _run_track_b_fusion(
    *,
    track_b_dir: Path,
    run_id: str,
    protocols: Mapping[str, TrackBProtocol],
    selected: Mapping[str, Mapping[str, object]],
    data: TrackBData,
    code_revision: str,
    execution_backend: str,
    final_seeds: tuple[int, ...],
    resume: bool,
) -> dict[str, object]:
    required_modalities = ("breath", "cough", "speech")
    if tuple(sorted(selected)) != required_modalities:
        raise ValueError("fusion requires frozen breath, cough, and speech lineages")
    source_frames: list[pd.DataFrame] = []
    source_receipts: dict[tuple[int, str], dict[str, object]] = {}
    for modality in required_modalities:
        for seed in final_seeds:
            model_config, train_config = _track_b_execution_configs(
                selected[modality], "dndf", seed
            )
            expected = _track_b_execution_expected(
                run_id=run_id,
                stage="final",
                protocol=protocols["existing"],
                modality=modality,
                model_config=model_config,
                train_config=train_config,
                selected=selected[modality],
                data=data,
                code_revision=code_revision,
                execution_backend=execution_backend,
            )
            receipt_path = _track_b_execution_unit_dir(
                track_b_dir,
                stage="final",
                protocol="existing",
                modality=modality,
                model_name="dndf",
                seed=seed,
            ) / "completion.json"
            receipt = _load_track_b_execution_receipt(receipt_path, expected=expected)
            if receipt is None:
                raise ValueError(
                    f"fusion requires completed final DNDF receipt: {modality}/{seed}"
                )
            source_receipts[(seed, modality)] = {
                "path": str(receipt_path),
                "sha256": checkpoint_sha256(receipt_path),
                "checkpoint_sha256": receipt["artifacts"]["checkpoint"]["sha256"],  # type: ignore[index]
            }
            participant_path = Path(
                str(receipt["artifacts"]["participant_predictions"]["path"])  # type: ignore[index]
            )
            source_frames.append(pd.read_csv(participant_path, low_memory=False))
    source_predictions = pd.concat(source_frames, ignore_index=True, sort=False)

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    weight_frames: list[pd.DataFrame] = []
    unit_receipts: list[dict[str, str]] = []
    methods = (
        (
            "validation_logistic_stack",
            "dndf_cough_speech_validation_stack",
            ("cough", "speech"),
        ),
        (
            "uniform_mean",
            "dndf_breath_cough_speech_uniform",
            required_modalities,
        ),
    )
    for seed in final_seeds:
        seed_source = source_predictions[
            source_predictions["seed"].astype(int).eq(seed)
            & source_predictions["model_name"].astype(str).eq("dndf")
        ].copy()
        if set(seed_source["modality"].astype(str)) != set(required_modalities):
            raise ValueError(f"fusion seed {seed} lacks a complete modality set")
        for fusion_method, model_name, method_modalities in methods:
            method_source = seed_source[
                seed_source["modality"].astype(str).isin(method_modalities)
            ].copy()
            if fusion_method == "validation_logistic_stack":
                fused, weights = fit_validation_logistic_fusion(
                    method_source, seed=seed, modalities=method_modalities
                )
                weight_payload = weights.to_dict(orient="records")
            else:
                fused = uniform_dndf_fusion(method_source)
                weights = pd.DataFrame(
                    [
                        {
                            "seed": seed,
                            "fusion_method": "uniform_mean",
                            "modality": modality,
                            "weight": 1.0 / len(required_modalities),
                        }
                        for modality in required_modalities
                    ]
                )
                weight_payload = weights.to_dict(orient="records")
            validation = fused[fused["split"].astype(str).eq("validation")]
            threshold = best_threshold_by_balanced_accuracy(
                validation["label_binary"].to_numpy(dtype=np.int64),
                validation["probability"].to_numpy(dtype=np.float64),
            )
            branch_receipts = {
                modality: source_receipts[(seed, modality)]
                for modality in method_modalities
            }
            selected_hash = _canonical_sha256(
                {
                    modality: selected[modality]["selected_configuration_sha256"]
                    for modality in method_modalities
                }
            )
            state = {
                "format_version": 1,
                "run_id": run_id,
                "track": "B",
                "stage": "fusion",
                "protocol": "existing",
                "seed": seed,
                "model_name": model_name,
                "fusion_method": fusion_method,
                "modalities": list(method_modalities),
                "weights": weight_payload,
                "threshold": float(threshold),
                "threshold_source": "source_validation_balanced_accuracy",
                "selected_configuration_sha256": selected_hash,
                "source_receipts": branch_receipts,
                "code_revision": code_revision,
                "execution_backend": execution_backend,
            }
            configuration_sha = _canonical_sha256(
                {
                    "format_version": 1,
                    "seed": seed,
                    "model_name": model_name,
                    "fusion_method": fusion_method,
                    "modalities": list(method_modalities),
                    "weights": weight_payload,
                    "selected_configuration_sha256": selected_hash,
                    "source_receipt_sha256": {
                        modality: branch_receipts[modality]["sha256"]
                        for modality in method_modalities
                    },
                    "code_revision": code_revision,
                }
            )
            unit_dir = track_b_dir / "fusion" / f"seed_{seed}" / fusion_method
            state_path = unit_dir / "state.json"
            prediction_path = unit_dir / "participant_predictions.csv"
            metrics_path = unit_dir / "metrics.json"
            weights_path = unit_dir / "weights.csv"
            receipt_path = unit_dir / "completion.json"
            state_payload = state | {"configuration_sha256": configuration_sha}
            receipt_identity = {
                "format_version": 1,
                "status": "complete",
                "run_id": run_id,
                "track": "B",
                "stage": "fusion",
                "protocol": "existing",
                "seed": seed,
                "model_name": model_name,
                "fusion_method": fusion_method,
                "configuration_sha256": configuration_sha,
                "selected_configuration_sha256": selected_hash,
                "code_revision": code_revision,
                "execution_backend": execution_backend,
            }
            weights = weights.copy()
            weights["protocol"] = "existing"
            weights["model_name"] = model_name
            weights["configuration_sha256"] = configuration_sha
            weights["weight_scope"] = (
                "validation_fitted"
                if fusion_method == "validation_logistic_stack"
                else "nominal_complete_case_renormalized_per_participant"
            )
            completed = (
                _load_track_b_fusion_unit(
                    receipt_path,
                    expected_identity=receipt_identity,
                    expected_source_receipts=branch_receipts,
                    expected_state=state_payload,
                    fused=fused,
                    method_source=method_source,
                    weights=weights,
                    threshold=threshold,
                )
                if resume
                else None
            )
            if completed is not None:
                _, decorated, metric, stored_weights = completed
                unit_receipts.append(
                    {
                        "path": str(receipt_path),
                        "sha256": checkpoint_sha256(receipt_path),
                    }
                )
                prediction_frames.append(decorated)
                metric_rows.append(metric)
                weight_frames.append(stored_weights)
                continue

            _atomic_json_write(state_payload, state_path)
            state_sha = checkpoint_sha256(state_path)
            decorated = _fusion_prediction_artifact(
                fused,
                source=method_source,
                run_id=run_id,
                seed=seed,
                model_name=model_name,
                fusion_method=fusion_method,
                threshold=threshold,
                configuration_sha256=configuration_sha,
                selected_configuration_sha256=selected_hash,
                state_sha256=state_sha,
            )
            metric = _track_b_fusion_metric(
                decorated,
                run_id=run_id,
                seed=seed,
                model_name=model_name,
                fusion_method=fusion_method,
                threshold=threshold,
                configuration_sha256=configuration_sha,
                selected_configuration_sha256=selected_hash,
                state_sha256=state_sha,
            )
            _atomic_dataframe_write(decorated, prediction_path)
            _atomic_json_write(metric, metrics_path)
            _atomic_dataframe_write(weights, weights_path)
            receipt = {
                **receipt_identity,
                "source_receipts": branch_receipts,
                "artifacts": {
                    "state": {"path": str(state_path), "sha256": state_sha},
                    "participant_predictions": {
                        "path": str(prediction_path),
                        "sha256": checkpoint_sha256(prediction_path),
                    },
                    "metrics": {
                        "path": str(metrics_path),
                        "sha256": checkpoint_sha256(metrics_path),
                    },
                    "weights": {
                        "path": str(weights_path),
                        "sha256": checkpoint_sha256(weights_path),
                    },
                },
            }
            _atomic_json_write(receipt, receipt_path)
            unit_receipts.append(
                {"path": str(receipt_path), "sha256": checkpoint_sha256(receipt_path)}
            )
            prediction_frames.append(decorated)
            metric_rows.append(metric)
            weight_frames.append(weights)

    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    metrics = pd.DataFrame(metric_rows)
    weights = pd.concat(weight_frames, ignore_index=True, sort=False)
    prediction_path = track_b_dir / "fusion_participant_predictions.csv"
    metrics_path = track_b_dir / "fusion_metrics.csv"
    weights_path = track_b_dir / "fusion_weights.csv"
    _atomic_dataframe_write(predictions, prediction_path)
    _atomic_dataframe_write(metrics, metrics_path)
    _atomic_dataframe_write(weights, weights_path)
    manifest = {
        "format_version": 1,
        "status": "complete",
        "run_id": run_id,
        "track": "B",
        "stage": "fusion",
        "completed_units": len(metric_rows),
        "total_units": len(final_seeds) * len(methods),
        "seeds": list(final_seeds),
        "code_revision": code_revision,
        "execution_backend": execution_backend,
        "authenticated_receipts": unit_receipts,
        "artifacts": {
            "participant_predictions": {
                "path": str(prediction_path),
                "sha256": checkpoint_sha256(prediction_path),
            },
            "metrics": {
                "path": str(metrics_path),
                "sha256": checkpoint_sha256(metrics_path),
            },
            "weights": {
                "path": str(weights_path),
                "sha256": checkpoint_sha256(weights_path),
            },
        },
    }
    _atomic_json_write(manifest, track_b_dir / "fusion_manifest.json")
    return manifest


def _validated_sorted_subset(
    name: str,
    values: Sequence[object] | None,
    *,
    default: Sequence[object],
    allowed: Sequence[object],
) -> tuple[object, ...]:
    selected = tuple(default if values is None else values)
    if not selected:
        raise ValueError(f"{name} must be nonempty")
    if tuple(sorted(set(selected))) != selected:
        raise ValueError(f"{name} must be sorted and unique")
    invalid = [value for value in selected if value not in allowed]
    if invalid:
        raise ValueError(f"{name} contains unsupported values: {invalid}")
    return selected


def _synthetic_track_b_data() -> TrackBData:
    feature_names = tuple(f"feature_{index:03d}" for index in range(800))
    rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    for participant_index in range(18):
        participant = f"smoke-p{participant_index:02d}"
        label = "positive" if participant_index % 2 else "negative"
        split = TRACK_B_SPLITS[participant_index % 3]
        metadata_rows.append(
            {
                "participant_id": participant,
                "label_binary": label,
                "split": split,
                "recording_date": f"2021-{participant_index // 6 + 1:02d}-{participant_index % 6 + 1:02d}",
            }
        )
        for modality_index, modality in enumerate(TRACK_B_MODALITIES):
            row = {
                "recording_id": f"smoke-r-{participant}-{modality}",
                "participant_id": participant,
                "dataset": "coswara",
                "modality": modality,
                "submodality": "smoke",
                "label_binary": label,
                "split": split,
            }
            row.update(
                {
                    feature: float(participant_index + modality_index + index / 1000)
                    for index, feature in enumerate(feature_names)
                }
            )
            rows.append(row)
    external_rows: list[dict[str, object]] = []
    for participant_index in range(6):
        row = {
            "recording_id": f"smoke-cv-r{participant_index}",
            "participant_id": f"smoke-cv-p{participant_index}",
            "dataset": "coughvid",
            "modality": "cough",
            "submodality": "smoke",
            "label_binary": "positive" if participant_index % 2 else "negative",
            "split": "external",
        }
        row.update(
            {
                feature: float(participant_index + index / 1000)
                for index, feature in enumerate(feature_names)
            }
        )
        external_rows.append(row)
    source_hashes = {
        name: _canonical_sha256({"smoke": name})
        for name in ("project", "external", "metadata")
    }
    return TrackBData(
        project=pd.DataFrame(rows),
        external=pd.DataFrame(external_rows),
        metadata=pd.DataFrame(metadata_rows),
        feature_columns=feature_names,
        feature_sha256=_canonical_sha256(list(feature_names)),
        source_sha256=source_hashes,
    )


def run_track_b(
    config: Mapping[str, object],
    *,
    run_id: str,
    stage: str,
    resume: bool = False,
    smoke: bool = False,
    modalities: Sequence[str] | None = None,
    protocols: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    code_revision: str,
    device: str | None = None,
    candidate_evaluator: Callable[..., Mapping[str, object]] | None = None,
    unit_evaluator: Callable[..., Mapping[str, object]] | None = None,
    feature_ranker: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    interrupt_after_receipt: tuple[str, str, str, str, int] | None = None,
) -> dict[str, object]:
    if stage not in {
        "candidates",
        "final",
        "prespecified_ladder_v2",
        "fusion",
        "shuffle",
    }:
        raise ValueError(
            "stage must be candidates, final, prespecified_ladder_v2, fusion, or shuffle"
        )
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be nonempty")
    if not isinstance(code_revision, str) or not code_revision.strip():
        raise ValueError("code_revision must be nonempty")
    expected_modalities = tuple(str(value) for value in config.get("modalities", TRACK_B_MODALITIES))
    if tuple(sorted(set(expected_modalities))) != expected_modalities or not set(
        expected_modalities
    ) <= set(TRACK_B_MODALITIES):
        raise ValueError("configured modalities must be sorted, unique, and allowed")
    requested_modalities = tuple(
        str(value)
        for value in _validated_sorted_subset(
            "modalities",
            modalities,
            default=expected_modalities,
            allowed=expected_modalities,
        )
    )
    final_seeds = tuple(int(value) for value in config.get("seeds", {}).get("final", (42, 314, 2026)))  # type: ignore[union-attr]
    if tuple(sorted(set(final_seeds))) != final_seeds:
        raise ValueError("configured final seeds must be sorted and unique")
    allowed_stage_seeds = (final_seeds[0],) if stage == "shuffle" else final_seeds
    requested_seeds = tuple(
        int(value)
        for value in _validated_sorted_subset(
            "seeds",
            seeds,
            default=allowed_stage_seeds,
            allowed=allowed_stage_seeds,
        )
    )
    ladder_protocols = (
        "early_to_late",
        "existing",
        "external_cough",
        "time_stratified",
    )
    default_protocols = (
        ladder_protocols
        if stage == "prespecified_ladder_v2"
        else (("shuffle",) if stage == "shuffle" else ("existing",))
    )
    allowed_protocols = default_protocols
    requested_protocols = tuple(
        str(value)
        for value in _validated_sorted_subset(
            "protocols",
            protocols,
            default=default_protocols,
            allowed=allowed_protocols,
        )
    )
    execution_backend = normalize_execution_backend(device or str(config.get("device", "cpu")))
    run_root = Path(str(config["run_root"]))
    track_b_dir = run_root / run_id / "track_b"
    data = (
        _synthetic_track_b_data()
        if smoke
        else load_track_b_data(
            str(config["project_features"]),
            str(config["external_features"]),
            str(config["metadata"]),
        )
    )
    protocol_data = build_track_b_protocols(data, output_dir=track_b_dir)
    if stage == "shuffle":
        shuffle_protocol = _build_track_b_shuffle_protocol(
            protocol_data["existing"], seed=final_seeds[0]
        )
        protocol_data["shuffle"] = shuffle_protocol
        split_dir = track_b_dir / "splits"
        assignments_path = split_dir / "shuffle_assignments.csv"
        summary_path = split_dir / "shuffle_summary.csv"
        _atomic_dataframe_write(shuffle_protocol.participant_assignments, assignments_path)
        _atomic_dataframe_write(shuffle_protocol.split_summary, summary_path)

    if stage == "candidates":
        selection_values = config.get("selection", {})
        if not isinstance(selection_values, Mapping):
            raise ValueError("selection configuration must be an object")
        max_trials = int(selection_values.get("max_trials_per_modality", 6))
        candidate_results: list[CandidateSelectionResult] = []
        for modality in requested_modalities:
            result = select_modality_configuration(
                protocol_data["existing"],
                feature_columns=data.feature_columns,
                feature_sha256=data.feature_sha256,
                source_sha256=data.source_sha256,
                modality=modality,
                max_trials=max_trials,
                storage=track_b_dir / "studies" / f"{modality}.sqlite3",
                output_dir=track_b_dir / "candidates",
                code_revision=code_revision,
                device=execution_backend,
                evaluator=candidate_evaluator,
                published=config.get("published", {}),  # type: ignore[arg-type]
                selection=selection_values,
                resume=resume,
                smoke=smoke,
            )
            candidate_results.append(result)
        selected_path = track_b_dir / "selected_configurations.json"
        authenticated_selected, authenticated_receipts = (
            _authenticate_track_b_selected_candidates(
                selected_path=selected_path,
                candidates_dir=track_b_dir / "candidates",
                protocol=protocol_data["existing"],
                data=data,
                code_revision=code_revision,
                execution_backend=execution_backend,
            )
        )
        completed = len(authenticated_selected)
        progress = {
            "status": "complete" if completed == len(expected_modalities) else "partial",
            "run_id": run_id,
            "track": "B",
            "stage": stage,
            "completed_units": completed,
            "total_units": len(expected_modalities),
            "modalities": sorted(authenticated_selected),
            "requested_modalities": list(requested_modalities),
            "execution_backend": execution_backend,
            "selected_configurations_sha256": checkpoint_sha256(selected_path),
            "artifacts": {
                "selected_configurations": {
                    "path": str(selected_path.relative_to(run_root / run_id)),
                    "sha256": checkpoint_sha256(selected_path),
                }
            },
            "authenticated_receipts": authenticated_receipts,
            "results": [asdict(result) for result in candidate_results],
        }
        _atomic_json_write(progress, track_b_dir / "candidates_manifest.json")
        return progress

    selected: dict[str, Mapping[str, object]]
    if stage == "prespecified_ladder_v2":
        selected = {}
    else:
        selected = _load_track_b_selected_configurations(
            track_b_dir / "selected_configurations.json",
            candidates_dir=track_b_dir / "candidates",
            data=data,
            existing_protocol=protocol_data["existing"],
            code_revision=code_revision,
            execution_backend=execution_backend,
            expected_modalities=expected_modalities,
        )
    if stage == "fusion":
        if requested_modalities != expected_modalities:
            raise ValueError("fusion stage requires all configured modalities")
        if requested_seeds != final_seeds:
            raise ValueError("fusion stage requires all configured final seeds")
        return _run_track_b_fusion(
            track_b_dir=track_b_dir,
            run_id=run_id,
            protocols=protocol_data,
            selected=selected,
            data=data,
            code_revision=code_revision,
            execution_backend=execution_backend,
            final_seeds=final_seeds,
            resume=resume,
        )
    execution_data_by_protocol: dict[str, TrackBData] | None = None
    execution_selected_by_protocol: dict[
        str, dict[str, Mapping[str, object]]
    ] | None = None
    if stage == "prespecified_ladder_v2":
        prespecified_values = config.get("prespecified_ladder_dndf")
        if not isinstance(prespecified_values, Mapping):
            raise ValueError("prespecified_ladder_dndf must be an object")
        execution_data_by_protocol = {
            "existing": data,
            "external_cough": data,
        }
        execution_selected_by_protocol = {}
        for protocol_name in ("existing", "external_cough"):
            execution_selected_by_protocol[protocol_name] = {
                modality: _track_b_prespecified_ladder_selection(
                    modality=modality,
                    protocol=protocol_data[protocol_name],
                    data=data,
                    prespecified=prespecified_values,
                    feature_selection_scope="frozen_existing_train_only_top800",
                    feature_selection_artifact_sha256=data.source_sha256["project"],
                    smoke=smoke,
                )
                for modality in expected_modalities
                if not (
                    protocol_name == "external_cough" and modality != "cough"
                )
            }
        requested_temporal = tuple(
            name
            for name in requested_protocols
            if name in {"early_to_late", "time_stratified"}
        )
        if requested_temporal:
            if smoke:
                full_project = data.project.copy()
                full_features = data.feature_columns
                full_project_sha = data.source_sha256["project"]
            else:
                full_path = config.get("project_features_full")
                if not isinstance(full_path, str) or not full_path:
                    raise ValueError(
                        "project_features_full is required for leakage-safe temporal ladder"
                    )
                full_project, full_features, full_project_sha = (
                    _load_track_b_full_project_features(
                        full_path,
                        base_data=data,
                    )
                )
            resolved_ranker = feature_ranker or _default_track_b_protocol_feature_ranker
            for protocol_name in requested_temporal:
                selected_protocol, selected_data, selection_artifact_sha = (
                    _prepare_track_b_temporal_feature_context(
                        protocol=protocol_data[protocol_name],
                        full_project=full_project,
                        full_feature_columns=full_features,
                        full_project_sha256=full_project_sha,
                        base_data=data,
                        track_b_dir=track_b_dir,
                        code_revision=code_revision,
                        feature_ranker=resolved_ranker,
                    )
                )
                protocol_data[protocol_name] = selected_protocol
                execution_data_by_protocol[protocol_name] = selected_data
                execution_selected_by_protocol[protocol_name] = {
                    modality: _track_b_prespecified_ladder_selection(
                        modality=modality,
                        protocol=selected_protocol,
                        data=selected_data,
                        prespecified=prespecified_values,
                        feature_selection_scope="protocol_train_only",
                        feature_selection_artifact_sha256=selection_artifact_sha,
                        smoke=smoke,
                    )
                    for modality in expected_modalities
                }
            del full_project
        selected = dict(execution_selected_by_protocol["existing"])
    all_specs = _track_b_expected_execution_specs(
        stage=stage, selected=selected, final_seeds=final_seeds
    )
    requested_specs = [
        spec
        for spec in all_specs
        if spec[1] in requested_modalities
        and spec[0] in requested_protocols
        and (stage != "final" or spec[3] in requested_seeds)
    ]
    for protocol_name, modality, model_name, seed in requested_specs:
        context_data = (
            execution_data_by_protocol[protocol_name]
            if execution_data_by_protocol is not None
            else data
        )
        context_selected = (
            execution_selected_by_protocol[protocol_name][modality]
            if execution_selected_by_protocol is not None
            else selected[modality]
        )
        model_config, train_config = _track_b_execution_configs(
            context_selected, model_name, seed
        )
        evaluate = unit_evaluator or _default_track_b_unit_evaluator(
            feature_sha256=context_data.feature_sha256,
            checkpoint_input_sha256=_track_b_checkpoint_input_sha256(
                context_data.feature_sha256, context_data.source_sha256
            ),
            split_sha256=protocol_data[protocol_name].split_sha256,
            code_revision=code_revision,
            execution_backend=execution_backend,
        )
        _run_or_load_track_b_execution_unit(
            track_b_dir=track_b_dir,
            run_id=run_id,
            stage=stage,
            protocol=protocol_data[protocol_name],
            modality=modality,
            model_config=model_config,
            train_config=train_config,
            selected=context_selected,
            data=context_data,
            code_revision=code_revision,
            execution_backend=execution_backend,
            resume=resume,
            evaluator=evaluate,
        )
        unit_identity = (stage, protocol_name, modality, model_name, seed)
        if interrupt_after_receipt == unit_identity:
            raise PlannedInterruption(
                "planned interruption after durable Track B receipt at "
                f"{protocol_name}/{modality}/{model_name}/{seed}"
            )
    return _aggregate_track_b_execution_receipts(
        track_b_dir=track_b_dir,
        run_id=run_id,
        stage=stage,
        protocols=protocol_data,
        selected=selected,
        data=data,
        code_revision=code_revision,
        execution_backend=execution_backend,
        final_seeds=final_seeds,
        requested_modalities=requested_modalities,
        requested_protocols=requested_protocols,
        requested_seeds=requested_seeds,
        execution_data_by_protocol=execution_data_by_protocol,
        execution_selected_by_protocol=execution_selected_by_protocol,
    )


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
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint_sha256(path)


@contextmanager
def _exclusive_rfecv_cache_lock(
    path: Path,
    *,
    timeout: float = _RFECV_LOCK_TIMEOUT_SECONDS,
    retry_interval: float = _RFECV_LOCK_RETRY_SECONDS,
):
    if timeout <= 0:
        raise ValueError("RFECV cache lock timeout must be positive")
    if retry_interval <= 0:
        raise ValueError("RFECV cache lock retry_interval must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b", buffering=0)
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        deadline = time.monotonic() + timeout
        acquired = False
        while not acquired:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "RFECV cache lock timeout after "
                        f"{timeout:.3f}s for {path}; another process may be "
                        "building the cache"
                    ) from exc
                time.sleep(retry_interval)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _track_a_rfecv_payload(artifacts: AuthorTrackAArtifacts) -> dict[str, object]:
    return {
        "source_hashes": artifacts.source_hashes,
        "rfecv_scope": "single_global_author_pass",
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


def track_a_rfecv_outer_test_exposure(
    artifacts: AuthorTrackAArtifacts,
) -> pd.DataFrame:
    """Quantify outer-test rows exposed to the global author RFECV split."""
    if not isinstance(artifacts, AuthorTrackAArtifacts):
        raise ValueError("artifacts must be AuthorTrackAArtifacts")
    n_rows = int(artifacts.features.shape[0])
    if artifacts.author_labels.shape != (n_rows,):
        raise ValueError("author labels must align with Track A feature rows")
    if set(np.unique(artifacts.author_labels).tolist()) - {0, 1}:
        raise ValueError("author labels must be binary")

    all_indices = np.arange(n_rows, dtype=np.int64)
    selection_train, selection_holdout = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=42,
    )
    selected_mask = np.zeros(n_rows, dtype=bool)
    selected_mask[np.asarray(selection_train, dtype=np.int64)] = True
    rows: list[dict[str, object]] = []
    for fold, raw_test_indices in enumerate(artifacts.test_folds):
        test_indices = np.asarray(raw_test_indices, dtype=np.int64)
        if (
            test_indices.ndim != 1
            or len(np.unique(test_indices)) != len(test_indices)
            or bool(((test_indices < 0) | (test_indices >= n_rows)).any())
        ):
            raise ValueError(f"Track A outer-test fold {fold} is invalid")
        exposed = test_indices[selected_mask[test_indices]]
        exposed_labels = artifacts.author_labels[exposed]
        exposed_positive = int(np.sum(exposed_labels == 1))
        exposed_negative = int(np.sum(exposed_labels == 0))
        rows.append(
            {
                "fold": fold,
                "n_outer_test": int(len(test_indices)),
                "n_outer_test_in_rfecv_training": int(len(exposed)),
                "outer_test_exposed_fraction": (
                    float(len(exposed) / len(test_indices))
                    if len(test_indices)
                    else float("nan")
                ),
                "exposed_author_positive": exposed_positive,
                "exposed_author_negative": exposed_negative,
                "exposed_covid_positive": exposed_negative,
                "exposed_covid_negative": exposed_positive,
                "rfecv_selection_train_n": int(len(selection_train)),
                "rfecv_selection_holdout_n": int(len(selection_holdout)),
                "selection_random_state": 42,
                "selection_test_fraction": 0.20,
                "selection_scope": "single_global_author_pass",
            }
        )
    return pd.DataFrame(rows)


def _load_valid_track_a_cache(
    cache_dir: Path,
    *,
    expected_payload: Mapping[str, object],
    cache_key: str,
    expected_rows: int,
    expected_indices: np.ndarray | None,
) -> TrackAFeatureCache | None:
    manifest_path = cache_dir / "manifest.json"
    features_path = cache_dir / "selected_features.npy"
    indices_path = cache_dir / "selected_indices.npy"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return None
        expected_manifest_fields = set(expected_payload) | {
            "cache_key",
            "selected_count",
            "selected_indices",
            "selected_features_shape",
            "selected_features_sha256",
            "selected_indices_sha256",
        }
        if set(manifest) != expected_manifest_fields:
            return None
        manifest_payload = {
            key: manifest.get(key) for key in expected_payload
        }
        if manifest_payload != dict(expected_payload):
            return None
        if _canonical_sha256(manifest_payload) != cache_key:
            return None
        if manifest.get("cache_key") != cache_key:
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
    if manifest.get("selected_count") != len(indices):
        return None
    if manifest.get("selected_indices") != indices.tolist():
        return None
    if manifest.get("selected_features_shape") != list(selected_features.shape):
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
    cache_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_rfecv_cache_lock(cache_dir / ".rfecv-cache.lock"):
        cached = _load_valid_track_a_cache(
            cache_dir,
            expected_payload=payload,
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
        selected_indices = np.flatnonzero(
            np.asarray(selector.support_, dtype=bool)
        ).astype(np.int64)
        if expected is not None and not np.array_equal(selected_indices, expected):
            raise RuntimeError(
                "RFECV selected indices differ from the pinned author reconstruction: "
                f"expected {expected.tolist()}, found {selected_indices.tolist()}"
            )
        selected_features = np.asarray(
            artifacts.features[:, selected_indices], dtype=np.float64
        )
        selected_features_sha256 = _atomic_numpy_save(
            selected_features, cache_dir / "selected_features.npy"
        )
        selected_indices_sha256 = _atomic_numpy_save(
            selected_indices, cache_dir / "selected_indices.npy"
        )
        manifest = {
            **payload,
            "cache_key": cache_key,
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
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_checkpoint_graph(value: object) -> object:
    """Build a deterministic graph accepted by torch's restricted loader."""
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
            return {
                "__covid_rars_safe_type__": "numpy.ndarray",
                "dtype": item.dtype.str,
                "shape": list(item.shape),
                "values": item.tolist(),
            }
        if isinstance(item, np.generic):
            return rebuild(item.item())
        if isinstance(item, dict):
            return {rebuild(key): rebuild(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return tuple(rebuild(child) for child in item)
        if isinstance(item, list):
            return [rebuild(child) for child in item]
        raise TypeError(
            "checkpoint payload contains an unsupported unsafe type: "
            f"{type(item).__module__}.{type(item).__qualname__}"
        )

    return rebuild(value)


def _restore_checkpoint_graph(value: object) -> object:
    if isinstance(value, dict):
        if value.get("__covid_rars_safe_type__") == "numpy.ndarray":
            if set(value) != {
                "__covid_rars_safe_type__",
                "dtype",
                "shape",
                "values",
            }:
                raise ValueError("checkpoint NumPy payload schema is invalid")
            dtype = value.get("dtype")
            shape = value.get("shape")
            if not isinstance(dtype, str) or not isinstance(shape, list):
                raise ValueError("checkpoint NumPy payload metadata is invalid")
            array = np.asarray(value.get("values"), dtype=np.dtype(dtype))
            expected_shape = tuple(int(dimension) for dimension in shape)
            if array.shape != expected_shape:
                raise ValueError("checkpoint NumPy payload shape is invalid")
            return array
        return {
            _restore_checkpoint_graph(key): _restore_checkpoint_graph(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_restore_checkpoint_graph(child) for child in value)
    if isinstance(value, list):
        return [_restore_checkpoint_graph(child) for child in value]
    if value is None or isinstance(value, (bool, int, float, str, torch.Tensor)):
        return value
    raise ValueError(
        "restricted checkpoint contained an unsupported decoded type: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _read_verified_checkpoint_bytes(path: Path, expected_sha256: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint is missing or is not a regular file")
    try:
        with path.open("rb") as handle:
            payload = handle.read()
    except OSError as exc:
        raise ValueError("checkpoint bytes cannot be read") from exc
    if hashlib.sha256(payload).hexdigest() != expected_sha256.lower():
        raise ValueError("checkpoint SHA256 mismatch")
    return payload


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
        try:
            checkpoint_bytes = _read_verified_checkpoint_bytes(
                path,
                str(descriptor["sha256"]),
            )
            loaded = torch.load(
                io.BytesIO(checkpoint_bytes),
                map_location=map_location,
                weights_only=True,
            )
            payload = _restore_checkpoint_graph(loaded)
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
        "covid_prediction": covid_prediction,
    }


def author_threshold_to_covid_threshold(author_threshold: float) -> float:
    if not math.isfinite(author_threshold) or not 0.0 <= author_threshold < 1.0:
        raise ValueError("author_threshold must be finite and within [0, 1)")
    return 1.0 - float(author_threshold)


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


def _participant_validation_arrays(
    labels: np.ndarray,
    probabilities: np.ndarray,
    participant_ids: object | None,
) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if labels_array.ndim != 1 or probability_array.ndim != 1:
        raise ValueError("validation labels and probabilities must be one-dimensional")
    if labels_array.shape != probability_array.shape:
        raise ValueError("validation labels and probabilities must have equal length")
    if participant_ids is None:
        return labels_array, probability_array
    identifiers = np.asarray(participant_ids)
    if identifiers.ndim != 1 or identifiers.shape[0] != labels_array.shape[0]:
        raise ValueError("validation_group_ids must match validation rows")
    normalized_ids = pd.Series(identifiers).astype(str).str.strip()
    if normalized_ids.eq("").any() or pd.isna(identifiers).any():
        raise ValueError("validation_group_ids must be present and nonempty")
    frame = pd.DataFrame(
        {
            "participant_id": normalized_ids.to_numpy(),
            "label": labels_array,
            "probability": probability_array,
        }
    )
    conflicts = frame.groupby("participant_id", sort=False)["label"].nunique()
    if (conflicts != 1).any():
        raise ValueError("conflicting validation labels within participant")
    participant = frame.groupby("participant_id", sort=False, as_index=False).agg(
        label=("label", "first"),
        probability=("probability", "mean"),
    )
    return (
        participant["label"].to_numpy(dtype=np.int64),
        participant["probability"].to_numpy(dtype=np.float64),
    )


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
    validation_group_ids: object | None = None,
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
    selection_validation_labels, _ = _participant_validation_arrays(
        validation_labels_array,
        np.zeros(validation_labels_array.shape[0], dtype=np.float64),
        validation_group_ids,
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
        selection_validation_labels, selection_validation_probability = (
            _participant_validation_arrays(
                validation_labels_array,
                validation_probability,
                validation_group_ids,
            )
        )
        epoch_auroc, epoch_auprc = _safe_validation_metrics(
            selection_validation_labels, selection_validation_probability
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
    selection_validation_labels, selection_validation_probability = (
        _participant_validation_arrays(
            validation_labels_array,
            validation_probability,
            validation_group_ids,
        )
    )
    validation_auroc, validation_auprc = _safe_validation_metrics(
        selection_validation_labels, selection_validation_probability
    )
    threshold = best_threshold_by_balanced_accuracy(
        selection_validation_labels, selection_validation_probability
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
            "validation_analysis_unit": (
                "participant" if validation_group_ids is not None else "row"
            ),
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
    checkpoint_path: Path
    checkpoint_sha256: str
    balancing_audit: dict[str, object]
    history: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _TrackAFixedEpochFit:
    model: NeuralDecisionClassifier
    checkpoint_path: Path
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
    execution_backend = normalize_execution_backend(device)
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
        "execution_backend": execution_backend,
        "external_scaler": False,
    }
    start_epoch = 1
    best_epoch = -1
    best_auroc = float("nan")
    best_auprc = float("nan")
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    history: list[dict[str, object]] = []
    if resume and _manifest_path(checkpoint_dir, "latest_recovery").is_file():
        recovered = _resolve_checkpoint_manifest(
            checkpoint_dir, role="latest_recovery", map_location="cpu"
        )
        payload = recovered.payload
        if payload.get("execution_backend") != execution_backend:
            raise ValueError(
                "execution_backend mismatch: checkpoint="
                f"{payload.get('execution_backend')!r}, requested={execution_backend!r}"
            )
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
        recovered_history = payload.get("history")
        if not isinstance(recovered_history, list):
            raise ValueError("corrected Track A checkpoint history is invalid")
        history = copy.deepcopy(recovered_history)
        _restore_rng_state(payload["rng_state"])
        if no_improvement >= patience:
            start_epoch = max_epochs + 1

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
        history.append(
            {
                "epoch": epoch,
                "validation_auroc": epoch_auroc,
                "validation_auprc": epoch_auprc,
                "improved": improved,
                "no_improvement": no_improvement,
            }
        )
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
            "history": history,
            "rng_state": _rng_state(),
            "balancing_audit": balanced.audit,
            "execution_backend": execution_backend,
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
            "execution_backend": execution_backend,
            "history": history,
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
        checkpoint_path=inference.path,
        checkpoint_sha256=str(inference.descriptor["sha256"]),
        balancing_audit=balanced.audit,
        history=tuple(copy.deepcopy(history)),
    )


def _track_a_fresh_fold_author_fit(
    training_features: np.ndarray,
    training_author_labels: np.ndarray,
    *,
    model_config: ModelConfig,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    reconstruction_seed: int,
    checkpoint_dir: Path,
    feature_sha256: str,
    split_sha256: str,
    code_revision: str,
    device: torch.device,
    fold: int,
    resume: bool,
    interrupt_after_epoch: int | None,
) -> _TrackAFixedEpochFit:
    execution_backend = normalize_execution_backend(device)
    balanced = balance_training_rows(
        training_features,
        training_author_labels,
        method="svm_smote",
        seed=reconstruction_seed,
    )
    mode = "fresh_fold_author_protocol"
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
        "reconstruction_seed": reconstruction_seed,
        "feature_sha256": feature_sha256,
        "split_sha256": split_sha256,
        "code_revision": code_revision,
        "execution_backend": execution_backend,
        "external_scaler": False,
        "training_order": "no_shuffle_repeated_dataset",
        "threshold_source": "test_balanced_accuracy_author_audit",
    }
    start_epoch = 1
    if resume and _manifest_path(checkpoint_dir, "latest_recovery").is_file():
        recovered = _resolve_checkpoint_manifest(
            checkpoint_dir, role="latest_recovery", map_location="cpu"
        )
        payload = recovered.payload
        if payload.get("track_a_fingerprint") != fingerprint:
            raise ValueError("fresh-fold author protocol resume fingerprint mismatch")
        model.load_state_dict(payload["model_state"])  # type: ignore[arg-type]
        optimizer.load_state_dict(payload["optimizer_state"])  # type: ignore[arg-type]
        _move_optimizer_state(optimizer, device)
        _restore_rng_state(payload["rng_state"])
        start_epoch = int(payload["completed_epoch"]) + 1
    batches = fixed_order_batch_indices(len(balanced.labels), batch_size)
    for epoch in range(start_epoch, max_epochs + 1):
        _track_a_train_epoch(
            model,
            optimizer,
            balanced.features,
            balanced.labels,
            batches=batches,
            device=device,
        )
        _publish_checkpoint_generation(
            {
                "format_version": 1,
                "checkpoint_role": "latest_recovery",
                "track_a_fingerprint": fingerprint,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "completed_epoch": epoch,
                "rng_state": _rng_state(),
                "balancing_audit": balanced.audit,
                "execution_backend": execution_backend,
            },
            checkpoint_dir,
            role="latest_recovery",
            epoch=epoch,
        )
        if interrupt_after_epoch == epoch:
            raise PlannedInterruption(
                f"planned Track A interruption at {mode}/{fold}/epoch {epoch}"
            )
    inference = _publish_checkpoint_generation(
        {
            "format_version": 1,
            "checkpoint_role": "best_inference",
            "track_a_fingerprint": fingerprint,
            "model_state": model.state_dict(),
            "completed_epoch": max_epochs,
            "balancing_audit": balanced.audit,
            "execution_backend": execution_backend,
        },
        checkpoint_dir,
        role="best_inference",
        epoch=max_epochs,
    )
    return _TrackAFixedEpochFit(
        model=model,
        checkpoint_path=inference.path,
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
    predicted_label: np.ndarray,
    threshold: float,
    threshold_comparator: str,
    threshold_source: str,
    configuration_sha256: str,
    split_sha256: str,
    feature_sha256: str,
    feature_indices_sha256: str,
    feature_cache_key: str,
    checkpoint_sha: str,
    author_commit: str,
    code_revision: str,
    execution_backend: str,
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
        "predicted_label": predicted_label.astype(np.int64),
        "threshold": threshold,
        "threshold_comparator": threshold_comparator,
        "threshold_source": threshold_source,
        "configuration_sha256": configuration_sha256,
        "split_sha256": split_sha256,
        "feature_sha256": feature_sha256,
        "checkpoint_sha256": checkpoint_sha,
        "analysis_id": analysis_ids,
        "analysis_unit": "author_sample",
        "mode": mode,
        "author_commit": author_commit,
        "feature_indices_sha256": feature_indices_sha256,
        "feature_cache_key": feature_cache_key,
        "code_revision": code_revision,
        "execution_backend": execution_backend,
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
    code_revision: str,
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
    predicted = frame["predicted_label"].to_numpy(dtype=np.int64)
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
        "protocol": str(frame["protocol"].iloc[0]),
        "mode": mode,
        "model_name": str(frame["model_name"].iloc[0]),
        "fold": int(frame["fold"].iloc[0]),
        "paper_thresholded_auc": paper_thresholded_auc,
        "author_threshold": author_threshold,
        "threshold_source": str(frame["threshold_source"].iloc[0]),
        "threshold_comparator": str(frame["threshold_comparator"].iloc[0]),
        "threshold_selected_on_outer_test": mode
        in {"author_behaviour_audit", "fresh_fold_author_protocol"},
        "model_reinitialized_per_fold": model_reinitialized,
        "optimizer_reinitialized_per_fold": model_reinitialized,
        "author_training_order": (
            "no_shuffle_repeated_dataset"
            if mode in {"author_behaviour_audit", "fresh_fold_author_protocol"}
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
        "configuration_sha256": str(frame["configuration_sha256"].iloc[0]),
        "feature_sha256": feature_cache.selected_features_sha256,
        "feature_indices_sha256": feature_cache.selected_indices_sha256,
        "feature_cache_key": feature_cache.cache_key,
        "split_sha256": fold_hash,
        "fold_sha256": fold_hash,
        "author_commit": author_commit,
        "code_revision": code_revision,
        "execution_backend": str(frame["execution_backend"].iloc[0]),
        "global_feature_selection_retained": True,
        "released_preprocessed_array_retained": True,
        "external_standard_scaler": False,
        "balancing_audit": dict(balancing_audit),
    }


def _track_a_relative_artifact_path(path: Path, fold_dir: Path) -> str:
    try:
        relative = path.resolve().relative_to(fold_dir.resolve())
    except ValueError as exc:
        raise ValueError("Track A fold artifacts must remain inside their fold directory") from exc
    return relative.as_posix()


def _track_a_receipt_identity(
    context: TrackAFoldContext,
    fold_dir: Path,
) -> dict[str, object]:
    predictions_path = fold_dir / "predictions.csv"
    metrics_path = fold_dir / "metrics.json"
    identity = {
        "status": "complete",
        "run_id": context.run_id,
        "track": "A",
        "mode": context.mode,
        "protocol": context.protocol,
        "fold": context.fold,
        "model_name": context.model_name,
        "author_commit": context.author_commit,
        "feature_sha256": context.feature_sha256,
        "feature_indices_sha256": context.feature_indices_sha256,
        "feature_cache_key": context.feature_cache_key,
        "configuration_sha256": context.configuration_sha256,
        "split_sha256": context.split_sha256,
        "code_revision": context.code_revision,
        "execution_backend": context.execution_backend,
        "threshold_source": context.threshold_source,
        "threshold_comparator": context.threshold_comparator,
        "predictions_path": "predictions.csv",
        "metrics_path": "metrics.json",
        "checkpoint_path": _track_a_relative_artifact_path(
            context.checkpoint_path,
            fold_dir,
        ),
        "checkpoint_sha256": context.checkpoint_sha256,
        "predictions_sha256": checkpoint_sha256(predictions_path),
        "metrics_sha256": checkpoint_sha256(metrics_path),
        "global_feature_selection_retained": True,
        "released_preprocessed_array_retained": True,
        "author_training_order": context.author_training_order,
        "author_randomness_unseeded": True,
        "reconstruction_seed": context.reconstruction_seed,
    }
    return identity


def _require_track_a_prediction_identity(
    predictions: pd.DataFrame,
    context: TrackAFoldContext,
) -> None:
    expected = {
        "run_id": context.run_id,
        "track": "A",
        "protocol": context.protocol,
        "fold": context.fold,
        "model_name": context.model_name,
        "mode": context.mode,
        "author_commit": context.author_commit,
        "threshold_source": context.threshold_source,
        "configuration_sha256": context.configuration_sha256,
        "split_sha256": context.split_sha256,
        "feature_sha256": context.feature_sha256,
        "feature_indices_sha256": context.feature_indices_sha256,
        "feature_cache_key": context.feature_cache_key,
        "code_revision": context.code_revision,
        "execution_backend": context.execution_backend,
        "checkpoint_sha256": context.checkpoint_sha256,
        "threshold_comparator": context.threshold_comparator,
    }
    if predictions.empty:
        raise ValueError("authenticated fold predictions cannot be empty")
    for field, value in expected.items():
        if field not in predictions or not predictions[field].eq(value).all():
            raise ValueError(
                f"authenticated fold prediction identity mismatch for {field}"
            )


def _require_track_a_metric_identity(
    metric: Mapping[str, object],
    context: TrackAFoldContext,
) -> None:
    expected = {
        "run_id": context.run_id,
        "track": "A",
        "protocol": context.protocol,
        "mode": context.mode,
        "fold": context.fold,
        "model_name": context.model_name,
        "author_commit": context.author_commit,
        "feature_sha256": context.feature_sha256,
        "feature_indices_sha256": context.feature_indices_sha256,
        "feature_cache_key": context.feature_cache_key,
        "configuration_sha256": context.configuration_sha256,
        "split_sha256": context.split_sha256,
        "code_revision": context.code_revision,
        "execution_backend": context.execution_backend,
        "threshold_source": context.threshold_source,
        "threshold_comparator": context.threshold_comparator,
        "checkpoint_sha256": context.checkpoint_sha256,
        "global_feature_selection_retained": True,
        "released_preprocessed_array_retained": True,
        "author_training_order": context.author_training_order,
        "author_randomness_unseeded": True,
        "reconstruction_seed": context.reconstruction_seed,
    }
    for field, value in expected.items():
        if metric.get(field) != value:
            raise ValueError(f"authenticated fold metric identity mismatch for {field}")


def _require_track_a_metric_recomputation(
    predictions: pd.DataFrame,
    metric: Mapping[str, object],
) -> None:
    labels = predictions["label_binary"].to_numpy(dtype=np.int64)
    probabilities = predictions["probability"].to_numpy(dtype=np.float64)
    thresholds = pd.to_numeric(predictions["threshold"], errors="coerce").unique()
    comparators = predictions["threshold_comparator"].astype(str).unique()
    if len(thresholds) != 1 or not np.isfinite(thresholds[0]):
        raise ValueError("authenticated fold metric recomputation mismatch for threshold")
    if len(comparators) != 1 or comparators[0] not in {"ge", "gt"}:
        raise ValueError(
            "authenticated fold metric recomputation mismatch for threshold comparator"
        )
    threshold = float(thresholds[0])
    predicted = (
        probabilities >= threshold
        if comparators[0] == "ge"
        else probabilities > threshold
    ).astype(np.int64)
    saved_predicted = predictions["predicted_label"].to_numpy(dtype=np.int64)
    if not np.array_equal(predicted, saved_predicted):
        raise ValueError(
            "authenticated fold metric recomputation mismatch for predicted_label"
        )

    recomputed = binary_metric_bundle(labels, probabilities, threshold=threshold)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    recomputed.update(
        {
            "balanced_accuracy": 0.5
            * (tp / max(1, tp + fn) + tn / max(1, tn + fp)),
            "f1": 2.0 * tp / max(1, 2 * tp + fp + fn),
            "sensitivity": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    )
    for field in (
        "auroc",
        "auprc",
        "balanced_accuracy",
        "f1",
        "sensitivity",
        "specificity",
        "brier",
        "ece",
        "nll",
        "threshold",
        "n_samples",
    ):
        stored = metric.get(field)
        expected_value = float(recomputed[field])
        if (
            not isinstance(stored, (int, float))
            or isinstance(stored, bool)
            or not math.isclose(
                float(stored), expected_value, rel_tol=1e-10, abs_tol=1e-12
            )
        ):
            raise ValueError(
                f"authenticated fold metric recomputation mismatch for {field}"
            )
    for field in ("tn", "fp", "fn", "tp"):
        if metric.get(field) != recomputed[field]:
            raise ValueError(
                f"authenticated fold metric recomputation mismatch for {field}"
            )
    paper_thresholded_auc = metric.get("paper_thresholded_auc")
    if paper_thresholded_auc is not None and not math.isclose(
        float(paper_thresholded_auc),
        float(recomputed["balanced_accuracy"]),
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "authenticated fold metric recomputation mismatch for paper_thresholded_auc"
        )


def _write_track_a_fold_outputs(
    fold_dir: Path,
    predictions: pd.DataFrame,
    metric: dict[str, object],
    *,
    context: TrackAFoldContext,
) -> Path:
    predictions_path = fold_dir / "predictions.csv"
    metrics_path = fold_dir / "metrics.json"
    if not context.checkpoint_path.is_file():
        raise ValueError("authenticated fold checkpoint is missing")
    if checkpoint_sha256(context.checkpoint_path) != context.checkpoint_sha256:
        raise ValueError("authenticated fold checkpoint hash mismatch")
    _require_track_a_prediction_identity(predictions, context)
    _require_track_a_metric_identity(metric, context)
    write_predictions(predictions, predictions_path)
    serializable_metric = {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in metric.items()
    }
    _atomic_json_write(serializable_metric, metrics_path)
    receipt = _track_a_receipt_identity(context, fold_dir)
    receipt_path = fold_dir / "receipt.json"
    _atomic_json_write(receipt, receipt_path)
    return receipt_path


def _load_completed_track_a_fold(
    fold_dir: Path,
    *,
    expected: TrackAFoldContext,
) -> tuple[pd.DataFrame, dict[str, object]] | None:
    receipt_path = fold_dir / "receipt.json"
    predictions_path = fold_dir / "predictions.csv"
    metrics_path = fold_dir / "metrics.json"
    present = tuple(
        path.is_file() for path in (receipt_path, predictions_path, metrics_path)
    )
    if not any(present):
        return None
    if not all(present):
        raise ValueError("authenticated fold output set is incomplete")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise ValueError("authenticated fold receipt must be a JSON object")
        expected_receipt = _track_a_receipt_identity(expected, fold_dir)
        if set(receipt) != set(expected_receipt):
            raise ValueError("authenticated fold receipt schema mismatch")
        for field, value in expected_receipt.items():
            if receipt.get(field) != value:
                raise ValueError(
                    f"authenticated fold receipt identity mismatch for {field}"
                )
        if checkpoint_sha256(predictions_path) != receipt.get("predictions_sha256"):
            raise ValueError("authenticated fold prediction hash mismatch")
        if checkpoint_sha256(metrics_path) != receipt.get("metrics_sha256"):
            raise ValueError("authenticated fold metric hash mismatch")
        if not expected.checkpoint_path.is_file():
            raise ValueError("authenticated fold checkpoint is missing")
        if checkpoint_sha256(expected.checkpoint_path) != expected.checkpoint_sha256:
            raise ValueError("authenticated fold checkpoint hash mismatch")
        metric = json.loads(metrics_path.read_text(encoding="utf-8"))
        predictions = validate_prediction_frame(pd.read_csv(predictions_path))
        if not isinstance(metric, dict):
            raise ValueError("authenticated fold metric must be a JSON object")
        _require_track_a_prediction_identity(predictions, expected)
        _require_track_a_metric_identity(metric, expected)
        _require_track_a_metric_recomputation(predictions, metric)
    except ValueError as exc:
        if str(exc).startswith("authenticated fold"):
            raise
        raise ValueError("authenticated fold output identity is invalid") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("authenticated fold output cannot be read") from exc
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


def _track_a_author_split_sha256(
    artifacts: AuthorTrackAArtifacts,
    fold: int,
) -> str:
    return _canonical_sha256(
        {
            "fold": fold,
            "train_hash": artifacts.fold_hashes.get(
                f"Train-Test Split/coswaradataset/train/{fold}.csv",
                _canonical_sha256(artifacts.train_folds[fold].tolist()),
            ),
            "test_hash": artifacts.fold_hashes.get(
                f"Train-Test Split/coswaradataset/test/{fold}.csv",
                _canonical_sha256(artifacts.test_folds[fold].tolist()),
            ),
        }
    )


def _track_a_corrected_split(
    artifacts: AuthorTrackAArtifacts,
    fold: int,
    reconstruction_seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
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
    return (
        np.asarray(inner_train, dtype=np.int64),
        np.asarray(inner_validation, dtype=np.int64),
        split_sha,
    )


def _track_a_mode_configuration(
    *,
    mode: str,
    model_name: str,
    model_config: ModelConfig,
    published: Mapping[str, object],
    patience: int,
    reconstruction_seed: int,
    feature_cache_key: str,
    code_revision: str,
    execution_backend: str,
) -> dict[str, object]:
    return {
        "mode": mode,
        "model_name": model_name,
        "model_config": asdict(model_config),
        "published": dict(published),
        "patience": patience,
        "reconstruction_seed": reconstruction_seed,
        "author_training_order": (
            "no_shuffle_repeated_dataset"
            if mode in {"author_behaviour_audit", "fresh_fold_author_protocol"}
            else "deterministic_per_epoch_shuffle"
        ),
        "author_randomness_unseeded": True,
        "feature_cache_key": feature_cache_key,
        "code_revision": code_revision,
        "execution_backend": execution_backend,
    }


def _resolve_track_a_fold_inference(fold_dir: Path) -> ResolvedCheckpoint:
    try:
        return _resolve_checkpoint_manifest(
            fold_dir / "checkpoints",
            role="best_inference",
            map_location="cpu",
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ValueError("authenticated fold checkpoint is missing or invalid") from exc


def _track_a_author_fold_context(
    *,
    run_dir: Path,
    run_id: str,
    fold: int,
    model_name: str,
    artifacts: AuthorTrackAArtifacts,
    feature_cache: TrackAFeatureCache,
    configuration_sha256: str,
    code_revision: str,
    reconstruction_seed: int,
    execution_backend: str,
) -> TrackAFoldContext:
    fold_dir = (
        run_dir
        / "track_a"
        / "author_behaviour_audit"
        / model_name
        / f"fold_{fold:02d}"
    )
    inference = _resolve_track_a_fold_inference(fold_dir)
    return TrackAFoldContext(
        run_id=run_id,
        mode="author_behaviour_audit",
        fold=fold,
        model_name=model_name,
        author_commit=artifacts.author_commit,
        feature_sha256=feature_cache.selected_features_sha256,
        feature_indices_sha256=feature_cache.selected_indices_sha256,
        feature_cache_key=feature_cache.cache_key,
        configuration_sha256=configuration_sha256,
        split_sha256=_track_a_author_split_sha256(artifacts, fold),
        code_revision=code_revision,
        execution_backend=execution_backend,
        threshold_source="test_balanced_accuracy_author_audit",
        threshold_comparator="gt",
        reconstruction_seed=reconstruction_seed,
        checkpoint_path=inference.path,
        checkpoint_sha256=str(inference.descriptor["sha256"]),
    )


def _track_a_receipt_state_binding(
    *,
    run_dir: Path,
    fold_dir: Path,
    context: TrackAFoldContext,
) -> dict[str, object]:
    receipt_path = fold_dir / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError("contiguous authenticated receipt chain is incomplete")
    receipt_identity = _track_a_receipt_identity(context, fold_dir)
    return {
        "completed_receipt_path": _track_a_relative_artifact_path(
            receipt_path,
            run_dir,
        ),
        "completed_receipt_sha256": checkpoint_sha256(receipt_path),
        "completed_receipt_identity": receipt_identity,
        "completed_receipt_identity_sha256": _canonical_sha256(receipt_identity),
    }


def _validate_track_a_author_receipt_chain(
    *,
    run_dir: Path,
    run_id: str,
    next_fold: int,
    model_name: str,
    artifacts: AuthorTrackAArtifacts,
    feature_cache: TrackAFeatureCache,
    configuration_sha256: str,
    code_revision: str,
    reconstruction_seed: int,
    execution_backend: str,
    state: Mapping[str, object],
) -> tuple[dict[int, tuple[pd.DataFrame, dict[str, object]]], dict[str, object]]:
    completed: dict[int, tuple[pd.DataFrame, dict[str, object]]] = {}
    latest_binding: dict[str, object] = {}
    for fold in range(next_fold):
        fold_dir = (
            run_dir
            / "track_a"
            / "author_behaviour_audit"
            / model_name
            / f"fold_{fold:02d}"
        )
        try:
            context = _track_a_author_fold_context(
                run_dir=run_dir,
                run_id=run_id,
                fold=fold,
                model_name=model_name,
                artifacts=artifacts,
                feature_cache=feature_cache,
                configuration_sha256=configuration_sha256,
                code_revision=code_revision,
                reconstruction_seed=reconstruction_seed,
                execution_backend=execution_backend,
            )
            loaded = _load_completed_track_a_fold(fold_dir, expected=context)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise ValueError(
                "contiguous authenticated receipt chain is missing or invalid"
            ) from exc
        if loaded is None:
            raise ValueError(
                "contiguous authenticated receipt chain is missing or invalid"
            )
        completed[fold] = loaded
        latest_binding = _track_a_receipt_state_binding(
            run_dir=run_dir,
            fold_dir=fold_dir,
            context=context,
        )
    if next_fold > 0:
        for field, value in latest_binding.items():
            if state.get(field) != value:
                raise ValueError(
                    "author between-fold state is not bound to the immediately "
                    f"preceding receipt: {field}"
                )
    return completed, latest_binding


def _publish_track_a_author_between_fold_state(
    *,
    state_dir: Path,
    configuration_sha256: str,
    model: NeuralDecisionClassifier,
    optimizer: torch.optim.Optimizer,
    completed_fold: int,
    fold_batch: Sequence[int],
    epochs_per_fold: int,
    receipt_binding: Mapping[str, object],
    execution_backend: str,
) -> ResolvedCheckpoint:
    return _publish_checkpoint_generation(
        {
            "format_version": 1,
            "checkpoint_role": "latest_recovery",
            "track_a_configuration_sha256": configuration_sha256,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "next_fold": completed_fold + 1,
            "completed_epoch": 0,
            "phase": "between_folds",
            "fold_batch": list(fold_batch),
            "rng_state": _rng_state(),
            "execution_backend": execution_backend,
            **receipt_binding,
        },
        state_dir,
        role="latest_recovery",
        epoch=(completed_fold + 1) * epochs_per_fold,
    )


def run_track_a(
    config: Mapping[str, object],
    *,
    run_id: str,
    resume: bool = False,
    smoke: bool = False,
    fold_batch: Sequence[int] | None = None,
    modes: Sequence[str] = (
        "author_behaviour_audit",
        "fresh_fold_author_protocol",
        "corrected_reference",
    ),
    model_names: Sequence[str] = ("dndt", "dndf"),
    artifacts: AuthorTrackAArtifacts | None = None,
    feature_cache: TrackAFeatureCache | None = None,
    code_revision: str,
    device: str | torch.device | None = None,
    interrupt_after: tuple[str, str, int, int] | None = None,
    interrupt_after_receipt: tuple[str, str, int] | None = None,
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
    valid_modes = {
        "author_behaviour_audit",
        "fresh_fold_author_protocol",
        "corrected_reference",
    }
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
        expected_commit = config.get("author_commit")
        if not isinstance(author_repo, str) or not author_repo:
            raise ValueError("config author_repo must be a nonempty path")
        if expected_commit != TRACK_A_AUTHOR_COMMIT:
            raise ValueError(
                "config author_commit must equal the pinned author commit "
                f"{TRACK_A_AUTHOR_COMMIT}"
            )
        artifacts = load_track_a_author_artifacts(
            author_repo,
            TRACK_A_AUTHOR_COMMIT,
        )
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
    rfecv_exposure_path = run_dir / "track_a_rfecv_outer_test_exposure.csv"
    _atomic_dataframe_write(
        track_a_rfecv_outer_test_exposure(artifacts),
        rfecv_exposure_path,
    )

    configurations: dict[tuple[str, str], tuple[ModelConfig, str]] = {}
    for selected_mode in requested_modes:
        for selected_model in requested_models:
            selected_model_config = _track_a_model_config(selected_model, published)
            mode_configuration = _track_a_mode_configuration(
                mode=selected_mode,
                model_name=selected_model,
                model_config=selected_model_config,
                published=published,
                patience=patience,
                reconstruction_seed=reconstruction_seed,
                feature_cache_key=feature_cache.cache_key,
                code_revision=code_revision,
                execution_backend=execution_backend,
            )
            configurations[(selected_mode, selected_model)] = (
                selected_model_config,
                _canonical_sha256(mode_configuration),
            )
    expected_folds = tuple(range(len(artifacts.test_folds)))
    run_contract = {
        "format_version": 1,
        "run_id": run_id,
        "track": "A",
        "modes": list(requested_modes),
        "model_names": list(requested_models),
        "expected_folds": list(expected_folds),
        "author_commit": artifacts.author_commit,
        "feature_cache_key": feature_cache.cache_key,
        "code_revision": code_revision,
        "execution_backend": execution_backend,
        "configuration_sha256": {
            f"{mode}/{model}": configurations[(mode, model)][1]
            for mode in requested_modes
            for model in requested_models
        },
    }
    contract_path = run_dir / "track_a_run_contract.json"
    if contract_path.is_file():
        try:
            existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Track A run contract cannot be authenticated") from exc
        if existing_contract != run_contract:
            raise ValueError("Track A run contract mismatch")
    else:
        _atomic_json_write(run_contract, contract_path)

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict[str, object]] = []
    total_units = len(requested_modes) * len(requested_models) * len(expected_folds)
    completed_units = 0

    for mode in requested_modes:
        for model_name in requested_models:
            model_config, configuration_sha = configurations[(mode, model_name)]
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
                state: dict[str, object] = {}
                if resume and _manifest_path(state_dir, "latest_recovery").is_file():
                    recovered = _resolve_checkpoint_manifest(
                        state_dir, role="latest_recovery", map_location="cpu"
                    )
                    state = recovered.payload
                    if state.get("execution_backend") != execution_backend:
                        raise ValueError(
                            "execution_backend mismatch: checkpoint="
                            f"{state.get('execution_backend')!r}, "
                            f"requested={execution_backend!r}"
                        )
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
                if phase not in {"between_folds", "training", "trained"}:
                    raise ValueError("author Track A resume phase is invalid")
                if phase == "between_folds" and resume_epoch != 0:
                    raise ValueError("author between-fold state must have completed_epoch=0")
                if phase != "between_folds":
                    stored_batch_value = state.get("fold_batch")
                    if not isinstance(stored_batch_value, list) or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in stored_batch_value
                    ):
                        raise ValueError("author in-fold checkpoint batch is invalid")
                    stored_batch = tuple(stored_batch_value)
                    if tuple(sorted(set(stored_batch))) != stored_batch:
                        raise ValueError("author in-fold checkpoint batch is invalid")
                    if next_fold not in stored_batch or next_fold not in folds:
                        raise ValueError(
                            "author in-fold resume batch must contain the interrupted fold"
                        )
                completed_prior, preceding_receipt_binding = (
                    _validate_track_a_author_receipt_chain(
                        run_dir=run_dir,
                        run_id=run_id,
                        next_fold=next_fold,
                        model_name=model_name,
                        artifacts=artifacts,
                        feature_cache=feature_cache,
                        configuration_sha256=configuration_sha,
                        code_revision=code_revision,
                        reconstruction_seed=reconstruction_seed,
                        execution_backend=execution_backend,
                        state=state,
                    )
                )
                if resume and phase == "trained":
                    completed_fold = next_fold
                    completed_fold_dir = (
                        run_dir
                        / "track_a"
                        / mode
                        / model_name
                        / f"fold_{completed_fold:02d}"
                    )
                    completion_files = tuple(
                        completed_fold_dir / name
                        for name in ("receipt.json", "predictions.csv", "metrics.json")
                    )
                    if any(path.exists() for path in completion_files):
                        completed_context = _track_a_author_fold_context(
                            run_dir=run_dir,
                            run_id=run_id,
                            fold=completed_fold,
                            model_name=model_name,
                            artifacts=artifacts,
                            feature_cache=feature_cache,
                            configuration_sha256=configuration_sha,
                            code_revision=code_revision,
                            reconstruction_seed=reconstruction_seed,
                            execution_backend=execution_backend,
                        )
                        completed_output = _load_completed_track_a_fold(
                            completed_fold_dir,
                            expected=completed_context,
                        )
                        if completed_output is None:
                            raise ValueError(
                                "trained author state has no authenticated completed fold"
                            )
                        completed_receipt_binding = _track_a_receipt_state_binding(
                            run_dir=run_dir,
                            fold_dir=completed_fold_dir,
                            context=completed_context,
                        )
                        _publish_track_a_author_between_fold_state(
                            state_dir=state_dir,
                            configuration_sha256=configuration_sha,
                            model=model,
                            optimizer=optimizer,
                            completed_fold=completed_fold,
                            fold_batch=folds,
                            epochs_per_fold=int(published["epochs"]),
                            receipt_binding=completed_receipt_binding,
                            execution_backend=execution_backend,
                        )
                        completed_prior[completed_fold] = completed_output
                        preceding_receipt_binding = completed_receipt_binding
                        next_fold = completed_fold + 1
                        resume_epoch = 0
                        phase = "between_folds"

                for fold in folds:
                    if fold < next_fold:
                        predictions, metric = completed_prior[fold]
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
                                "fold_batch": list(folds),
                                "rng_state": _rng_state(),
                                "fold_seed": fold_seed,
                                "balancing_audit": balanced.audit,
                                "execution_backend": execution_backend,
                                **preceding_receipt_binding,
                            },
                            state_dir,
                            role="latest_recovery",
                            epoch=fold * int(published["epochs"]) + epoch,
                        )
                        if interrupt_after == (mode, model_name, fold, epoch):
                            raise PlannedInterruption(
                                f"planned Track A interruption at {mode}/{model_name}/{fold}/{epoch}"
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
                    split_sha = _track_a_author_split_sha256(artifacts, fold)
                    fold_dir = (
                        run_dir
                        / "track_a"
                        / mode
                        / model_name
                        / f"fold_{fold:02d}"
                    )
                    fold_inference = _publish_checkpoint_generation(
                        {
                            "format_version": 1,
                            "checkpoint_role": "best_inference",
                            "track_a_configuration_sha256": configuration_sha,
                            "model_state": model.state_dict(),
                            "fold": fold,
                            "split_sha256": split_sha,
                            "execution_backend": execution_backend,
                        },
                        fold_dir / "checkpoints",
                        role="best_inference",
                        epoch=int(published["epochs"]),
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
                        predicted_label=np.asarray(
                            threshold_audit["covid_prediction"], dtype=np.int64
                        ),
                        threshold=public_threshold,
                        threshold_comparator="gt",
                        threshold_source="test_balanced_accuracy_author_audit",
                        configuration_sha256=configuration_sha,
                        split_sha256=split_sha,
                        feature_sha256=feature_cache.selected_features_sha256,
                        feature_indices_sha256=feature_cache.selected_indices_sha256,
                        feature_cache_key=feature_cache.cache_key,
                        checkpoint_sha=str(fold_inference.descriptor["sha256"]),
                        author_commit=artifacts.author_commit,
                        code_revision=code_revision,
                        execution_backend=execution_backend,
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
                        checkpoint_sha=str(fold_inference.descriptor["sha256"]),
                        author_commit=artifacts.author_commit,
                        code_revision=code_revision,
                        feature_cache=feature_cache,
                        fold_hash=split_sha,
                        balancing_audit=balanced.audit,
                    )
                    completed_context = TrackAFoldContext(
                        run_id=run_id,
                        mode=mode,
                        fold=fold,
                        model_name=model_name,
                        author_commit=artifacts.author_commit,
                        feature_sha256=feature_cache.selected_features_sha256,
                        feature_indices_sha256=feature_cache.selected_indices_sha256,
                        feature_cache_key=feature_cache.cache_key,
                        configuration_sha256=configuration_sha,
                        split_sha256=split_sha,
                        code_revision=code_revision,
                        execution_backend=execution_backend,
                        threshold_source="test_balanced_accuracy_author_audit",
                        threshold_comparator="gt",
                        reconstruction_seed=reconstruction_seed,
                        checkpoint_path=fold_inference.path,
                        checkpoint_sha256=str(fold_inference.descriptor["sha256"]),
                    )
                    receipt_path = _write_track_a_fold_outputs(
                        fold_dir,
                        predictions,
                        metric,
                        context=completed_context,
                    )
                    if receipt_path != fold_dir / "receipt.json":
                        raise RuntimeError("Track A receipt path is not deterministic")
                    if interrupt_after_receipt == (mode, model_name, fold):
                        raise PlannedInterruption(
                            f"planned Track A interruption after fold receipt at "
                            f"{mode}/{model_name}/{fold}"
                        )
                    completed_receipt_binding = _track_a_receipt_state_binding(
                        run_dir=run_dir,
                        fold_dir=fold_dir,
                        context=completed_context,
                    )
                    _publish_track_a_author_between_fold_state(
                        state_dir=state_dir,
                        configuration_sha256=configuration_sha,
                        model=model,
                        optimizer=optimizer,
                        completed_fold=fold,
                        fold_batch=folds,
                        epochs_per_fold=int(published["epochs"]),
                        receipt_binding=completed_receipt_binding,
                        execution_backend=execution_backend,
                    )
                    next_fold = fold + 1
                    resume_epoch = 0
                    phase = "between_folds"
                    preceding_receipt_binding = completed_receipt_binding
                    completed_prior[fold] = (predictions, metric)
                    all_predictions.append(predictions)
                    all_metrics.append(metric)
                    completed_units += 1
            elif mode == "fresh_fold_author_protocol":
                for fold in folds:
                    fold_dir = run_dir / "track_a" / mode / model_name / f"fold_{fold:02d}"
                    split_sha = _track_a_author_split_sha256(artifacts, fold)
                    fold_seed = reconstruction_seed + fold
                    completed = None
                    completion_files = tuple(
                        fold_dir / name
                        for name in ("receipt.json", "predictions.csv", "metrics.json")
                    )
                    if resume and any(path.exists() for path in completion_files):
                        try:
                            inference = _resolve_checkpoint_manifest(
                                fold_dir / "checkpoints",
                                role="best_inference",
                                map_location="cpu",
                            )
                        except (FileNotFoundError, RuntimeError, ValueError) as exc:
                            raise ValueError(
                                "authenticated fold checkpoint is missing or invalid"
                            ) from exc
                        completed_context = TrackAFoldContext(
                            run_id=run_id,
                            mode=mode,
                            fold=fold,
                            model_name=model_name,
                            author_commit=artifacts.author_commit,
                            feature_sha256=feature_cache.selected_features_sha256,
                            feature_indices_sha256=feature_cache.selected_indices_sha256,
                            feature_cache_key=feature_cache.cache_key,
                            configuration_sha256=configuration_sha,
                            split_sha256=split_sha,
                            code_revision=code_revision,
                            execution_backend=execution_backend,
                            threshold_source="test_balanced_accuracy_author_audit",
                            threshold_comparator="gt",
                            reconstruction_seed=fold_seed,
                            checkpoint_path=inference.path,
                            checkpoint_sha256=str(inference.descriptor["sha256"]),
                        )
                        completed = _load_completed_track_a_fold(
                            fold_dir,
                            expected=completed_context,
                        )
                    if completed is not None:
                        predictions, metric = completed
                        all_predictions.append(predictions)
                        all_metrics.append(metric)
                        completed_units += 1
                        continue
                    requested_interrupt = (
                        interrupt_after[3]
                        if interrupt_after is not None
                        and interrupt_after[:3] == (mode, model_name, fold)
                        else None
                    )
                    train_indices = artifacts.train_folds[fold]
                    fitted = _track_a_fresh_fold_author_fit(
                        feature_cache.selected_features[train_indices],
                        artifacts.author_labels[train_indices],
                        model_config=model_config,
                        learning_rate=float(published["learning_rate"]),
                        batch_size=int(published["batch_size"]),
                        max_epochs=int(published["epochs"]),
                        reconstruction_seed=fold_seed,
                        checkpoint_dir=fold_dir / "checkpoints",
                        feature_sha256=feature_cache.selected_features_sha256,
                        split_sha256=split_sha,
                        code_revision=code_revision,
                        device=torch_device,
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
                    threshold_audit = author_threshold_sweep(
                        artifacts.author_labels[test_indices], probability_n
                    )
                    public_threshold = author_threshold_to_covid_threshold(
                        float(threshold_audit["author_threshold"])
                    )
                    predictions = _track_a_prediction_frame(
                        run_id=run_id,
                        mode=mode,
                        fold=fold,
                        seed=fold_seed,
                        model_name=model_name,
                        test_indices=test_indices,
                        covid_labels=artifacts.covid_labels[test_indices],
                        covid_probability=1.0 - probability_n,
                        predicted_label=np.asarray(
                            threshold_audit["covid_prediction"], dtype=np.int64
                        ),
                        threshold=public_threshold,
                        threshold_comparator="gt",
                        threshold_source="test_balanced_accuracy_author_audit",
                        configuration_sha256=configuration_sha,
                        split_sha256=split_sha,
                        feature_sha256=feature_cache.selected_features_sha256,
                        feature_indices_sha256=feature_cache.selected_indices_sha256,
                        feature_cache_key=feature_cache.cache_key,
                        checkpoint_sha=fitted.checkpoint_sha256,
                        author_commit=artifacts.author_commit,
                        code_revision=code_revision,
                        execution_backend=execution_backend,
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
                        reconstruction_seed=fold_seed,
                        model_reinitialized=True,
                        checkpoint_sha=fitted.checkpoint_sha256,
                        author_commit=artifacts.author_commit,
                        code_revision=code_revision,
                        feature_cache=feature_cache,
                        fold_hash=split_sha,
                        balancing_audit=fitted.balancing_audit,
                    )
                    completed_context = TrackAFoldContext(
                        run_id=run_id,
                        mode=mode,
                        fold=fold,
                        model_name=model_name,
                        author_commit=artifacts.author_commit,
                        feature_sha256=feature_cache.selected_features_sha256,
                        feature_indices_sha256=feature_cache.selected_indices_sha256,
                        feature_cache_key=feature_cache.cache_key,
                        configuration_sha256=configuration_sha,
                        split_sha256=split_sha,
                        code_revision=code_revision,
                        execution_backend=execution_backend,
                        threshold_source="test_balanced_accuracy_author_audit",
                        threshold_comparator="gt",
                        reconstruction_seed=fold_seed,
                        checkpoint_path=fitted.checkpoint_path,
                        checkpoint_sha256=fitted.checkpoint_sha256,
                    )
                    _write_track_a_fold_outputs(
                        fold_dir,
                        predictions,
                        metric,
                        context=completed_context,
                    )
                    if interrupt_after_receipt == (mode, model_name, fold):
                        raise PlannedInterruption(
                            "planned Track A interruption after fold receipt at "
                            f"{mode}/{model_name}/{fold}"
                        )
                    all_predictions.append(predictions)
                    all_metrics.append(metric)
                    completed_units += 1
            else:
                for fold in folds:
                    fold_dir = run_dir / "track_a" / mode / model_name / f"fold_{fold:02d}"
                    inner_train, inner_validation, split_sha = (
                        _track_a_corrected_split(
                            artifacts,
                            fold,
                            reconstruction_seed,
                        )
                    )
                    completed = None
                    completion_files = tuple(
                        fold_dir / name
                        for name in ("receipt.json", "predictions.csv", "metrics.json")
                    )
                    if resume and any(path.exists() for path in completion_files):
                        try:
                            inference = _resolve_checkpoint_manifest(
                                fold_dir / "checkpoints",
                                role="best_inference",
                                map_location="cpu",
                            )
                        except (FileNotFoundError, RuntimeError, ValueError) as exc:
                            raise ValueError(
                                "authenticated fold checkpoint is missing or invalid"
                            ) from exc
                        completed_context = TrackAFoldContext(
                            run_id=run_id,
                            mode=mode,
                            fold=fold,
                            model_name=model_name,
                            author_commit=artifacts.author_commit,
                            feature_sha256=feature_cache.selected_features_sha256,
                            feature_indices_sha256=feature_cache.selected_indices_sha256,
                            feature_cache_key=feature_cache.cache_key,
                            configuration_sha256=configuration_sha,
                            split_sha256=split_sha,
                            code_revision=code_revision,
                            execution_backend=execution_backend,
                            threshold_source="inner_validation_balanced_accuracy",
                            threshold_comparator="ge",
                            reconstruction_seed=reconstruction_seed + fold,
                            checkpoint_path=inference.path,
                            checkpoint_sha256=str(inference.descriptor["sha256"]),
                        )
                        completed = _load_completed_track_a_fold(
                            fold_dir,
                            expected=completed_context,
                        )
                    if completed is not None:
                        predictions, metric = completed
                        all_predictions.append(predictions)
                        all_metrics.append(metric)
                        completed_units += 1
                        continue
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
                        predicted_label=(
                            (1.0 - probability_n) >= fitted.threshold
                        ).astype(np.int64),
                        threshold=fitted.threshold,
                        threshold_comparator="ge",
                        threshold_source="inner_validation_balanced_accuracy",
                        configuration_sha256=configuration_sha,
                        split_sha256=split_sha,
                        feature_sha256=feature_cache.selected_features_sha256,
                        feature_indices_sha256=feature_cache.selected_indices_sha256,
                        feature_cache_key=feature_cache.cache_key,
                        checkpoint_sha=fitted.checkpoint_sha256,
                        author_commit=artifacts.author_commit,
                        code_revision=code_revision,
                        execution_backend=execution_backend,
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
                        code_revision=code_revision,
                        feature_cache=feature_cache,
                        fold_hash=split_sha,
                        balancing_audit=fitted.balancing_audit,
                        inner_split_seed=reconstruction_seed + fold,
                        inner_validation_fraction=0.125,
                    )
                    completed_context = TrackAFoldContext(
                        run_id=run_id,
                        mode=mode,
                        fold=fold,
                        model_name=model_name,
                        author_commit=artifacts.author_commit,
                        feature_sha256=feature_cache.selected_features_sha256,
                        feature_indices_sha256=feature_cache.selected_indices_sha256,
                        feature_cache_key=feature_cache.cache_key,
                        configuration_sha256=configuration_sha,
                        split_sha256=split_sha,
                        code_revision=code_revision,
                        execution_backend=execution_backend,
                        threshold_source="inner_validation_balanced_accuracy",
                        threshold_comparator="ge",
                        reconstruction_seed=reconstruction_seed + fold,
                        checkpoint_path=fitted.checkpoint_path,
                        checkpoint_sha256=fitted.checkpoint_sha256,
                    )
                    _write_track_a_fold_outputs(
                        fold_dir,
                        predictions,
                        metric,
                        context=completed_context,
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

    del all_predictions, all_metrics
    authenticated_predictions: list[pd.DataFrame] = []
    authenticated_metrics: list[dict[str, object]] = []
    authenticated_receipts: list[dict[str, object]] = []
    for mode in requested_modes:
        for model_name in requested_models:
            _, configuration_sha = configurations[(mode, model_name)]
            for fold in expected_folds:
                fold_dir = (
                    run_dir / "track_a" / mode / model_name / f"fold_{fold:02d}"
                )
                completion_paths = tuple(
                    fold_dir / name
                    for name in ("receipt.json", "predictions.csv", "metrics.json")
                )
                if not any(path.exists() for path in completion_paths):
                    continue
                try:
                    inference = _resolve_track_a_fold_inference(fold_dir)
                    if mode in {
                        "author_behaviour_audit",
                        "fresh_fold_author_protocol",
                    }:
                        split_sha = _track_a_author_split_sha256(artifacts, fold)
                        fold_seed = (
                            reconstruction_seed
                            if mode == "author_behaviour_audit"
                            else reconstruction_seed + fold
                        )
                        comparator = "gt"
                        threshold_source = "test_balanced_accuracy_author_audit"
                    else:
                        _, _, split_sha = _track_a_corrected_split(
                            artifacts,
                            fold,
                            reconstruction_seed,
                        )
                        fold_seed = reconstruction_seed + fold
                        comparator = "ge"
                        threshold_source = "inner_validation_balanced_accuracy"
                    context = TrackAFoldContext(
                        run_id=run_id,
                        mode=mode,
                        fold=fold,
                        model_name=model_name,
                        author_commit=artifacts.author_commit,
                        feature_sha256=feature_cache.selected_features_sha256,
                        feature_indices_sha256=feature_cache.selected_indices_sha256,
                        feature_cache_key=feature_cache.cache_key,
                        configuration_sha256=configuration_sha,
                        split_sha256=split_sha,
                        code_revision=code_revision,
                        execution_backend=execution_backend,
                        threshold_source=threshold_source,
                        threshold_comparator=comparator,
                        reconstruction_seed=fold_seed,
                        checkpoint_path=inference.path,
                        checkpoint_sha256=str(inference.descriptor["sha256"]),
                    )
                    completed = _load_completed_track_a_fold(
                        fold_dir,
                        expected=context,
                    )
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    raise ValueError(
                        "authenticated fold aggregation encountered an invalid prior output"
                    ) from exc
                if completed is None:
                    raise ValueError(
                        "authenticated fold aggregation encountered an incomplete output"
                    )
                predictions, metric = completed
                authenticated_predictions.append(predictions)
                authenticated_metrics.append(metric)
                receipt_path = fold_dir / "receipt.json"
                authenticated_receipts.append(
                    {
                        "path": _track_a_relative_artifact_path(
                            receipt_path,
                            run_dir,
                        ),
                        "sha256": checkpoint_sha256(receipt_path),
                    }
                )

    prediction_frame = (
        pd.concat(authenticated_predictions, ignore_index=True)
        if authenticated_predictions
        else pd.DataFrame()
    )
    metric_frame = pd.DataFrame(authenticated_metrics)
    aggregate_artifacts: dict[str, dict[str, str]] = {}
    if not prediction_frame.empty:
        prediction_path = run_dir / "track_a_predictions.csv"
        write_predictions(prediction_frame, prediction_path)
        aggregate_artifacts["predictions"] = {
            "path": _track_a_relative_artifact_path(prediction_path, run_dir),
            "sha256": checkpoint_sha256(prediction_path),
        }
    if not metric_frame.empty:
        serializable_metrics = metric_frame.copy()
        if "balancing_audit" in serializable_metrics:
            serializable_metrics["balancing_audit"] = serializable_metrics[
                "balancing_audit"
            ].map(lambda value: json.dumps(value, sort_keys=True))
        metric_path = run_dir / "track_a_metrics.csv"
        _atomic_dataframe_write(serializable_metrics, metric_path)
        aggregate_artifacts["metrics"] = {
            "path": _track_a_relative_artifact_path(metric_path, run_dir),
            "sha256": checkpoint_sha256(metric_path),
        }
    aggregate_artifacts["rfecv_outer_test_exposure"] = {
        "path": _track_a_relative_artifact_path(rfecv_exposure_path, run_dir),
        "sha256": checkpoint_sha256(rfecv_exposure_path),
    }
    completed_units = len(authenticated_metrics)
    final_status = "complete" if completed_units == total_units else "partial"
    required_design = {
        "modes": [
            "author_behaviour_audit",
            "fresh_fold_author_protocol",
            "corrected_reference",
        ],
        "model_names": ["dndf"],
        "folds": list(expected_folds),
    }
    authenticated_design = {
        (str(metric["mode"]), str(metric["model_name"]), int(metric["fold"]))
        for metric in authenticated_metrics
    }
    required_units = {
        (mode, "dndf", fold)
        for mode in required_design["modes"]
        for fold in expected_folds
    }
    final = {
        "status": final_status,
        "run_id": run_id,
        "track": "A",
        "completed_units": completed_units,
        "total_units": total_units,
        "fold_batch": list(folds),
        "expected_folds": list(expected_folds),
        "modes": list(requested_modes),
        "model_names": list(requested_models),
        "required_design": required_design,
        "design_contract_complete": required_units.issubset(authenticated_design),
        "author_commit": artifacts.author_commit,
        "feature_cache_key": feature_cache.cache_key,
        "author_training_order": "no_shuffle_repeated_dataset",
        "author_randomness_unseeded": True,
        "reconstruction_seed": reconstruction_seed,
        "execution_backend": execution_backend,
        "artifacts": aggregate_artifacts,
        "authenticated_receipts": authenticated_receipts,
        "metrics": metric_frame.to_dict(orient="records"),
        "predictions": prediction_frame.to_dict(orient="records"),
    }
    persisted_final = {
        key: value for key, value in final.items() if key not in {"metrics", "predictions"}
    }
    _atomic_json_write(persisted_final, run_dir / "progress.json")
    _atomic_json_write(persisted_final, run_dir / "track_a_manifest.json")
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
        comparators = predictions["threshold_comparator"].astype(str)
        if not comparators.isin(["gt", "ge"]).all():
            raise ValueError("Track A threshold_comparator must be 'gt' or 'ge'")
        if not predictions["predicted_label"].isin([0, 1]).all():
            raise ValueError("Track A predicted_label must contain binary labels")
        expected_prediction = np.where(
            comparators.to_numpy() == "gt",
            predictions["probability"].to_numpy(dtype=np.float64)
            > predictions["threshold"].to_numpy(dtype=np.float64),
            predictions["probability"].to_numpy(dtype=np.float64)
            >= predictions["threshold"].to_numpy(dtype=np.float64),
        ).astype(np.int64)
        if not np.array_equal(
            predictions["predicted_label"].to_numpy(dtype=np.int64),
            expected_prediction,
        ):
            raise ValueError(
                "Track A predicted_label does not match threshold_comparator semantics"
            )
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
    "CandidateSelectionResult",
    "EXPECTED_TRACK_A_SELECTED_INDICES",
    "FitResult",
    "FittedPreprocessor",
    "PREDICTION_COLUMNS",
    "PlannedInterruption",
    "TrackAFeatureCache",
    "TrackBData",
    "TrackBProtocol",
    "TrainConfig",
    "aggregate_participant_probabilities",
    "author_threshold_to_covid_threshold",
    "author_threshold_sweep",
    "balance_training_rows",
    "build_track_b_protocols",
    "checkpoint_sha256",
    "deterministic_batch_indices",
    "fixed_order_batch_indices",
    "fit_model",
    "fit_preprocessor",
    "load_track_a_author_artifacts",
    "load_track_b_data",
    "prepare_track_a_rfecv_cache",
    "track_a_rfecv_outer_test_exposure",
    "run_track_a",
    "run_track_b",
    "select_modality_configuration",
    "transform_features",
    "validate_prediction_frame",
    "validate_frozen_track_b_lineage",
    "write_predictions",
]
