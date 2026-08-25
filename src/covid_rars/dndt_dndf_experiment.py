from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE, SVMSMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from covid_rars.dndt_dndf_models import ModelConfig, NeuralDecisionClassifier
from covid_rars.metrics import best_threshold_by_balanced_accuracy


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


class PlannedInterruption(RuntimeError):
    """Raised by tests or controllers only after an epoch checkpoint is durable."""


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
    "BalanceResult",
    "FitResult",
    "FittedPreprocessor",
    "PREDICTION_COLUMNS",
    "PlannedInterruption",
    "TrainConfig",
    "aggregate_participant_probabilities",
    "balance_training_rows",
    "checkpoint_sha256",
    "deterministic_batch_indices",
    "fit_model",
    "fit_preprocessor",
    "transform_features",
    "validate_prediction_frame",
    "write_predictions",
]
