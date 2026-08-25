#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable


GIB = 1024**3
DECIMAL_GB = 1_000_000_000
MINIMUM_CUDA_MEMORY_BYTES = 6_000_000_000
MINIMUM_CUDA_MEMORY_GB = MINIMUM_CUDA_MEMORY_BYTES / DECIMAL_GB
DEFAULT_MINIMUM_FREE_GIB = 10.0
EXPECTED_SAMPLE_COUNT = 1319
EXPECTED_FEATURE_SHAPE = (1319, 193)
EXPECTED_CLASS_COUNTS = {"C": 185, "N": 1134}
PROJECT_IDENTITY_COLUMNS = (
    "recording_id",
    "participant_id",
    "dataset",
    "modality",
    "submodality",
    "label_binary",
    "split",
)
METADATA_REQUIRED_COLUMNS = (
    "participant_id",
    "label_binary",
    "split",
    "recording_date",
)


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CUDA mode requires CUDA-enabled PyTorch installed from the official cu130 index"
        ) from exc
    return torch


def _header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV has no header: {path}") from exc
    if not columns or any(not column.strip() for column in columns):
        raise ValueError(f"CSV header contains an empty column name: {path}")
    return [column.strip() for column in columns]


def _duplicate_names(columns: list[str]) -> list[str]:
    counts = Counter(columns)
    return sorted(column for column, count in counts.items() if count > 1)


def _validate_numeric_feature_content(
    path: Path, columns: list[str], feature_columns: list[str]
) -> None:
    feature_set = set(feature_columns)
    feature_indices = {
        column: index for index, column in enumerate(columns) if column in feature_set
    }
    has_finite_value = {column: False for column in feature_columns}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            streamed_header = [column.strip() for column in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"CSV has no header: {path}") from exc
        if streamed_header != columns:
            raise ValueError(f"CSV header changed while auditing feature content: {path}")

        # Stream one row at a time so memory does not scale with the feature table.
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(columns):
                raise ValueError(
                    f"CSV row {line_number} in {path} has {len(row)} fields; "
                    f"expected {len(columns)}"
                )
            for column, index in feature_indices.items():
                raw = row[index].strip()
                if not raw:
                    continue
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"feature column {column!r} contains a nonnumeric value "
                        f"at CSV row {line_number} in {path}: {raw[:80]!r}"
                    ) from exc
                if math.isnan(value):
                    continue
                if math.isinf(value):
                    raise ValueError(
                        f"feature column {column!r} contains an infinite value "
                        f"at CSV row {line_number} in {path}: {raw!r}"
                    )
                has_finite_value[column] = True

    without_finite_values = [
        column for column, has_finite in has_finite_value.items() if not has_finite
    ]
    if without_finite_values:
        displayed = ", ".join(repr(column) for column in without_finite_values[:10])
        if len(without_finite_values) > 10:
            displayed += f", ... ({len(without_finite_values)} total)"
        raise ValueError(
            f"feature columns {displayed} have no finite numeric values in {path}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_writable_directory(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=run_root,
            prefix=".dndt-dndf-preflight-",
            delete=False,
        ) as handle:
            handle.write(b"preflight\n")
            probe_path = Path(handle.name)
    except OSError as exc:
        raise RuntimeError(f"run root is not writable: {run_root}: {exc}") from exc
    finally:
        if probe_path is not None:
            try:
                probe_path.unlink()
            except OSError:
                pass


def validate_runtime(
    device: str, run_root: Path, minimum_free_gib: float
) -> dict[str, Any]:
    normalized_device = device.strip().lower()
    if normalized_device not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'")
    if not math.isfinite(minimum_free_gib) or minimum_free_gib < 0:
        raise ValueError("minimum_free_gib must be a finite non-negative value")

    run_root = Path(run_root)
    _verify_writable_directory(run_root)
    free_disk_gib = shutil.disk_usage(run_root).free / GIB
    if free_disk_gib < minimum_free_gib:
        raise RuntimeError(
            f"run root has {free_disk_gib:.2f} GiB free; "
            f"at least {minimum_free_gib:.2f} GiB is required"
        )

    audit: dict[str, Any] = {
        "device": normalized_device,
        "run_root": str(run_root),
        "run_root_writable": True,
        "minimum_free_gib": float(minimum_free_gib),
        "free_disk_gib": free_disk_gib,
    }
    if normalized_device == "cpu":
        return audit

    torch = _load_torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA mode requires CUDA-enabled PyTorch installed from the official cu130 index"
        )
    device_count = int(torch.cuda.device_count())
    if device_count < 1:
        raise RuntimeError("CUDA is available but no CUDA devices were reported")
    properties = torch.cuda.get_device_properties(0)
    total_memory_bytes = int(properties.total_memory)
    total_memory_gb = total_memory_bytes / DECIMAL_GB
    total_memory_gib = total_memory_bytes / GIB
    if total_memory_bytes < MINIMUM_CUDA_MEMORY_BYTES:
        raise RuntimeError(
            f"CUDA device 0 has {total_memory_bytes} bytes of total memory "
            f"({total_memory_gb:.9f} GB decimal; {total_memory_gib:.9f} GiB); "
            f"at least {MINIMUM_CUDA_MEMORY_BYTES} bytes "
            f"({MINIMUM_CUDA_MEMORY_GB:.9f} GB decimal) are required"
        )

    tiny = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda")
    probe = (tiny * tiny).sum()
    if not bool(torch.isfinite(probe).item()):
        raise RuntimeError("tiny CUDA tensor computation produced a non-finite result")

    audit.update(
        {
            "cuda_device_count": device_count,
            "cuda_device_name": str(properties.name),
            "cuda_total_memory_bytes": total_memory_bytes,
            "cuda_total_memory_gb": total_memory_gb,
            "cuda_total_memory_gib": total_memory_gib,
            "cuda_probe_value": float(probe.item()),
        }
    )
    return audit


def audit_project_feature_table(
    path: Path, expected_feature_count: int = 800
) -> dict[str, Any]:
    path = Path(path)
    columns = _header(path)
    duplicates = _duplicate_names(columns)
    if duplicates:
        raise ValueError(f"duplicate column names in {path}: {duplicates}")

    missing = [column for column in PROJECT_IDENTITY_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"missing required identity columns in {path}: {missing}")
    feature_columns = [
        column for column in columns if column not in PROJECT_IDENTITY_COLUMNS
    ]
    if len(feature_columns) != expected_feature_count:
        raise ValueError(
            f"expected exactly {expected_feature_count} feature columns in {path}, "
            f"found {len(feature_columns)}"
        )
    _validate_numeric_feature_content(path, columns, feature_columns)
    return {
        "path": str(path),
        "column_count": len(columns),
        "feature_count": len(feature_columns),
        "required_columns_present": True,
        "feature_columns": feature_columns,
    }


def audit_external_alignment(
    project_path: Path, external_path: Path
) -> dict[str, Any]:
    project_path = Path(project_path)
    external_path = Path(external_path)
    project = audit_project_feature_table(project_path)
    external_columns = _header(external_path)
    duplicates = _duplicate_names(external_columns)
    if duplicates:
        raise ValueError(f"duplicate column names in {external_path}: {duplicates}")
    external_features = [
        column for column in external_columns if column not in PROJECT_IDENTITY_COLUMNS
    ]
    if external_features != project["feature_columns"]:
        raise ValueError(
            "project and external ordered feature columns do not match: "
            f"{project_path} != {external_path}"
        )
    _validate_numeric_feature_content(
        external_path, external_columns, external_features
    )
    return {
        "project_path": str(project_path),
        "external_path": str(external_path),
        "feature_count": len(external_features),
        "ordered_features_match": True,
    }


def audit_metadata_table(path: Path) -> dict[str, Any]:
    path = Path(path)
    columns = _header(path)
    duplicates = _duplicate_names(columns)
    if duplicates:
        raise ValueError(f"duplicate column names in {path}: {duplicates}")
    missing = [column for column in METADATA_REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"missing required metadata columns in {path}: {missing}")
    return {
        "path": str(path),
        "column_count": len(columns),
        "required_columns_present": True,
    }


def _git(author_repo: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(author_repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed for {author_repo}: {detail}")
    return process.stdout.strip()


def _parse_fold_indices(path: Path, expected_header: str) -> list[int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"fold CSV has no header: {path}") from exc
        if not header or header[0].strip() != expected_header:
            raise ValueError(
                f"fold CSV {path} must use {expected_header!r} as its first column"
            )
        indices: list[int] = []
        for line_number, row in enumerate(reader, start=2):
            if not row or not row[0].strip():
                continue
            raw = row[0].strip()
            try:
                value = Decimal(raw)
            except InvalidOperation as exc:
                raise ValueError(
                    f"non-numeric index at {path}:{line_number}: {raw!r}"
                ) from exc
            if not value.is_finite() or value != value.to_integral_value():
                raise ValueError(
                    f"non-integer index at {path}:{line_number}: {raw!r}"
                )
            indices.append(int(value))
    return indices


def audit_author_artifacts(
    author_repo: Path, expected_commit: str
) -> dict[str, Any]:
    author_repo = Path(author_repo)
    if not author_repo.is_dir():
        raise FileNotFoundError(f"author repository does not exist: {author_repo}")
    head = _git(author_repo, "rev-parse", "HEAD")
    if head != expected_commit:
        raise RuntimeError(
            f"author repository HEAD mismatch: expected {expected_commit}, found {head}"
        )
    tracked_status = _git(author_repo, "status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise RuntimeError("author repository has tracked worktree changes")

    import numpy as np

    feature_relative = Path(
        "Extracted Features/Coswara/cough_X_features_np.npy"
    )
    label_relative = Path(
        "Extracted Features/Coswara/cough_y_features_np.npy"
    )
    feature_path = author_repo / feature_relative
    label_path = author_repo / label_relative
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    if features.shape != EXPECTED_FEATURE_SHAPE:
        raise ValueError(
            f"author feature shape mismatch: expected {EXPECTED_FEATURE_SHAPE}, "
            f"found {features.shape}"
        )
    if labels.shape != (EXPECTED_SAMPLE_COUNT,):
        raise ValueError(
            f"author label shape mismatch: expected {(EXPECTED_SAMPLE_COUNT,)}, "
            f"found {labels.shape}"
        )
    class_counts = {
        label: int(np.count_nonzero(labels == label)) for label in EXPECTED_CLASS_COUNTS
    }
    unexpected_labels = sorted(
        str(label) for label in np.unique(labels) if str(label) not in EXPECTED_CLASS_COUNTS
    )
    if class_counts != EXPECTED_CLASS_COUNTS or unexpected_labels:
        raise ValueError(
            "author label counts mismatch: "
            f"expected {EXPECTED_CLASS_COUNTS}, found {class_counts}, "
            f"unexpected labels={unexpected_labels}"
        )

    all_indices = set(range(EXPECTED_SAMPLE_COUNT))
    test_sets: list[set[int]] = []
    fold_hashes: dict[str, str] = {}
    for fold in range(10):
        train_relative = Path(
            f"Train-Test Split/coswaradataset/train/{fold}.csv"
        )
        test_relative = Path(
            f"Train-Test Split/coswaradataset/test/{fold}.csv"
        )
        train_path = author_repo / train_relative
        test_path = author_repo / test_relative
        train_indices = _parse_fold_indices(train_path, "train_index")
        test_indices = _parse_fold_indices(test_path, "test_index")
        train_set = set(train_indices)
        test_set = set(test_indices)
        if len(train_set) != len(train_indices) or len(test_set) != len(test_indices):
            raise ValueError(f"fold {fold} contains duplicate indices")
        if not train_set <= all_indices or not test_set <= all_indices:
            raise ValueError(f"fold {fold} contains indices outside 0..1318")
        if train_set & test_set:
            raise ValueError(f"fold {fold} train and test indices overlap")
        if train_set | test_set != all_indices:
            raise ValueError(f"fold {fold} train and test do not cover all 1319 samples")
        test_sets.append(test_set)
        fold_hashes[train_relative.as_posix()] = _sha256(train_path)
        fold_hashes[test_relative.as_posix()] = _sha256(test_path)

    combined_test_indices: set[int] = set()
    for fold, test_set in enumerate(test_sets):
        if combined_test_indices & test_set:
            raise ValueError(f"test fold {fold} overlaps an earlier test fold")
        combined_test_indices.update(test_set)
    if combined_test_indices != all_indices:
        raise ValueError("ten test folds do not form a union of all 1319 samples")

    return {
        "author_repo": str(author_repo),
        "head": head,
        "tracked_worktree_clean": True,
        "arrays": {
            "features": {
                "path": feature_relative.as_posix(),
                "shape": list(features.shape),
                "dtype": str(features.dtype),
                "sha256": _sha256(feature_path),
            },
            "labels": {
                "path": label_relative.as_posix(),
                "shape": list(labels.shape),
                "dtype": str(labels.dtype),
                "sha256": _sha256(label_path),
            },
        },
        "class_counts": class_counts,
        "folds": {
            "fold_count": len(test_sets),
            "test_union_count": len(combined_test_indices),
            "sha256": fold_hashes,
        },
    }


def _record_check(
    audits: dict[str, Any],
    errors: list[dict[str, str]],
    name: str,
    check: Callable[[], dict[str, Any]],
) -> None:
    try:
        audits[name] = check()
    except Exception as exc:
        errors.append(
            {
                "check": name,
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )


def run_preflight(
    config_path: Path, device_override: str | None = None
) -> dict[str, Any]:
    config_path = Path(config_path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("configuration root must be a JSON object")
    except Exception as exc:
        return {
            "status": "blocked",
            "config_path": str(config_path),
            "device": device_override,
            "audits": {},
            "errors": [
                {
                    "check": "config",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            ],
        }

    device = str(device_override or config.get("device", "cuda"))
    audits: dict[str, Any] = {}
    errors: list[dict[str, str]] = []

    def configured_path(key: str) -> Path:
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"configuration key {key!r} must be a non-empty path string")
        return Path(value)

    _record_check(
        audits,
        errors,
        "runtime",
        lambda: validate_runtime(
            device=device,
            run_root=configured_path("run_root"),
            minimum_free_gib=float(
                config.get("minimum_free_gib", DEFAULT_MINIMUM_FREE_GIB)
            ),
        ),
    )
    _record_check(
        audits,
        errors,
        "project_features",
        lambda: audit_project_feature_table(configured_path("project_features")),
    )
    _record_check(
        audits,
        errors,
        "external_alignment",
        lambda: audit_external_alignment(
            configured_path("project_features"), configured_path("external_features")
        ),
    )
    _record_check(
        audits,
        errors,
        "metadata",
        lambda: audit_metadata_table(configured_path("metadata")),
    )
    _record_check(
        audits,
        errors,
        "author_artifacts",
        lambda: audit_author_artifacts(
            configured_path("author_repo"), str(config.get("author_commit", ""))
        ),
    )
    return {
        "status": "ready" if not errors else "blocked",
        "config_path": str(config_path),
        "device": device,
        "audits": audits,
        "errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the DNDT/DNDF two-day experiment runtime and inputs."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = run_preflight(args.config, device_override=args.device)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
