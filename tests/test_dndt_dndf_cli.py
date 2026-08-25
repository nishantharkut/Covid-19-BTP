from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_COLUMNS = [
    "recording_id",
    "participant_id",
    "dataset",
    "modality",
    "submodality",
    "label_binary",
    "split",
]


def _load_preflight() -> ModuleType:
    script_path = PROJECT_ROOT / "scripts" / "79_dndt_dndf_preflight.py"
    spec = importlib.util.spec_from_file_location("dndt_dndf_preflight", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, header: list[str], row: list[object] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        if row is not None:
            writer.writerow(row)
    return path


def _write_feature_fixture(
    path: Path,
    *,
    feature_names: list[str] | None = None,
    identity_columns: list[str] | None = None,
    feature_values: list[object] | None = None,
) -> Path:
    features = feature_names or [f"feature_{index:03d}" for index in range(800)]
    identities = identity_columns or IDENTITY_COLUMNS
    identity_values: list[object] = ["r1", "p1", "coswara", "cough", "heavy", 1, "train"]
    values = (
        feature_values
        if feature_values is not None
        else [float(index) for index in range(len(features))]
    )
    assert len(values) == len(features)
    return _write_csv(
        path,
        identities + features,
        identity_values[: len(identities)] + values,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fake_cuda_torch(total_memory_bytes: int) -> SimpleNamespace:
    class Scalar:
        def __init__(self, value: float | bool) -> None:
            self.value = value

        def item(self) -> float | bool:
            return self.value

    class Tensor:
        def __mul__(self, other: object) -> Tensor:
            return self

        def sum(self) -> Scalar:
            return Scalar(14.0)

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_properties(index: int) -> SimpleNamespace:
            assert index == 0
            return SimpleNamespace(
                name="NVIDIA GeForce RTX 4050 Laptop GPU",
                total_memory=total_memory_bytes,
            )

    return SimpleNamespace(
        cuda=Cuda(),
        float32="float32",
        tensor=lambda *args, **kwargs: Tensor(),
        isfinite=lambda value: Scalar(True),
    )


def _write_author_repo(path: Path) -> tuple[str, Path, Path]:
    x_path = path / "Extracted Features" / "Coswara" / "cough_X_features_np.npy"
    y_path = path / "Extracted Features" / "Coswara" / "cough_y_features_np.npy"
    x_path.parent.mkdir(parents=True)
    labels = np.array(["C"] * 185 + ["N"] * 1134)
    np.save(x_path, np.arange(1319 * 193, dtype=np.float64).reshape(1319, 193))
    np.save(y_path, labels)

    all_indices = np.arange(1319)
    for fold, test_indices in enumerate(np.array_split(all_indices, 10)):
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        train_path = path / "Train-Test Split" / "coswaradataset" / "train" / f"{fold}.csv"
        test_path = path / "Train-Test Split" / "coswaradataset" / "test" / f"{fold}.csv"
        _write_csv(
            train_path,
            ["train_index", "label_up"],
        )
        _write_csv(
            test_path,
            ["test_index", "label_up"],
        )
        with train_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows((int(index), labels[index]) for index in train_indices)
        with test_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows((int(index), labels[index]) for index in test_indices)

    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Task Test"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "task-test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, x_path, y_path


def test_dependency_and_config_contracts_are_exact() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".venv-dndt-dndf/" in gitignore
    assert "configs/*.local.json" in gitignore

    requirements = (PROJECT_ROOT / "requirements-dndt-dndf.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert requirements == [
        "-r requirements.txt",
        "-e .",
        "imbalanced-learn>=0.14,<0.15",
        "optuna>=4.5,<5",
        "joblib>=1.5,<2",
        "jupyterlab>=4.4,<5",
        "ipykernel>=6.30,<7",
        "pytest>=8.4,<9",
    ]
    assert all("torch" not in requirement.lower() for requirement in requirements)

    config = json.loads(
        (PROJECT_ROOT / "configs" / "dndt_dndf_two_day.json").read_text(
            encoding="utf-8"
        )
    )
    assert config == {
        "author_repo": "G:/Covid-19-BTP/external/COVID-19-Detection-from-Cough-Sound",
        "author_commit": "feb0e63c790c042eaa21e9f3fc83ed64bdc8a24e",
        "project_features": "G:/Covid-19-BTP/covid_audio_btp/data/processed/features_compare_is10_top800.csv",
        "external_features": "G:/Covid-19-BTP/covid_audio_btp/data/processed/coughvid_features_compare_is10_top800.csv",
        "metadata": "G:/Covid-19-BTP/covid_audio_btp/data/processed/metadata_with_quality.csv",
        "run_root": "G:/Covid-19-BTP/dndt_dndf_runs",
        "device": "cuda",
        "published": {
            "depth": 11,
            "used_features_rate": 0.6,
            "lr": 0.01,
            "batch": 16,
            "epochs": 14,
            "dndt_trees": 1,
            "dndf_trees": 25,
        },
        "balancing": {"track_a": "svm_smote", "track_b": "smote"},
        "selection": {
            "primary": "auroc",
            "tie": "auprc",
            "max_trials_per_modality": 6,
            "patience": 3,
        },
        "seeds": {"candidate": [42], "final": [42, 314, 2026]},
        "modalities": ["breath", "cough", "speech"],
        "protocols": [
            "existing",
            "time_stratified",
            "early_to_late",
            "external_cough",
        ],
    }


def test_preflight_rejects_cpu_only_torch_for_cuda_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_preflight()

    class CpuOnlyCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class CpuOnlyTorch:
        cuda = CpuOnlyCuda()

    monkeypatch.setattr(module, "_load_torch", lambda: CpuOnlyTorch())

    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        module.validate_runtime(device="cuda", run_root=tmp_path, minimum_free_gib=0.0)


def test_preflight_accepts_nominal_six_gb_rtx_4050_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_preflight()
    total_memory_bytes = 6_438_780_928
    monkeypatch.setattr(
        module, "_load_torch", lambda: _fake_cuda_torch(total_memory_bytes)
    )

    audit = module.validate_runtime(
        device="cuda", run_root=tmp_path, minimum_free_gib=0.0
    )

    assert audit["cuda_total_memory_gb"] == pytest.approx(6.438780928)
    assert audit["cuda_total_memory_gib"] == pytest.approx(
        total_memory_bytes / 1024**3
    )


def test_preflight_rejects_materially_smaller_than_six_decimal_gb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_preflight()
    monkeypatch.setattr(
        module, "_load_torch", lambda: _fake_cuda_torch(5_900_000_000)
    )

    with pytest.raises(RuntimeError, match=r"at least 6\.00 GB \(decimal\)"):
        module.validate_runtime(device="cuda", run_root=tmp_path, minimum_free_gib=0.0)


def test_preflight_accepts_exact_project_feature_schema(tmp_path: Path) -> None:
    module = _load_preflight()
    path = _write_feature_fixture(tmp_path / "features.csv")
    feature_names = [f"feature_{index:03d}" for index in range(800)]

    audit = module.audit_project_feature_table(path)

    assert audit == {
        "path": str(path),
        "column_count": 807,
        "feature_count": 800,
        "required_columns_present": True,
        "feature_columns": feature_names,
    }


def test_project_feature_schema_rejects_free_text_as_feature(tmp_path: Path) -> None:
    module = _load_preflight()
    features = [f"feature_{index:03d}" for index in range(799)] + ["clinical_notes"]
    path = _write_feature_fixture(
        tmp_path / "free-text.csv",
        feature_names=features,
        feature_values=[float(index) for index in range(799)] + ["persistent cough"],
    )

    with pytest.raises(ValueError, match=r"clinical_notes.*nonnumeric"):
        module.audit_project_feature_table(path)


@pytest.mark.parametrize(
    ("invalid_value", "message"),
    [(float("inf"), "infinite"), ("", "no finite numeric values")],
)
def test_project_feature_schema_rejects_infinite_or_all_missing_feature(
    tmp_path: Path, invalid_value: object, message: str
) -> None:
    module = _load_preflight()
    values: list[object] = [float(index) for index in range(800)]
    values[17] = invalid_value
    path = _write_feature_fixture(tmp_path / "non-finite.csv", feature_values=values)

    with pytest.raises(ValueError, match=rf"feature_017.*{message}"):
        module.audit_project_feature_table(path)


@pytest.mark.parametrize("fault", ["missing_identity", "duplicate_feature"])
def test_project_feature_schema_rejects_missing_or_duplicate_columns(
    tmp_path: Path, fault: str
) -> None:
    module = _load_preflight()
    if fault == "missing_identity":
        path = _write_feature_fixture(
            tmp_path / "missing.csv",
            identity_columns=IDENTITY_COLUMNS[:-1],
        )
        match = "missing required identity columns"
    else:
        features = [f"feature_{index:03d}" for index in range(799)] + ["feature_000"]
        path = _write_feature_fixture(tmp_path / "duplicate.csv", feature_names=features)
        match = "duplicate column names"

    with pytest.raises(ValueError, match=match):
        module.audit_project_feature_table(path)


def test_external_features_require_the_same_ordered_columns(tmp_path: Path) -> None:
    module = _load_preflight()
    project = _write_feature_fixture(tmp_path / "project.csv")
    external = _write_feature_fixture(tmp_path / "external.csv")

    assert module.audit_external_alignment(project, external) == {
        "project_path": str(project),
        "external_path": str(external),
        "feature_count": 800,
        "ordered_features_match": True,
    }

    reordered = [f"feature_{index:03d}" for index in range(800)]
    reordered[10], reordered[11] = reordered[11], reordered[10]
    mismatch = _write_feature_fixture(tmp_path / "mismatch.csv", feature_names=reordered)
    with pytest.raises(ValueError, match="ordered feature columns do not match"):
        module.audit_external_alignment(project, mismatch)


def test_external_features_reject_nonnumeric_content(tmp_path: Path) -> None:
    module = _load_preflight()
    project = _write_feature_fixture(tmp_path / "project.csv")
    values: list[object] = [float(index) for index in range(800)]
    values[23] = "free text"
    external = _write_feature_fixture(
        tmp_path / "external-free-text.csv", feature_values=values
    )

    with pytest.raises(ValueError, match=r"feature_023.*nonnumeric"):
        module.audit_external_alignment(project, external)


def test_metadata_requires_temporal_and_split_columns(tmp_path: Path) -> None:
    module = _load_preflight()
    metadata = _write_csv(
        tmp_path / "metadata.csv",
        ["participant_id", "label_binary", "split", "recording_date", "extra"],
    )
    assert module.audit_metadata_table(metadata) == {
        "path": str(metadata),
        "column_count": 5,
        "required_columns_present": True,
    }

    missing = _write_csv(
        tmp_path / "metadata-missing.csv",
        ["participant_id", "label_binary", "split"],
    )
    with pytest.raises(ValueError, match="recording_date"):
        module.audit_metadata_table(missing)


def test_author_audit_verifies_arrays_first_index_column_and_ten_folds(
    tmp_path: Path,
) -> None:
    module = _load_preflight()
    author_repo = tmp_path / "author"
    commit, x_path, y_path = _write_author_repo(author_repo)

    audit = module.audit_author_artifacts(author_repo, commit)

    assert audit["head"] == commit
    assert audit["tracked_worktree_clean"] is True
    assert audit["arrays"]["features"] == {
        "path": "Extracted Features/Coswara/cough_X_features_np.npy",
        "shape": [1319, 193],
        "dtype": "float64",
        "sha256": _sha256(x_path),
    }
    assert audit["arrays"]["labels"] == {
        "path": "Extracted Features/Coswara/cough_y_features_np.npy",
        "shape": [1319],
        "dtype": "<U1",
        "sha256": _sha256(y_path),
    }
    assert audit["class_counts"] == {"C": 185, "N": 1134}
    assert audit["folds"]["fold_count"] == 10
    assert audit["folds"]["test_union_count"] == 1319
    assert len(audit["folds"]["sha256"]) == 20


def test_run_preflight_and_cli_emit_blocked_json_without_real_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_preflight()
    config_path = tmp_path / "blocked.json"
    config_path.write_text(
        json.dumps(
            {
                "author_repo": str(tmp_path / "missing-author"),
                "author_commit": "f" * 40,
                "project_features": str(tmp_path / "missing-project.csv"),
                "external_features": str(tmp_path / "missing-external.csv"),
                "metadata": str(tmp_path / "missing-metadata.csv"),
                "run_root": str(tmp_path / "runs"),
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )

    audit = module.run_preflight(config_path, device_override="cpu")
    assert audit["status"] == "blocked"
    assert audit["device"] == "cpu"
    assert audit["errors"]
    json.dumps(audit)

    monkeypatch.setattr(
        sys,
        "argv",
        ["79_dndt_dndf_preflight.py", "--config", str(config_path), "--device", "cpu"],
    )
    with pytest.raises(SystemExit) as exc_info:
        module.main()
    assert exc_info.value.code == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "blocked"
