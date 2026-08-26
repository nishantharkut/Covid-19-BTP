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


def _load_track_a_cli() -> ModuleType:
    script_path = PROJECT_ROOT / "scripts" / "80_run_dndt_dndf_track_a.py"
    spec = importlib.util.spec_from_file_location("dndt_dndf_track_a_cli", script_path)
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


def _author_artifact_hashes(author_repo: Path) -> dict[str, str]:
    relative_paths = [
        Path("Extracted Features/Coswara/cough_X_features_np.npy"),
        Path("Extracted Features/Coswara/cough_y_features_np.npy"),
    ]
    for fold in range(10):
        relative_paths.extend(
            [
                Path(f"Train-Test Split/coswaradataset/train/{fold}.csv"),
                Path(f"Train-Test Split/coswaradataset/test/{fold}.csv"),
            ]
        )
    return {
        relative.as_posix(): _sha256(author_repo / relative)
        for relative in relative_paths
    }


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
            "learning_rate": 0.01,
            "batch_size": 16,
            "epochs": 14,
            "dndt_trees": 1,
            "dndf_trees": 25,
        },
        "balancing": {"track_a": "svm_smote", "track_b": "smote"},
        "selection": {
            "primary_metric": "auroc",
            "tie_breaker": "auprc",
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

    with pytest.raises(RuntimeError, match=r"5900000000 bytes.*6000000000 bytes"):
        module.validate_runtime(device="cuda", run_root=tmp_path, minimum_free_gib=0.0)


def test_preflight_rejects_one_byte_below_six_decimal_gb_without_rounding_ambiguity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_preflight()
    monkeypatch.setattr(
        module, "_load_torch", lambda: _fake_cuda_torch(5_999_999_999)
    )

    with pytest.raises(RuntimeError) as exc_info:
        module.validate_runtime(device="cuda", run_root=tmp_path, minimum_free_gib=0.0)

    assert str(exc_info.value) == (
        "CUDA device 0 has 5999999999 bytes of total memory "
        "(5.999999999 GB decimal; 5.587935447 GiB); "
        "at least 6000000000 bytes (6.000000000 GB decimal) are required"
    )


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


def test_external_features_require_all_project_identity_columns(tmp_path: Path) -> None:
    module = _load_preflight()
    project = _write_feature_fixture(tmp_path / "project.csv")
    external_identities = [
        column for column in IDENTITY_COLUMNS if column != "label_binary"
    ]
    external = _write_feature_fixture(
        tmp_path / "external-missing-label.csv",
        identity_columns=external_identities,
    )

    with pytest.raises(
        ValueError, match=r"external.*missing required identity columns.*label_binary"
    ):
        module.audit_external_alignment(project, external)


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


def test_run_preflight_ready_path_reuses_project_audit_and_is_json_serializable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_preflight()
    project = _write_feature_fixture(tmp_path / "project.csv")
    external = _write_feature_fixture(tmp_path / "external.csv")
    metadata = _write_csv(
        tmp_path / "metadata.csv",
        ["participant_id", "label_binary", "split", "recording_date"],
    )
    config_path = tmp_path / "ready.json"
    config_path.write_text(
        json.dumps(
            {
                "author_repo": str(tmp_path / "synthetic-author"),
                "author_commit": "a" * 40,
                "project_features": str(project),
                "external_features": str(external),
                "metadata": str(metadata),
                "run_root": str(tmp_path / "runs"),
                "device": "cpu",
                "minimum_free_gib": 0.0,
            }
        ),
        encoding="utf-8",
    )

    real_project_audit = module.audit_project_feature_table
    project_audit_calls = 0

    def counting_project_audit(path: Path) -> dict[str, object]:
        nonlocal project_audit_calls
        project_audit_calls += 1
        return real_project_audit(path)

    monkeypatch.setattr(module, "audit_project_feature_table", counting_project_audit)
    monkeypatch.setattr(
        module,
        "validate_runtime",
        lambda device, run_root, minimum_free_gib: {
            "device": device,
            "run_root": str(run_root),
            "run_root_writable": True,
            "minimum_free_gib": minimum_free_gib,
        },
    )
    monkeypatch.setattr(
        module,
        "audit_author_artifacts",
        lambda author_repo, expected_commit: {
            "author_repo": str(author_repo),
            "head": expected_commit,
            "tracked_worktree_clean": True,
        },
    )

    audit = module.run_preflight(config_path, device_override="cpu")

    assert audit["status"] == "ready"
    assert audit["errors"] == []
    assert set(audit["audits"]) == {
        "runtime",
        "project_features",
        "external_alignment",
        "metadata",
        "author_artifacts",
    }
    assert audit["audits"]["external_alignment"]["ordered_features_match"] is True
    assert project_audit_calls == 1
    assert json.loads(json.dumps(audit)) == audit


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


def test_track_a_loader_enforces_exact_artifacts_hashes_and_label_orientations(
    tmp_path: Path,
) -> None:
    from covid_rars.dndt_dndf_experiment import load_track_a_author_artifacts

    author_repo = tmp_path / "author"
    commit, _, _ = _write_author_repo(author_repo)
    expected_hashes = _author_artifact_hashes(author_repo)

    artifacts = load_track_a_author_artifacts(
        author_repo,
        commit,
        expected_hashes=expected_hashes,
    )

    assert artifacts.features.shape == (1319, 193)
    assert np.issubdtype(artifacts.features.dtype, np.floating)
    assert np.isfinite(artifacts.features).all()
    assert np.count_nonzero(artifacts.author_labels == 0) == 185
    assert np.count_nonzero(artifacts.author_labels == 1) == 1134
    np.testing.assert_array_equal(artifacts.covid_labels, 1 - artifacts.author_labels)
    assert np.all(artifacts.covid_labels[:185] == 1)
    assert np.all(artifacts.covid_labels[185:] == 0)
    combined = np.concatenate(artifacts.test_folds)
    np.testing.assert_array_equal(np.sort(combined), np.arange(1319))
    assert len(np.unique(combined)) == 1319
    assert len(artifacts.fold_hashes) == 20

    first_test = author_repo / "Train-Test Split" / "coswaradataset" / "test" / "0.csv"
    first_test.write_text(first_test.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        load_track_a_author_artifacts(
            author_repo,
            commit,
            expected_hashes=expected_hashes,
            require_clean_tracked_tree=False,
        )


def test_track_a_loader_rejects_nonfinite_features_even_with_matching_hash(
    tmp_path: Path,
) -> None:
    from covid_rars.dndt_dndf_experiment import load_track_a_author_artifacts

    author_repo = tmp_path / "author"
    commit, x_path, _ = _write_author_repo(author_repo)
    features = np.load(x_path)
    features[0, 0] = np.inf
    np.save(x_path, features)
    expected_hashes = _author_artifact_hashes(author_repo)

    with pytest.raises(ValueError, match="finite"):
        load_track_a_author_artifacts(
            author_repo,
            commit,
            expected_hashes=expected_hashes,
            require_clean_tracked_tree=False,
        )


def test_track_a_fold_batch_parser_and_cli_smoke_are_machine_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_track_a_cli()
    assert module.parse_fold_batch(None) == tuple(range(10))
    assert module.parse_fold_batch("2-4") == (2, 3, 4)
    assert module.parse_fold_batch("0,3,9") == (0, 3, 9)
    with pytest.raises(ValueError):
        module.parse_fold_batch("4-2")
    with pytest.raises(ValueError):
        module.parse_fold_batch("0,10")

    config_path = tmp_path / "config.json"
    original = {"run_root": str(tmp_path / "runs"), "device": "cpu"}
    config_path.write_text(json.dumps(original), encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_runner(config: dict[str, object], **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        assert config == original
        return {"status": "complete", "completed_units": 4, "total_units": 4}

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--run-id",
            "smoke-contract",
            "--resume",
            "--smoke",
            "--fold-batch",
            "0-3",
        ],
        runner=fake_runner,
        revision_resolver=lambda: "task4-clean-source-test",
    )

    assert exit_code == 0
    assert observed["run_id"] == "smoke-contract-smoke"
    assert observed["resume"] is True
    assert observed["smoke"] is True
    assert observed["fold_batch"] == (0,)
    assert json.loads(config_path.read_text(encoding="utf-8")) == original
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize(
    "arguments",
    (
        ("--config", "{config}"),
        ("--config", "{config}", "--run-id", "bad-range", "--fold-batch", "x-y"),
        ("--config", "{config}", "--run-id", "out-of-range", "--fold-batch", "0,10"),
        ("--config", "{config}", "--run-id", "bad-device", "--device", "quantum"),
    ),
)
def test_track_a_cli_parser_failures_use_one_machine_readable_envelope(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    resolved = tuple(
        str(config_path) if argument == "{config}" else argument
        for argument in arguments
    )
    process = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "80_run_dndt_dndf_track_a.py"),
            *resolved,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode != 0
    assert process.stderr == ""
    envelope = json.loads(process.stdout)
    assert envelope["status"] == "failed"
    assert envelope["error_type"] == "CliArgumentError"
    assert isinstance(envelope["message"], str) and envelope["message"]
    assert set(envelope) == {"status", "error_type", "message"}


def test_track_a_cli_allows_untracked_files_but_rejects_dirty_tracked_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_track_a_cli()
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
            text=True,
            check=True,
        )

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Track A Test")
    tracked = repository / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    git("add", "tracked.py")
    git("commit", "-m", "initial")
    expected_revision = git("rev-parse", "HEAD").stdout.strip()

    assert module._code_revision(repository) == expected_revision
    (repository / "untracked-runtime.json").write_text("{}\n", encoding="utf-8")
    assert module._code_revision(repository) == expected_revision

    tracked.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked source tree is dirty"):
        module._code_revision(repository)

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"run_root": str(tmp_path / "runs"), "device": "cpu"}),
        encoding="utf-8",
    )

    def forbidden_runner(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("scientific execution started with dirty tracked source")

    exit_code = module.main(
        ["--config", str(config_path), "--run-id", "dirty-source"],
        runner=forbidden_runner,
        revision_resolver=lambda: module._code_revision(repository),
    )
    envelope = json.loads(capsys.readouterr().out)
    assert exit_code != 0
    assert envelope["status"] == "failed"
    assert envelope["error_type"] == "RuntimeError"
    assert "tracked source tree is dirty" in envelope["message"]


def test_author_mode_rejects_late_fold_batch_without_preceding_state(
    tmp_path: Path,
) -> None:
    from covid_rars.dndt_dndf_experiment import run_track_a

    with pytest.raises(ValueError, match="preceding state"):
        run_track_a(
            {
                "run_root": str(tmp_path / "runs"),
                "device": "cpu",
                "published": {
                    "depth": 2,
                    "used_features_rate": 1.0,
                    "learning_rate": 0.01,
                    "batch_size": 8,
                    "epochs": 2,
                    "dndt_trees": 1,
                    "dndf_trees": 2,
                },
                "selection": {"patience": 2},
                "seeds": {"candidate": [42]},
            },
            run_id="late-start",
            fold_batch=(1,),
            modes=("author_behaviour_audit",),
            model_names=("dndt",),
            code_revision="task4-test",
            device="cpu",
        )
