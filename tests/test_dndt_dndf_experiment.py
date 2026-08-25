from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from covid_rars.dndt_dndf_experiment import (
    PREDICTION_COLUMNS,
    FitResult,
    PlannedInterruption,
    TrainConfig,
    _atomic_torch_save,
    _manifest_path,
    _publish_checkpoint_generation,
    _resolve_checkpoint_manifest,
    aggregate_participant_probabilities,
    balance_training_rows,
    checkpoint_sha256,
    deterministic_batch_indices,
    fit_model,
    fit_preprocessor,
    normalize_execution_backend,
    transform_features,
    validate_prediction_frame,
    write_predictions,
)
from covid_rars.dndt_dndf_models import ModelConfig


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _tiny_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(91)
    train_x = rng.normal(size=(20, 4)).astype(np.float32)
    train_y = np.array([0, 1] * 10, dtype=np.int64)
    train_x[:, 0] += train_y * 1.25
    validation_x = rng.normal(size=(10, 4)).astype(np.float32)
    validation_y = np.array([0, 1] * 5, dtype=np.int64)
    validation_x[:, 0] += validation_y * 1.25
    return train_x, train_y, validation_x, validation_y


def _model_config() -> ModelConfig:
    return ModelConfig("dndt", num_trees=1, depth=2, used_features_rate=1.0)


def _train_config(**overrides: object) -> TrainConfig:
    values: dict[str, object] = {
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "batch_size": 6,
        "max_epochs": 5,
        "patience": 5,
        "seed": 17,
        "balance_method": "class_weight",
    }
    values.update(overrides)
    return TrainConfig(**values)  # type: ignore[arg-type]


def _fit(
    directory: Path,
    *,
    train_config: TrainConfig | None = None,
    feature_sha: str | None = None,
    split_sha: str | None = None,
    code_revision: str = "task3-test-revision",
    resume: bool = False,
    interrupt_after_epoch: int | None = None,
    device: str = "cpu",
) -> FitResult:
    train_x, train_y, validation_x, validation_y = _tiny_data()
    return fit_model(
        train_x,
        train_y,
        validation_x,
        validation_y,
        model_config=_model_config(),
        train_config=train_config or _train_config(),
        checkpoint_dir=directory,
        input_feature_hash=feature_sha or _sha("features"),
        split_hash=split_sha or _sha("split"),
        code_revision=code_revision,
        device=device,
        resume=resume,
        interrupt_after_epoch=interrupt_after_epoch,
        prediction_batch_size=4,
    )


def test_train_config_and_fit_result_are_frozen_and_validate_contract() -> None:
    config = _train_config()
    with pytest.raises(FrozenInstanceError):
        config.seed = 4  # type: ignore[misc]
    assert tuple(TrainConfig.__dataclass_fields__) == (
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "seed",
        "balance_method",
    )
    assert tuple(FitResult.__dataclass_fields__) == (
        "best_epoch",
        "validation_auroc",
        "validation_auprc",
        "threshold",
        "checkpoint_path",
        "validation_probability",
    )
    for updates in (
        {"learning_rate": 0.0},
        {"weight_decay": -1.0},
        {"batch_size": 1},
        {"max_epochs": 0},
        {"patience": 0},
        {"seed": -1},
        {"balance_method": "bad"},
    ):
        with pytest.raises(ValueError):
            _train_config(**updates)


def test_preprocessor_statistics_use_training_rows_only() -> None:
    train = np.array([[0.0, np.nan], [2.0, 4.0]], dtype=np.float64)
    validation = np.array([[1000.0, 1000.0]], dtype=np.float64)
    fitted = fit_preprocessor(train)

    assert fitted.scaler.mean_[0] == pytest.approx(1.0)
    assert fitted.imputer.statistics_[1] == pytest.approx(4.0)
    assert transform_features(fitted, validation)[0, 0] > 100.0


def test_preprocessor_rejects_all_missing_and_nonfinite_transforms() -> None:
    with pytest.raises(ValueError, match="all-missing"):
        fit_preprocessor(np.array([[1.0, np.nan], [2.0, np.nan]]))
    fitted = fit_preprocessor(np.array([[0.0, 1.0], [2.0, 3.0]]))
    with pytest.raises(ValueError, match="finite"):
        transform_features(fitted, np.array([[np.inf, 2.0]]))


def test_oversampling_uses_only_supplied_training_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, np.ndarray] = {}

    class SpySmote:
        def __init__(self, *, random_state: int, k_neighbors: int) -> None:
            observed["seed"] = np.array([random_state])
            observed["k_neighbors"] = np.array([k_neighbors])

        def fit_resample(self, features: np.ndarray, labels: np.ndarray):
            observed["features"] = features.copy()
            observed["labels"] = labels.copy()
            return features, labels

    import covid_rars.dndt_dndf_experiment as experiment

    monkeypatch.setattr(experiment, "SMOTE", SpySmote)
    train_x = np.arange(48, dtype=float).reshape(12, 4)
    train_y = np.array([0] * 6 + [1] * 6)
    result = balance_training_rows(train_x, train_y, method="smote", seed=9)

    np.testing.assert_array_equal(observed["features"], train_x)
    np.testing.assert_array_equal(observed["labels"], train_y)
    np.testing.assert_array_equal(observed["k_neighbors"], [5])
    assert result.audit["requested_method"] == "smote"
    assert result.audit["applied_method"] == "smote"


@pytest.mark.parametrize("method", ["smote", "svm_smote"])
def test_too_few_minority_rows_fall_back_without_data_loss(method: str) -> None:
    features = np.arange(28, dtype=float).reshape(7, 4)
    labels = np.array([0, 0, 0, 0, 0, 0, 1])
    result = balance_training_rows(features, labels, method=method, seed=3)

    np.testing.assert_array_equal(result.features, features)
    np.testing.assert_array_equal(result.labels, labels)
    assert result.sample_weights is not None
    assert result.audit["applied_method"] == "class_weight"
    assert "minority" in str(result.audit["fallback_reason"]).lower()
    assert result.audit["n_rows_before"] == result.audit["n_rows_after"] == 7


def test_svm_smote_accepts_six_minority_rows_with_default_neighbors() -> None:
    rng = np.random.default_rng(4)
    features = np.vstack(
        [rng.normal(0.0, 1.0, size=(20, 3)), rng.normal(0.35, 1.0, size=(6, 3))]
    )
    labels = np.array([0] * 20 + [1] * 6)

    result = balance_training_rows(features, labels, method="svm_smote", seed=7)

    assert result.audit["applied_method"] == "svm_smote"
    assert result.audit["fallback_reason"] is None
    assert result.audit["n_rows_after"] > result.audit["n_rows_before"]
    assert result.sample_weights is None


def test_svm_smote_five_minority_rows_fall_back_for_k_neighbors() -> None:
    features = np.arange(75, dtype=float).reshape(25, 3)
    labels = np.array([0] * 20 + [1] * 5)

    result = balance_training_rows(features, labels, method="svm_smote", seed=7)

    assert result.audit["applied_method"] == "class_weight"
    assert "k_neighbors=5" in str(result.audit["fallback_reason"])
    assert "minority count 5" in str(result.audit["fallback_reason"])


def test_svm_smote_total_neighborhood_shortfall_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_rars.dndt_dndf_experiment as experiment

    monkeypatch.setattr(experiment, "SVMSMOTE_M_NEIGHBORS", 30, raising=False)
    features = np.arange(78, dtype=float).reshape(26, 3)
    labels = np.array([0] * 20 + [1] * 6)

    result = balance_training_rows(features, labels, method="svm_smote", seed=7)

    assert result.audit["applied_method"] == "class_weight"
    assert "total row count 26" in str(result.audit["fallback_reason"])
    assert "m_neighbors=30" in str(result.audit["fallback_reason"])


def test_svm_smote_value_error_falls_back_with_sampler_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_rars.dndt_dndf_experiment as experiment

    class FailingSvmSmote:
        def __init__(
            self, *, random_state: int, k_neighbors: int, m_neighbors: int
        ) -> None:
            assert (random_state, k_neighbors, m_neighbors) == (7, 5, 10)

        def fit_resample(self, *_args: object) -> tuple[np.ndarray, np.ndarray]:
            raise ValueError("forced neighbor failure")

    monkeypatch.setattr(experiment, "SVMSMOTE", FailingSvmSmote)
    features = np.arange(78, dtype=float).reshape(26, 3)
    labels = np.array([0] * 20 + [1] * 6)

    result = balance_training_rows(features, labels, method="svm_smote", seed=7)

    assert result.audit["applied_method"] == "class_weight"
    assert "SVMSMOTE raised ValueError" in str(result.audit["fallback_reason"])
    assert "forced neighbor failure" in str(result.audit["fallback_reason"])


@pytest.mark.parametrize(
    "n_rows,batch_size,expected_sizes",
    [(5, 2, [2, 3]), (9, 4, [4, 5]), (3, 2, [3]), (2, 8, [2])],
)
def test_deterministic_batches_use_every_row_once_without_singletons(
    n_rows: int, batch_size: int, expected_sizes: list[int]
) -> None:
    first = deterministic_batch_indices(n_rows, batch_size, seed=11, epoch=2)
    second = deterministic_batch_indices(n_rows, batch_size, seed=11, epoch=2)
    other_epoch = deterministic_batch_indices(n_rows, batch_size, seed=11, epoch=3)

    assert [len(batch) for batch in first] == expected_sizes
    assert all(len(batch) >= 2 for batch in first)
    np.testing.assert_array_equal(np.concatenate(first), np.concatenate(second))
    assert sorted(np.concatenate(first).tolist()) == list(range(n_rows))
    if n_rows > 2:
        assert not np.array_equal(np.concatenate(first), np.concatenate(other_epoch))


def test_fit_model_has_no_test_data_surface() -> None:
    parameters = set(inspect.signature(fit_model).parameters)
    assert not any("test" in parameter.lower() for parameter in parameters)


def test_interruption_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    uninterrupted = _fit(tmp_path / "full")
    with pytest.raises(PlannedInterruption, match="epoch 2"):
        _fit(tmp_path / "resumed", interrupt_after_epoch=2)
    resumed = _fit(tmp_path / "resumed", resume=True)

    np.testing.assert_allclose(
        resumed.validation_probability,
        uninterrupted.validation_probability,
        rtol=0.0,
        atol=1e-7,
    )
    assert resumed.best_epoch == uninterrupted.best_epoch
    assert resumed.validation_auroc == pytest.approx(uninterrupted.validation_auroc)
    assert resumed.validation_auprc == pytest.approx(uninterrupted.validation_auprc)
    assert resumed.threshold == pytest.approx(uninterrupted.threshold)
    assert checkpoint_sha256(resumed.checkpoint_path) == checkpoint_sha256(
        uninterrupted.checkpoint_path
    )
    assert _manifest_path(tmp_path / "resumed", "latest_recovery").exists()
    assert _manifest_path(tmp_path / "resumed", "best_inference").exists()
    assert (tmp_path / "resumed" / "completion.json").exists()


def test_resume_does_not_train_past_an_already_reached_patience_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_rars.dndt_dndf_experiment as experiment

    monkeypatch.setattr(
        experiment, "_safe_validation_metrics", lambda *_args: (0.5, 0.5)
    )
    config = _train_config(max_epochs=5, patience=1)
    full = _fit(tmp_path / "full-stop", train_config=config)
    with pytest.raises(PlannedInterruption):
        _fit(
            tmp_path / "resumed-stop",
            train_config=config,
            interrupt_after_epoch=2,
        )
    resumed = _fit(tmp_path / "resumed-stop", train_config=config, resume=True)

    assert full.best_epoch == resumed.best_epoch == 1
    full_latest = _resolve_checkpoint_manifest(
        tmp_path / "full-stop", role="latest_recovery", map_location="cpu"
    )
    resumed_latest = _resolve_checkpoint_manifest(
        tmp_path / "resumed-stop", role="latest_recovery", map_location="cpu"
    )
    assert full_latest.descriptor["sha256"] == resumed_latest.descriptor["sha256"]
    np.testing.assert_allclose(
        full.validation_probability, resumed.validation_probability, atol=1e-7
    )


def test_single_class_validation_has_deterministic_nan_metrics_and_receipt(
    tmp_path: Path,
) -> None:
    train_x, train_y, validation_x, _ = _tiny_data()
    result = fit_model(
        train_x,
        train_y,
        validation_x,
        np.zeros(validation_x.shape[0], dtype=np.int64),
        model_config=_model_config(),
        train_config=_train_config(max_epochs=2, patience=2),
        checkpoint_dir=tmp_path,
        input_feature_hash=_sha("features"),
        split_hash=_sha("single-class-split"),
        code_revision="task3-test-revision",
    )

    assert math.isnan(result.validation_auroc)
    assert math.isnan(result.validation_auprc)
    assert result.best_epoch == 1
    receipt = pd.read_json(tmp_path / "completion.json", typ="series")
    assert pd.isna(receipt["validation_auroc"])
    assert pd.isna(receipt["validation_auprc"])


def test_resume_reconstructs_best_checkpoint_from_durable_latest_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_rars.dndt_dndf_experiment as experiment

    monkeypatch.setattr(
        experiment, "_safe_validation_metrics", lambda *_args: (0.5, 0.5)
    )
    with pytest.raises(PlannedInterruption):
        _fit(tmp_path, interrupt_after_epoch=2)
    best_manifest = _manifest_path(tmp_path, "best_inference")
    manifest = json.loads(best_manifest.read_text(encoding="utf-8"))
    for descriptor in (manifest.get("current"), manifest.get("previous")):
        if descriptor:
            (tmp_path / descriptor["filename"]).unlink(missing_ok=True)
    best_manifest.unlink()

    result = _fit(tmp_path, resume=True)

    assert result.checkpoint_path.is_file()
    rebuilt = _resolve_checkpoint_manifest(
        tmp_path, role="best_inference", map_location="cpu"
    )
    assert result.checkpoint_path == rebuilt.path
    assert checkpoint_sha256(result.checkpoint_path) == rebuilt.descriptor["sha256"]


@pytest.mark.parametrize("mismatch", ["feature", "split", "config", "code"])
def test_resume_rejects_provenance_mismatch_before_checkpoint_mutation(
    tmp_path: Path, mismatch: str
) -> None:
    run_dir = tmp_path / mismatch
    with pytest.raises(PlannedInterruption):
        _fit(run_dir, interrupt_after_epoch=2)
    latest = _resolve_checkpoint_manifest(
        run_dir, role="latest_recovery", map_location="cpu"
    )
    digest_before = latest.descriptor["sha256"]
    kwargs: dict[str, object] = {"resume": True}
    if mismatch == "feature":
        kwargs["feature_sha"] = _sha("other-features")
    elif mismatch == "split":
        kwargs["split_sha"] = _sha("other-split")
    elif mismatch == "config":
        kwargs["train_config"] = _train_config(learning_rate=0.02)
    else:
        kwargs["code_revision"] = "different-revision"

    with pytest.raises(ValueError, match="mismatch"):
        _fit(run_dir, **kwargs)  # type: ignore[arg-type]
    latest_after = _resolve_checkpoint_manifest(
        run_dir, role="latest_recovery", map_location="cpu"
    )
    assert latest_after.descriptor["sha256"] == digest_before


def test_atomic_checkpoint_removes_temporary_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "latest.pt"

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated"):
        _atomic_torch_save({"epoch": 1}, destination)
    assert not destination.exists()
    assert not (tmp_path / ".latest.pt.tmp").exists()


def test_generation_publish_failure_before_manifest_keeps_old_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_rars.dndt_dndf_experiment as experiment

    first = _publish_checkpoint_generation(
        {"marker": 1}, tmp_path, role="latest_recovery", epoch=1
    )
    manifest_path = _manifest_path(tmp_path, "latest_recovery")
    manifest_before = manifest_path.read_bytes()

    def fail_manifest(*_args: object, **_kwargs: object) -> None:
        raise OSError("power loss before manifest replacement")

    monkeypatch.setattr(experiment, "_atomic_write_manifest", fail_manifest)
    with pytest.raises(OSError, match="power loss"):
        _publish_checkpoint_generation(
            {"marker": 2}, tmp_path, role="latest_recovery", epoch=2
        )

    assert manifest_path.read_bytes() == manifest_before
    resolved = _resolve_checkpoint_manifest(
        tmp_path, role="latest_recovery", map_location="cpu"
    )
    assert resolved.path == first.path
    assert resolved.payload["marker"] == 1
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("failure", ["corrupt", "missing"])
def test_manifest_resolution_falls_back_without_deserializing_invalid_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    first = _publish_checkpoint_generation(
        {"marker": 1}, tmp_path, role="latest_recovery", epoch=1
    )
    second = _publish_checkpoint_generation(
        {"marker": 2}, tmp_path, role="latest_recovery", epoch=2
    )
    if failure == "corrupt":
        second.path.write_bytes(b"corrupt generation")
    else:
        second.path.unlink()

    real_load = torch.load
    loaded_paths: list[Path] = []

    def observe_load(path: Path, *args: object, **kwargs: object):
        loaded_paths.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", observe_load)
    resolved = _resolve_checkpoint_manifest(
        tmp_path, role="latest_recovery", map_location="cpu"
    )

    assert resolved.used_fallback is True
    assert resolved.path == first.path
    assert resolved.payload["marker"] == 1
    assert loaded_paths == [first.path]


def test_manifest_descriptor_must_match_immutable_generation_identity(
    tmp_path: Path,
) -> None:
    first = _publish_checkpoint_generation(
        {"marker": 1}, tmp_path, role="latest_recovery", epoch=1
    )
    _publish_checkpoint_generation(
        {"marker": 2}, tmp_path, role="latest_recovery", epoch=2
    )
    manifest_path = _manifest_path(tmp_path, "latest_recovery")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current"]["epoch"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resolved = _resolve_checkpoint_manifest(
        tmp_path, role="latest_recovery", map_location="cpu"
    )

    assert resolved.used_fallback is True
    assert resolved.path == first.path


def test_successful_publish_retains_only_current_and_previous_generations(
    tmp_path: Path,
) -> None:
    for epoch in range(1, 5):
        _publish_checkpoint_generation(
            {"marker": epoch}, tmp_path, role="latest_recovery", epoch=epoch
        )

    manifest = json.loads(
        _manifest_path(tmp_path, "latest_recovery").read_text(encoding="utf-8")
    )
    retained = {
        manifest["current"]["filename"],
        manifest["previous"]["filename"],
    }
    actual = {path.name for path in tmp_path.glob("latest_recovery-g*.pt")}
    assert actual == retained
    assert not list(tmp_path.glob("*.tmp"))


def test_locked_stale_generation_cleanup_is_deferred_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _publish_checkpoint_generation(
        {"marker": 1}, tmp_path, role="latest_recovery", epoch=1
    )
    second = _publish_checkpoint_generation(
        {"marker": 2}, tmp_path, role="latest_recovery", epoch=2
    )
    real_unlink = Path.unlink
    attempted: list[str] = []

    def reject_locked_stale(path: Path, *args: object, **kwargs: object) -> None:
        if path.suffix == ".pt":
            attempted.append(path.name)
        if path == first.path:
            raise PermissionError("simulated Windows file lock")
        real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as locked:
        locked.setattr(Path, "unlink", reject_locked_stale)
        third = _publish_checkpoint_generation(
            {"marker": 3}, tmp_path, role="latest_recovery", epoch=3
        )

    manifest = json.loads(
        _manifest_path(tmp_path, "latest_recovery").read_text(encoding="utf-8")
    )
    assert manifest["current"]["filename"] == third.path.name
    assert manifest["previous"]["filename"] == second.path.name
    assert attempted == [first.path.name]
    assert third.deferred_cleanup == (first.path.name,)
    assert first.path.exists()
    resolved = _resolve_checkpoint_manifest(
        tmp_path, role="latest_recovery", map_location="cpu"
    )
    assert resolved.path == third.path
    assert resolved.payload["marker"] == 3

    fourth = _publish_checkpoint_generation(
        {"marker": 4}, tmp_path, role="latest_recovery", epoch=4
    )
    final_manifest = json.loads(
        _manifest_path(tmp_path, "latest_recovery").read_text(encoding="utf-8")
    )
    retained = {
        final_manifest["current"]["filename"],
        final_manifest["previous"]["filename"],
    }
    assert fourth.deferred_cleanup == ()
    assert {path.name for path in tmp_path.glob("latest_recovery-g*.pt")} == retained


@pytest.mark.parametrize("failure", ["corrupt", "missing"])
def test_resume_from_invalid_current_generation_matches_uninterrupted(
    tmp_path: Path, failure: str
) -> None:
    full = _fit(tmp_path / "full")
    with pytest.raises(PlannedInterruption):
        _fit(tmp_path / "recover", interrupt_after_epoch=2)
    current = _resolve_checkpoint_manifest(
        tmp_path / "recover", role="latest_recovery", map_location="cpu"
    )
    if failure == "corrupt":
        current.path.write_bytes(b"simulated interrupted disk write")
    else:
        current.path.unlink()

    resumed = _fit(tmp_path / "recover", resume=True)

    np.testing.assert_allclose(
        resumed.validation_probability, full.validation_probability, atol=1e-7
    )
    assert resumed.best_epoch == full.best_epoch
    assert checkpoint_sha256(resumed.checkpoint_path) == checkpoint_sha256(
        full.checkpoint_path
    )


def test_best_and_latest_checkpoint_roles_are_not_conflated(tmp_path: Path) -> None:
    result = _fit(tmp_path)
    latest = _resolve_checkpoint_manifest(
        tmp_path, role="latest_recovery", map_location="cpu"
    )
    best = _resolve_checkpoint_manifest(
        tmp_path, role="best_inference", map_location="cpu"
    )

    assert latest.payload["checkpoint_role"] == "latest_recovery"
    for key in (
        "optimizer_state",
        "completed_epoch",
        "rng_state",
        "early_stop_counter",
    ):
        assert key in latest.payload
        assert key not in best.payload
    assert best.payload["checkpoint_role"] == "best_inference"
    assert best.payload["best_epoch"] == result.best_epoch
    assert result.checkpoint_path == best.path
    receipt = json.loads((tmp_path / "completion.json").read_text(encoding="utf-8"))
    assert receipt["execution_backend"] == "cpu"
    assert receipt["checkpoints"]["latest_recovery"]["sha256"] == latest.descriptor[
        "sha256"
    ]
    assert receipt["checkpoints"]["best_inference"]["sha256"] == best.descriptor[
        "sha256"
    ]


def test_final_best_checkpoint_rejects_stale_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_rars.dndt_dndf_experiment as experiment

    monkeypatch.setattr(
        experiment, "_safe_validation_metrics", lambda *_args: (0.5, 0.5)
    )
    with pytest.raises(PlannedInterruption):
        _fit(tmp_path, interrupt_after_epoch=2)
    selected = _resolve_checkpoint_manifest(
        tmp_path, role="best_inference", map_location="cpu"
    )
    stale_payload = dict(selected.payload)
    stale_payload["input_feature_hash"] = _sha("stale-feature-contract")
    _publish_checkpoint_generation(
        stale_payload,
        tmp_path,
        role="best_inference",
        epoch=int(stale_payload["best_epoch"]),
    )

    result = _fit(tmp_path, resume=True)
    rebuilt = _resolve_checkpoint_manifest(
        tmp_path, role="best_inference", map_location="cpu"
    )

    assert result.checkpoint_path == rebuilt.path
    assert rebuilt.payload["input_feature_hash"] == _sha("features")


def test_backend_normalization_and_cpu_cuda_resume_mismatch(
    tmp_path: Path,
) -> None:
    assert normalize_execution_backend("cpu") == "cpu"
    assert normalize_execution_backend(torch.device("cpu")) == "cpu"
    with pytest.raises(PlannedInterruption):
        _fit(tmp_path / "cpu", interrupt_after_epoch=1, device="cpu")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    assert normalize_execution_backend("cuda") == normalize_execution_backend("cuda:0")
    with pytest.raises(ValueError, match="execution_backend.*mismatch"):
        _fit(tmp_path / "cpu", resume=True, device="cuda")

    with pytest.raises(PlannedInterruption):
        _fit(tmp_path / "cuda", interrupt_after_epoch=1, device="cuda:0")
    with pytest.raises(ValueError, match="execution_backend.*mismatch"):
        _fit(tmp_path / "cuda", resume=True, device="cpu")


def test_bare_cuda_normalizes_to_cuda_zero_independent_of_current_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    assert normalize_execution_backend("cuda") == "cuda:0"
    assert normalize_execution_backend("cuda:0") == "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_different_cuda_index_is_rejected_before_model_state_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_rars.dndt_dndf_experiment as experiment

    with pytest.raises(PlannedInterruption):
        _fit(tmp_path, interrupt_after_epoch=1, device="cuda:0")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def forbidden_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("model state must not load after backend mismatch")

    monkeypatch.setattr(experiment.NeuralDecisionClassifier, "load_state_dict", forbidden_load)
    with pytest.raises(ValueError, match="execution_backend.*mismatch"):
        _fit(tmp_path, resume=True, device="cuda:1")


def test_participant_aggregation_is_deterministic_and_rejects_label_conflicts() -> None:
    frame = pd.DataFrame(
        {
            "participant_id": ["p2", "p1", "p1"],
            "recording_id": ["r3", "r2", "r1"],
            "label_binary": [0, 1, 1],
            "probability": [0.2, 0.8, 0.6],
        }
    )
    result = aggregate_participant_probabilities(frame)
    assert result["participant_id"].tolist() == ["p1", "p2"]
    assert result["probability"].tolist() == pytest.approx([0.7, 0.2])
    assert result["n_recordings"].tolist() == [2, 1]
    assert bool((result["n_recordings"] > 0).all())

    conflicted = frame.copy()
    conflicted.loc[0, "participant_id"] = "p1"
    with pytest.raises(ValueError, match="conflicting labels"):
        aggregate_participant_probabilities(conflicted)

    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate recording"):
        aggregate_participant_probabilities(duplicated)


@pytest.mark.parametrize(
    "column,value",
    [
        ("participant_id", None),
        ("participant_id", "  "),
        ("recording_id", None),
        ("recording_id", ""),
    ],
)
def test_participant_aggregation_rejects_missing_or_blank_identifiers(
    column: str, value: object
) -> None:
    frame = pd.DataFrame(
        {
            "participant_id": ["p1"],
            "recording_id": ["r1"],
            "label_binary": [1],
            "probability": [0.7],
        }
    )
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match=column):
        aggregate_participant_probabilities(frame)


@pytest.mark.parametrize("label", [None, np.nan, "unknown", 2, -1])
def test_participant_aggregation_rejects_null_or_nonbinary_labels(label: object) -> None:
    frame = pd.DataFrame(
        {
            "participant_id": ["p1"],
            "recording_id": ["r1"],
            "label_binary": [label],
            "probability": [0.7],
        }
    )

    with pytest.raises(ValueError, match="label_binary"):
        aggregate_participant_probabilities(frame)


def test_participant_aggregation_treats_missing_and_known_labels_as_conflict() -> None:
    frame = pd.DataFrame(
        {
            "participant_id": ["p1", "p1"],
            "recording_id": ["r1", "r2"],
            "label_binary": [1, None],
            "probability": [0.7, 0.6],
        }
    )

    with pytest.raises(ValueError, match="conflicting labels"):
        aggregate_participant_probabilities(frame)


@pytest.mark.parametrize("probability", ["not-numeric", np.nan, np.inf, -0.01, 1.01])
def test_participant_aggregation_rejects_invalid_probability_before_grouping(
    probability: object,
) -> None:
    frame = pd.DataFrame(
        {
            "participant_id": ["p1"],
            "recording_id": ["r1"],
            "label_binary": [1],
            "probability": [probability],
        }
    )

    with pytest.raises(ValueError, match="probability"):
        aggregate_participant_probabilities(frame)


def _prediction_frame(track: str = "B") -> pd.DataFrame:
    row: dict[str, object] = {
        "run_id": "run-1",
        "track": track,
        "protocol": "existing",
        "fold": 0,
        "seed": 42,
        "dataset": "coswara",
        "split": "validation",
        "model_name": "dndf",
        "modality": "cough",
        "participant_id": "p1" if track != "A" else "",
        "recording_id": "r1",
        "label_binary": 1,
        "probability": 0.7,
        "threshold": 0.5,
        "threshold_source": "validation_balanced_accuracy",
        "configuration_sha256": "a" * 64,
        "split_sha256": "b" * 64,
        "feature_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
    }
    columns = list(PREDICTION_COLUMNS)
    if track == "A":
        row.update({"analysis_id": "sample-1", "analysis_unit": "author_sample"})
        columns.extend(["analysis_id", "analysis_unit"])
    return pd.DataFrame([row], columns=columns)


def test_prediction_schema_and_provenance_are_exact(tmp_path: Path) -> None:
    frame = _prediction_frame()
    validated = validate_prediction_frame(frame)
    assert validated.columns.tolist() == list(PREDICTION_COLUMNS)

    track_a = validate_prediction_frame(_prediction_frame("A"))
    assert track_a.loc[0, "analysis_unit"] == "author_sample"

    with pytest.raises(ValueError, match="columns"):
        validate_prediction_frame(frame.assign(unexpected="x"))
    with pytest.raises(ValueError, match="SHA256"):
        validate_prediction_frame(frame.assign(feature_sha256="not-a-hash"))
    with pytest.raises(ValueError, match="participant_id"):
        validate_prediction_frame(frame.assign(participant_id=""))

    output = write_predictions(frame, tmp_path / "predictions.csv")
    assert output == tmp_path / "predictions.csv"
    pd.testing.assert_frame_equal(pd.read_csv(output), frame, check_dtype=False)
    assert not (tmp_path / ".predictions.csv.tmp").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_resume_matches_uninterrupted_probabilities(tmp_path: Path) -> None:
    config = _train_config(max_epochs=3, patience=3)
    full = _fit(tmp_path / "cuda-full", train_config=config, device="cuda")
    with pytest.raises(PlannedInterruption):
        _fit(
            tmp_path / "cuda-resume",
            train_config=config,
            device="cuda",
            interrupt_after_epoch=1,
        )
    resumed = _fit(
        tmp_path / "cuda-resume", train_config=config, device="cuda:0", resume=True
    )
    np.testing.assert_allclose(
        resumed.validation_probability, full.validation_probability, rtol=0.0, atol=1e-6
    )
    assert checkpoint_sha256(resumed.checkpoint_path) == checkpoint_sha256(
        full.checkpoint_path
    )
    full_latest = _resolve_checkpoint_manifest(
        tmp_path / "cuda-full", role="latest_recovery", map_location="cpu"
    )
    resumed_latest = _resolve_checkpoint_manifest(
        tmp_path / "cuda-resume", role="latest_recovery", map_location="cpu"
    )
    assert resumed_latest.descriptor["sha256"] == full_latest.descriptor["sha256"]
