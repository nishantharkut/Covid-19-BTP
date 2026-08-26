from __future__ import annotations

import hashlib
import io
import inspect
import json
import math
import multiprocessing
import os
import shutil
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from covid_rars.dndt_dndf_experiment import (
    AuthorTrackAArtifacts,
    EXPECTED_TRACK_A_SELECTED_INDICES,
    PREDICTION_COLUMNS,
    FitResult,
    PlannedInterruption,
    TrackAFeatureCache,
    TrainConfig,
    _predict_probabilities,
    _track_a_corrected_fit_no_scaler,
    _atomic_torch_save,
    _exclusive_rfecv_cache_lock,
    _manifest_path,
    _publish_checkpoint_generation,
    _resolve_checkpoint_manifest,
    aggregate_participant_probabilities,
    author_threshold_to_covid_threshold,
    author_threshold_sweep,
    balance_training_rows,
    checkpoint_sha256,
    deterministic_batch_indices,
    fixed_order_batch_indices,
    fit_model,
    fit_preprocessor,
    normalize_execution_backend,
    prepare_track_a_rfecv_cache,
    run_track_a,
    transform_features,
    validate_prediction_frame,
    write_predictions,
)
from covid_rars.dndt_dndf_models import ModelConfig


def _malicious_checkpoint_side_effect(path: str) -> dict[str, object]:
    Path(path).write_text("executed", encoding="utf-8")
    return {}


class _MaliciousCheckpointValue:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _malicious_checkpoint_side_effect, (str(self.marker),)


def _rfecv_lock_contention_worker(
    lock_path_value: str,
    output_path_value: str,
    start_path_value: str,
    worker_id: int,
    iterations: int,
) -> None:
    lock_path = Path(lock_path_value)
    output_path = Path(output_path_value)
    start_path = Path(start_path_value)
    guard_path = lock_path.with_suffix(".guard")
    deadline = time.monotonic() + 15.0
    while not start_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("contention worker did not receive start signal")
        time.sleep(0.005)
    for iteration in range(iterations):
        with _exclusive_rfecv_cache_lock(
            lock_path,
            timeout=15.0,
            retry_interval=0.005,
        ):
            if guard_path.exists():
                raise RuntimeError("more than one process entered the lock")
            guard_path.write_text(str(worker_id), encoding="ascii")
            try:
                with output_path.open("a", encoding="ascii", newline="\n") as handle:
                    handle.write(f"{worker_id}:{iteration}\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                time.sleep(0.002)
            finally:
                guard_path.unlink(missing_ok=True)


def _rfecv_lock_holder_worker(lock_path_value: str, acquired_path_value: str) -> None:
    with _exclusive_rfecv_cache_lock(
        Path(lock_path_value), timeout=10.0, retry_interval=0.005
    ):
        Path(acquired_path_value).write_text("held", encoding="ascii")
        while True:
            time.sleep(1.0)


def _rfecv_lock_waiter_worker(lock_path_value: str, acquired_path_value: str) -> None:
    with _exclusive_rfecv_cache_lock(
        Path(lock_path_value), timeout=10.0, retry_interval=0.005
    ):
        Path(acquired_path_value).write_text("acquired", encoding="ascii")


def _rfecv_prepare_worker(
    run_root_value: str,
    start_path_value: str,
    ready_path_value: str,
    result_path_value: str,
    computation_path_value: str,
) -> None:
    import covid_rars.dndt_dndf_experiment as experiment

    class SlowRfecv:
        def __init__(self, **_: object) -> None:
            pass

        def fit(self, features: np.ndarray, labels: np.ndarray) -> SlowRfecv:
            del labels
            with Path(computation_path_value).open(
                "a", encoding="ascii", newline="\n"
            ) as handle:
                handle.write(f"{features.shape[0]}\n")
                handle.flush()
                os.fsync(handle.fileno())
            time.sleep(0.25)
            self.support_ = np.array([True, False, True, False, True, False])
            return self

    experiment.RFECV = SlowRfecv
    start_path = Path(start_path_value)
    Path(ready_path_value).write_text("ready", encoding="ascii")
    deadline = time.monotonic() + 15.0
    while not start_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("RFECV prepare worker did not receive start signal")
        time.sleep(0.005)
    cache = prepare_track_a_rfecv_cache(
        _tiny_track_a_artifacts(),
        Path(run_root_value),
        expected_indices=np.array([0, 2, 4]),
    )
    Path(result_path_value).write_text(
        json.dumps(
            {
                "cache_key": cache.cache_key,
                "features_sha256": cache.selected_features_sha256,
                "indices_sha256": cache.selected_indices_sha256,
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )


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
    loaded_digests: list[str] = []

    def observe_load(source: object, *args: object, **kwargs: object):
        assert isinstance(source, io.BytesIO)
        loaded_digests.append(hashlib.sha256(source.getvalue()).hexdigest())
        assert kwargs["weights_only"] is True
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(torch, "load", observe_load)
    resolved = _resolve_checkpoint_manifest(
        tmp_path, role="latest_recovery", map_location="cpu"
    )

    assert resolved.used_fallback is True
    assert resolved.path == first.path
    assert resolved.payload["marker"] == 1
    assert loaded_digests == [first.descriptor["sha256"]]


def test_checkpoint_loader_rejects_unsupported_pickle_without_execution(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "pickle-executed.txt"
    stream = io.BytesIO()
    torch.save(
        {
            "checkpoint_role": "latest_recovery",
            "unsafe": _MaliciousCheckpointValue(marker),
        },
        stream,
    )
    checkpoint_bytes = stream.getvalue()
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    filename = f"latest_recovery-g000001-e000001-{digest}.pt"
    (tmp_path / filename).write_bytes(checkpoint_bytes)
    (tmp_path / "latest_recovery.manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "role": "latest_recovery",
                "current": {
                    "filename": filename,
                    "sha256": digest,
                    "epoch": 1,
                    "generation": 1,
                    "kind": "recovery",
                },
                "previous": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="deserialization failed"):
        _resolve_checkpoint_manifest(
            tmp_path,
            role="latest_recovery",
            map_location="cpu",
        )

    assert not marker.exists()


def test_checkpoint_loader_deserializes_the_same_verified_immutable_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = _publish_checkpoint_generation(
        {"marker": "verified"},
        tmp_path,
        role="latest_recovery",
        epoch=1,
    )
    real_load = torch.load
    observed: list[tuple[object, bool | None]] = []

    def replace_path_after_read(source: object, *args: object, **kwargs: object):
        observed.append((source, kwargs.get("weights_only")))
        published.path.write_bytes(b"replacement after immutable read")
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(torch, "load", replace_path_after_read)
    resolved = _resolve_checkpoint_manifest(
        tmp_path,
        role="latest_recovery",
        map_location="cpu",
    )

    assert resolved.payload["marker"] == "verified"
    assert len(observed) == 1
    assert isinstance(observed[0][0], io.BytesIO)
    assert observed[0][1] is True


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
        track_a_provenance = {
            "analysis_id": "sample-1",
            "analysis_unit": "author_sample",
            "mode": "corrected_reference",
            "author_commit": "e" * 40,
            "feature_indices_sha256": "f" * 64,
            "feature_cache_key": "0" * 64,
            "code_revision": "task4-test-revision",
            "execution_backend": "cpu",
            "predicted_label": 1,
            "threshold_comparator": "ge",
        }
        row.update(track_a_provenance)
        columns.extend(track_a_provenance)
    return pd.DataFrame([row], columns=columns)


def test_prediction_schema_and_provenance_are_exact(tmp_path: Path) -> None:
    frame = _prediction_frame()
    validated = validate_prediction_frame(frame)
    assert validated.columns.tolist() == list(PREDICTION_COLUMNS)

    track_a = validate_prediction_frame(_prediction_frame("A"))
    assert track_a.loc[0, "analysis_unit"] == "author_sample"
    strict_boundary = _prediction_frame("A").assign(
        probability=0.5,
        threshold=0.5,
        threshold_comparator="gt",
        predicted_label=0,
    )
    validate_prediction_frame(strict_boundary)
    with pytest.raises(ValueError, match="threshold_comparator semantics"):
        validate_prediction_frame(strict_boundary.assign(predicted_label=1))

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


def _tiny_track_a_artifacts() -> AuthorTrackAArtifacts:
    rng = np.random.default_rng(2026)
    author_labels = np.array([0, 1] * 30, dtype=np.int64)
    features = rng.normal(size=(60, 6)).astype(np.float64)
    features[:, 0] += author_labels * 0.9
    test_folds = tuple(np.asarray(part, dtype=np.int64) for part in np.array_split(np.arange(60), 3))
    train_folds = tuple(
        np.setdiff1d(np.arange(60), test, assume_unique=True) for test in test_folds
    )
    return AuthorTrackAArtifacts(
        features=features,
        author_labels=author_labels,
        covid_labels=1 - author_labels,
        train_folds=train_folds,
        test_folds=test_folds,
        source_hashes={"features": _sha("author-x"), "labels": _sha("author-y")},
        fold_hashes={f"fold-{index}": _sha(f"fold-{index}") for index in range(3)},
        author_commit="f" * 40,
    )


def _tiny_track_a_cache(tmp_path: Path) -> TrackAFeatureCache:
    artifacts = _tiny_track_a_artifacts()
    selected = artifacts.features[:, :4].copy()
    selected_path = tmp_path / "cache" / "selected_features.npy"
    indices_path = tmp_path / "cache" / "selected_indices.npy"
    manifest_path = tmp_path / "cache" / "manifest.json"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(selected_path, selected)
    np.save(indices_path, np.arange(4, dtype=np.int64))
    manifest_path.write_text("{}\n", encoding="utf-8")
    return TrackAFeatureCache(
        selected_features=selected,
        selected_indices=np.arange(4, dtype=np.int64),
        cache_key=_sha("tiny-track-a-cache"),
        selected_features_sha256=checkpoint_sha256(selected_path),
        selected_indices_sha256=checkpoint_sha256(indices_path),
        manifest_path=manifest_path,
    )


def _tiny_track_a_config(tmp_path: Path) -> dict[str, object]:
    return {
        "author_commit": "f" * 40,
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
    }


@pytest.fixture(scope="module")
def completed_corrected_track_a_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, AuthorTrackAArtifacts, TrackAFeatureCache, dict[str, object]]:
    root = tmp_path_factory.mktemp("track-a-corrected-template")
    artifacts = _tiny_track_a_artifacts()
    cache = _tiny_track_a_cache(root)
    config = _tiny_track_a_config(root)
    run_track_a(
        config,
        run_id="resume",
        artifacts=artifacts,
        feature_cache=cache,
        fold_batch=(0,),
        modes=("corrected_reference",),
        model_names=("dndt",),
        code_revision="task4-provenance-test",
        device="cpu",
    )
    return root, artifacts, cache, config


def _copied_corrected_track_a_case(
    tmp_path: Path,
    template: tuple[
        Path,
        AuthorTrackAArtifacts,
        TrackAFeatureCache,
        dict[str, object],
    ],
) -> tuple[Path, AuthorTrackAArtifacts, TrackAFeatureCache, dict[str, object]]:
    template_root, artifacts, cache, _ = template
    destination = tmp_path / "runs" / "resume"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template_root / "runs" / "resume", destination)
    config = _tiny_track_a_config(tmp_path)
    return destination, artifacts, cache, config


def _resume_copied_corrected_track_a(
    config: dict[str, object],
    artifacts: AuthorTrackAArtifacts,
    cache: TrackAFeatureCache,
) -> dict[str, object]:
    return run_track_a(
        config,
        run_id="resume",
        resume=True,
        artifacts=artifacts,
        feature_cache=cache,
        fold_batch=(0,),
        modes=("corrected_reference",),
        model_names=("dndt",),
        code_revision="task4-provenance-test",
        device="cpu",
    )


def test_fixed_order_batches_repeat_stable_rows_and_never_shuffle() -> None:
    first = fixed_order_batch_indices(11, 4)
    second = fixed_order_batch_indices(11, 4)

    assert [batch.tolist() for batch in first] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
    assert [batch.tolist() for batch in second] == [batch.tolist() for batch in first]
    assert np.concatenate(first).tolist() == list(range(11))


def test_author_threshold_sweep_matches_independent_oracle_and_covid_confusion() -> None:
    from sklearn.metrics import confusion_matrix, roc_auc_score

    y_author = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
    probability_n = np.array([0.15, 0.45, 0.55, 0.85, 0.65, 0.25])
    thresholds = np.arange(0.0, 1.0, 0.001)
    scores = np.array(
        [roc_auc_score(y_author, (probability_n >= threshold).astype(int)) for threshold in thresholds]
    )

    audit = author_threshold_sweep(y_author, probability_n)

    expected_index = int(np.argmax(scores))
    assert audit["author_threshold"] == pytest.approx(float(thresholds[expected_index]))
    assert audit["paper_thresholded_auc"] == pytest.approx(float(scores[expected_index]))
    y_covid = 1 - y_author
    predicted_covid = 1 - (probability_n >= audit["author_threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_covid, predicted_covid, labels=[0, 1]).ravel()
    assert audit["tn"] == int(tn)
    assert audit["fp"] == int(fp)
    assert audit["fn"] == int(fn)
    assert audit["tp"] == int(tp)
    converted = author_threshold_to_covid_threshold(float(audit["author_threshold"]))
    np.testing.assert_array_equal(
        (1.0 - probability_n > converted).astype(int),
        predicted_covid,
    )


@pytest.mark.parametrize(
    ("author_threshold", "probability_n"),
    (
        (0.0, np.array([0.0, 0.4, 1.0], dtype=np.float64)),
        (0.999, np.array([0.998, 0.999, 1.0], dtype=np.float64)),
        (0.4, np.array([0.399, 0.4, 0.401], dtype=np.float64)),
    ),
)
def test_author_threshold_conversion_preserves_strict_covid_boundary_semantics(
    author_threshold: float,
    probability_n: np.ndarray,
) -> None:
    covid_threshold = author_threshold_to_covid_threshold(author_threshold)
    probability_covid = 1.0 - probability_n
    expected = 1 - (probability_n >= author_threshold).astype(np.int64)

    assert covid_threshold == pytest.approx(1.0 - author_threshold)
    np.testing.assert_array_equal(
        (probability_covid > covid_threshold).astype(np.int64),
        expected,
    )


def test_rfecv_cache_is_global_hashed_and_recomputed_after_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _tiny_track_a_artifacts()
    calls: list[tuple[int, int]] = []

    class FakeRfecv:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["step"] == 1
            assert kwargs["scoring"] == "roc_auc"
            assert kwargs["min_features_to_select"] == 1

        def fit(self, features: np.ndarray, labels: np.ndarray) -> FakeRfecv:
            calls.append(features.shape)
            self.support_ = np.array([True, False, True, False, True, False])
            return self

    import covid_rars.dndt_dndf_experiment as experiment

    monkeypatch.setattr(experiment, "RFECV", FakeRfecv)
    first = prepare_track_a_rfecv_cache(
        artifacts,
        tmp_path,
        expected_indices=np.array([0, 2, 4]),
    )
    second = prepare_track_a_rfecv_cache(
        artifacts,
        tmp_path,
        expected_indices=np.array([0, 2, 4]),
    )

    assert calls == [(48, 6)]
    np.testing.assert_array_equal(first.selected_indices, [0, 2, 4])
    np.testing.assert_array_equal(second.selected_features, artifacts.features[:, [0, 2, 4]])
    assert first.cache_key == second.cache_key
    assert json.loads(first.manifest_path.read_text(encoding="utf-8"))["rfecv_scope"] == "single_global_author_pass"

    selected_path = first.manifest_path.parent / "selected_features.npy"
    selected_path.write_bytes(b"corrupt")
    repaired = prepare_track_a_rfecv_cache(
        artifacts,
        tmp_path,
        expected_indices=np.array([0, 2, 4]),
    )
    assert calls == [(48, 6), (48, 6)]
    np.testing.assert_array_equal(repaired.selected_features, artifacts.features[:, [0, 2, 4]])
    assert not list(first.manifest_path.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    "corruption",
    (
        "source_hashes",
        "sklearn_version",
        "algorithm_settings",
        "rfecv_scope",
        "cache_key",
        "extra_manifest_field",
        "missing_manifest_field",
        "selected_features_bytes",
        "selected_indices_bytes",
    ),
)
def test_rfecv_cache_rejects_every_provenance_or_byte_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    artifacts = _tiny_track_a_artifacts()
    calls: list[tuple[int, int]] = []

    class FakeRfecv:
        def __init__(self, **_: object) -> None:
            pass

        def fit(self, features: np.ndarray, labels: np.ndarray) -> FakeRfecv:
            calls.append(features.shape)
            self.support_ = np.array([True, False, True, False, True, False])
            return self

    import covid_rars.dndt_dndf_experiment as experiment

    monkeypatch.setattr(experiment, "RFECV", FakeRfecv)
    cached = prepare_track_a_rfecv_cache(
        artifacts,
        tmp_path,
        expected_indices=np.array([0, 2, 4]),
    )
    manifest = json.loads(cached.manifest_path.read_text(encoding="utf-8"))
    if corruption == "source_hashes":
        manifest["source_hashes"]["features"] = _sha("wrong-source")
    elif corruption == "sklearn_version":
        manifest["sklearn_version"] = "0.0-corrupt"
    elif corruption == "algorithm_settings":
        manifest["algorithm"]["rfecv"]["step"] = 2
    elif corruption == "rfecv_scope":
        manifest["rfecv_scope"] = "per_fold"
    elif corruption == "cache_key":
        manifest["cache_key"] = _sha("wrong-key")
    elif corruption == "extra_manifest_field":
        manifest["unexpected"] = "not-allowed"
    elif corruption == "missing_manifest_field":
        del manifest["rfecv_scope"]
    elif corruption == "selected_features_bytes":
        (cached.manifest_path.parent / "selected_features.npy").write_bytes(b"corrupt")
    elif corruption == "selected_indices_bytes":
        (cached.manifest_path.parent / "selected_indices.npy").write_bytes(b"corrupt")
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(corruption)
    if corruption not in {"selected_features_bytes", "selected_indices_bytes"}:
        cached.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    repaired = prepare_track_a_rfecv_cache(
        artifacts,
        tmp_path,
        expected_indices=np.array([0, 2, 4]),
    )

    assert calls == [(48, 6), (48, 6)]
    np.testing.assert_array_equal(
        repaired.selected_features,
        artifacts.features[:, [0, 2, 4]],
    )
    repaired_manifest = json.loads(
        repaired.manifest_path.read_text(encoding="utf-8")
    )
    assert repaired_manifest["source_hashes"] == artifacts.source_hashes
    assert repaired_manifest["cache_key"] == repaired.cache_key
    assert repaired_manifest["rfecv_scope"] == "single_global_author_pass"
    assert "unexpected" not in repaired_manifest


def test_rfecv_cache_concurrent_build_publishes_once_without_partial_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_rars.dndt_dndf_experiment as experiment

    artifacts = _tiny_track_a_artifacts()
    worker_count = 8
    barrier = threading.Barrier(worker_count)
    calls: list[int] = []
    calls_lock = threading.Lock()

    class SlowRfecv:
        def __init__(self, **_: object) -> None:
            pass

        def fit(self, features: np.ndarray, labels: np.ndarray) -> SlowRfecv:
            del labels
            with calls_lock:
                calls.append(features.shape[0])
            time.sleep(0.15)
            self.support_ = np.array([True, False, True, False, True, False])
            return self

    monkeypatch.setattr(experiment, "RFECV", SlowRfecv)
    results: list[TrackAFeatureCache] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            results.append(
                prepare_track_a_rfecv_cache(
                    artifacts,
                    tmp_path,
                    expected_indices=np.array([0, 2, 4]),
                )
            )
        except BaseException as exc:  # pragma: no branch - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert calls == [48]
    assert len(results) == worker_count
    assert {result.cache_key for result in results} == {results[0].cache_key}
    for result in results[1:]:
        np.testing.assert_array_equal(
            results[0].selected_features,
            result.selected_features,
        )
    cache_dir = results[0].manifest_path.parent
    assert not list(cache_dir.glob("*.tmp"))
    assert (cache_dir / ".rfecv-cache.lock").is_file()
    assert len(list(cache_dir.glob("manifest.json"))) == 1
    assert len(list(cache_dir.glob("selected_features.npy"))) == 1
    assert len(list(cache_dir.glob("selected_indices.npy"))) == 1


def test_rfecv_cache_waiting_processes_revalidate_and_publish_once(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    process_count = 4
    start_path = tmp_path / "start"
    computation_path = tmp_path / "rfecv-computations.txt"
    ready_paths = [tmp_path / f"ready-{worker}" for worker in range(process_count)]
    result_paths = [tmp_path / f"result-{worker}.json" for worker in range(process_count)]
    processes = [
        context.Process(
            target=_rfecv_prepare_worker,
            args=(
                str(tmp_path),
                str(start_path),
                str(ready_paths[worker]),
                str(result_paths[worker]),
                str(computation_path),
            ),
        )
        for worker in range(process_count)
    ]
    for process in processes:
        process.start()
    try:
        deadline = time.monotonic() + 20.0
        while not all(path.exists() for path in ready_paths):
            if time.monotonic() >= deadline:
                raise TimeoutError("RFECV prepare workers did not become ready")
            time.sleep(0.01)
        start_path.write_text("start", encoding="ascii")
        for process in processes:
            process.join(timeout=20)

        assert not any(process.is_alive() for process in processes)
        assert [process.exitcode for process in processes] == [0] * process_count
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5)

    assert computation_path.read_text(encoding="ascii").splitlines() == ["48"]
    results = [json.loads(path.read_text(encoding="ascii")) for path in result_paths]
    assert len({json.dumps(result, sort_keys=True) for result in results}) == 1

    cache_dirs = list((tmp_path / "cache").glob("track_a_rfecv_*"))
    assert len(cache_dirs) == 1
    cache_dir = cache_dirs[0]
    assert {path.name for path in cache_dir.iterdir()} == {
        ".rfecv-cache.lock",
        "manifest.json",
        "selected_features.npy",
        "selected_indices.npy",
    }
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    selected_features_path = cache_dir / "selected_features.npy"
    selected_indices_path = cache_dir / "selected_indices.npy"
    assert checkpoint_sha256(selected_features_path) == manifest[
        "selected_features_sha256"
    ]
    assert checkpoint_sha256(selected_indices_path) == manifest[
        "selected_indices_sha256"
    ]
    selected_indices = np.load(selected_indices_path, allow_pickle=False)
    selected_features = np.load(selected_features_path, allow_pickle=False)
    np.testing.assert_array_equal(selected_indices, [0, 2, 4])
    np.testing.assert_array_equal(
        selected_features,
        _tiny_track_a_artifacts().features[:, [0, 2, 4]],
    )


@pytest.mark.parametrize("round_index", range(3))
def test_rfecv_os_lock_survives_repeated_process_contention(
    tmp_path: Path,
    round_index: int,
) -> None:
    context = multiprocessing.get_context("spawn")
    lock_path = tmp_path / f"round-{round_index}.lock"
    output_path = tmp_path / f"round-{round_index}.txt"
    start_path = tmp_path / f"round-{round_index}.start"
    process_count = 6
    iterations = 8
    processes = [
        context.Process(
            target=_rfecv_lock_contention_worker,
            args=(str(lock_path), str(output_path), str(start_path), worker, iterations),
        )
        for worker in range(process_count)
    ]
    for process in processes:
        process.start()
    start_path.write_text("start", encoding="ascii")
    for process in processes:
        process.join(timeout=30)

    assert not any(process.is_alive() for process in processes)
    assert [process.exitcode for process in processes] == [0] * process_count
    assert lock_path.is_file()
    assert not lock_path.with_suffix(".guard").exists()
    lines = output_path.read_text(encoding="ascii").splitlines()
    assert len(lines) == process_count * iterations
    assert len(set(lines)) == process_count * iterations


def test_rfecv_os_lock_is_released_immediately_when_owner_process_dies(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    lock_path = tmp_path / ".rfecv-cache.lock"
    acquired_path = tmp_path / "owner-acquired"
    owner = context.Process(
        target=_rfecv_lock_holder_worker,
        args=(str(lock_path), str(acquired_path)),
    )
    owner.start()
    deadline = time.monotonic() + 10.0
    while not acquired_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert acquired_path.exists()

    owner.kill()
    owner.join(timeout=10)
    assert not owner.is_alive()
    with _exclusive_rfecv_cache_lock(
        lock_path,
        timeout=1.0,
        retry_interval=0.005,
    ):
        assert lock_path.is_file()
    assert lock_path.is_file()


def test_rfecv_delayed_waiter_never_invalidates_current_owner(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    lock_path = tmp_path / ".rfecv-cache.lock"
    waiter_acquired_path = tmp_path / "waiter-acquired"

    with _exclusive_rfecv_cache_lock(
        lock_path,
        timeout=1.0,
        retry_interval=0.005,
    ):
        waiter = context.Process(
            target=_rfecv_lock_waiter_worker,
            args=(str(lock_path), str(waiter_acquired_path)),
        )
        waiter.start()
        time.sleep(0.5)
        assert waiter.is_alive()
        assert not waiter_acquired_path.exists()
        assert lock_path.is_file()

    waiter.join(timeout=10)
    assert not waiter.is_alive()
    assert waiter.exitcode == 0
    assert waiter_acquired_path.read_text(encoding="ascii") == "acquired"
    assert lock_path.is_file()


def test_rfecv_os_lock_timeout_is_bounded_and_actionable(tmp_path: Path) -> None:
    lock_path = tmp_path / ".rfecv-cache.lock"
    with _exclusive_rfecv_cache_lock(lock_path, timeout=1.0, retry_interval=0.005):
        started = time.monotonic()
        with pytest.raises(TimeoutError, match=r"RFECV cache lock.*0\.05.*lock"):
            with _exclusive_rfecv_cache_lock(
                lock_path,
                timeout=0.05,
                retry_interval=0.005,
            ):
                raise AssertionError("locked byte must not be acquired")
        assert time.monotonic() - started < 0.5


def test_rfecv_os_lock_closes_handle_when_initialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingHandle:
        closed = False

        def seek(self, *_: object) -> None:
            raise OSError("forced lock handle initialization failure")

        def close(self) -> None:
            self.closed = True

    handle = FailingHandle()
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: handle)

    with pytest.raises(OSError, match="initialization failure"):
        with _exclusive_rfecv_cache_lock(tmp_path / ".rfecv-cache.lock"):
            raise AssertionError("lock acquisition must not be reached")

    assert handle.closed


def test_rfecv_expected_index_assertion_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _tiny_track_a_artifacts()

    class WrongRfecv:
        def __init__(self, **_: object) -> None:
            pass

        def fit(self, features: np.ndarray, labels: np.ndarray) -> WrongRfecv:
            self.support_ = np.array([True, True, False, False, False, False])
            return self

    import covid_rars.dndt_dndf_experiment as experiment

    monkeypatch.setattr(experiment, "RFECV", WrongRfecv)
    with pytest.raises(RuntimeError, match="RFECV selected indices"):
        prepare_track_a_rfecv_cache(
            artifacts,
            tmp_path,
            expected_indices=np.array([0, 2, 4]),
        )
    assert len(EXPECTED_TRACK_A_SELECTED_INDICES) == 33


@pytest.mark.parametrize("configured_commit", (None, "f" * 40))
def test_track_a_production_loader_requires_immutable_author_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_commit: str | None,
) -> None:
    import covid_rars.dndt_dndf_experiment as experiment

    config = _tiny_track_a_config(tmp_path)
    config["author_repo"] = str(tmp_path / "author")
    if configured_commit is None:
        config.pop("author_commit")
    else:
        config["author_commit"] = configured_commit

    def unexpected_loader(*_: object, **__: object) -> AuthorTrackAArtifacts:
        raise AssertionError("artifact loader must not run for an invalid production pin")

    monkeypatch.setattr(experiment, "load_track_a_author_artifacts", unexpected_loader)
    with pytest.raises(ValueError, match="pinned author commit"):
        run_track_a(
            config,
            run_id="invalid-author-pin",
            code_revision="task4-pin-test",
            device="cpu",
        )


def test_track_a_modes_record_identity_threshold_and_training_order_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_rars.dndt_dndf_experiment as experiment

    artifacts = _tiny_track_a_artifacts()
    cache = _tiny_track_a_cache(tmp_path)
    identities: list[tuple[str, int, int]] = []
    original = experiment._create_track_a_model_optimizer

    def observing_factory(*args: object, **kwargs: object):
        model, optimizer = original(*args, **kwargs)
        identities.append((str(kwargs["mode"]), id(model), id(optimizer)))
        return model, optimizer

    monkeypatch.setattr(experiment, "_create_track_a_model_optimizer", observing_factory)
    result = run_track_a(
        _tiny_track_a_config(tmp_path),
        run_id="identity-contract",
        artifacts=artifacts,
        feature_cache=cache,
        fold_batch=(0, 1),
        modes=("author_behaviour_audit", "corrected_reference"),
        model_names=("dndt",),
        code_revision="task4-test",
        device="cpu",
    )

    author_ids = [(model, optimizer) for mode, model, optimizer in identities if mode == "author_behaviour_audit"]
    corrected_ids = [(model, optimizer) for mode, model, optimizer in identities if mode == "corrected_reference"]
    assert len(author_ids) == 1
    assert len(corrected_ids) == 2
    assert len(set(corrected_ids)) == 2
    rows = pd.DataFrame(result["metrics"])
    author = rows[rows["mode"] == "author_behaviour_audit"]
    corrected = rows[rows["mode"] == "corrected_reference"]
    assert author["threshold_source"].eq("test_balanced_accuracy_author_audit").all()
    assert author["threshold_selected_on_outer_test"].eq(True).all()
    assert author["model_reinitialized_per_fold"].eq(False).all()
    assert author["optimizer_reinitialized_per_fold"].eq(False).all()
    assert author["author_training_order"].eq("no_shuffle_repeated_dataset").all()
    assert author["author_randomness_unseeded"].eq(True).all()
    assert author["reconstruction_seed"].eq(42).all()
    assert corrected["threshold_source"].eq("inner_validation_balanced_accuracy").all()
    assert corrected["threshold_selected_on_outer_test"].eq(False).all()
    assert corrected["model_reinitialized_per_fold"].eq(True).all()
    assert corrected["outer_test_evaluation_count"].eq(1).all()
    predictions = pd.DataFrame(result["predictions"])
    author_predictions = predictions[
        predictions["mode"] == "author_behaviour_audit"
    ]
    corrected_predictions = predictions[
        predictions["mode"] == "corrected_reference"
    ]
    assert author_predictions["threshold_comparator"].eq("gt").all()
    assert corrected_predictions["threshold_comparator"].eq("ge").all()
    assert predictions["execution_backend"].eq("cpu").all()
    assert rows["execution_backend"].eq("cpu").all()
    run_dir = tmp_path / "runs" / "identity-contract"
    for receipt_path in run_dir.rglob("receipt.json"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["execution_backend"] == "cpu"
    author_state = _resolve_checkpoint_manifest(
        run_dir / "track_a" / "author_behaviour_audit" / "dndt" / "state",
        role="latest_recovery",
        map_location="cpu",
    )
    assert author_state.payload["execution_backend"] == "cpu"
    for checkpoint_dir in run_dir.glob("track_a/*/dndt/fold_*/checkpoints"):
        inference = _resolve_checkpoint_manifest(
            checkpoint_dir,
            role="best_inference",
            map_location="cpu",
        )
        assert inference.payload["execution_backend"] == "cpu"
    np.testing.assert_array_equal(
        author_predictions["predicted_label"].to_numpy(dtype=np.int64),
        (
            author_predictions["probability"].to_numpy(dtype=np.float64)
            > author_predictions["threshold"].to_numpy(dtype=np.float64)
        ).astype(np.int64),
    )
    np.testing.assert_array_equal(
        corrected_predictions["predicted_label"].to_numpy(dtype=np.int64),
        (
            corrected_predictions["probability"].to_numpy(dtype=np.float64)
            >= corrected_predictions["threshold"].to_numpy(dtype=np.float64)
        ).astype(np.int64),
    )


def test_track_a_resume_rejects_backend_mismatch_before_optimizer_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _tiny_track_a_artifacts()
    cache = _tiny_track_a_cache(tmp_path)
    config = _tiny_track_a_config(tmp_path)
    common = {
        "run_id": "backend-binding",
        "artifacts": artifacts,
        "feature_cache": cache,
        "fold_batch": (0,),
        "modes": ("corrected_reference",),
        "model_names": ("dndt",),
        "code_revision": "task4-backend-test",
        "device": "cpu",
    }
    with pytest.raises(PlannedInterruption):
        run_track_a(
            config,
            interrupt_after=("corrected_reference", "dndt", 0, 1),
            **common,
        )
    state_dir = (
        tmp_path
        / "runs"
        / "backend-binding"
        / "track_a"
        / "corrected_reference"
        / "dndt"
        / "fold_00"
        / "checkpoints"
    )
    recovered = _resolve_checkpoint_manifest(
        state_dir,
        role="latest_recovery",
        map_location="cpu",
    )
    assert recovered.payload["execution_backend"] == "cpu"
    tampered = dict(recovered.payload)
    fingerprint = dict(tampered["track_a_fingerprint"])
    fingerprint["execution_backend"] = "cuda:0"
    tampered["track_a_fingerprint"] = fingerprint
    tampered["execution_backend"] = "cuda:0"
    _publish_checkpoint_generation(
        tampered,
        state_dir,
        role="latest_recovery",
        epoch=int(tampered["completed_epoch"]),
    )

    def forbidden_optimizer_restore(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("optimizer state loaded before backend validation")

    monkeypatch.setattr(
        torch.optim.Adam,
        "load_state_dict",
        forbidden_optimizer_restore,
    )
    with pytest.raises(ValueError, match="execution_backend.*mismatch"):
        run_track_a(config, resume=True, **common)


def test_corrected_resume_at_patience_exhaustion_runs_no_additional_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_rars.dndt_dndf_experiment as experiment

    artifacts = _tiny_track_a_artifacts()
    training = artifacts.train_folds[0]
    validation = artifacts.test_folds[0]
    calls = {"full": 0, "resumed": 0}
    active = "full"
    original_train_epoch = experiment._track_a_train_epoch

    def count_epoch(*args: object, **kwargs: object) -> None:
        calls[active] += 1
        original_train_epoch(*args, **kwargs)

    monkeypatch.setattr(experiment, "_track_a_train_epoch", count_epoch)
    monkeypatch.setattr(
        experiment,
        "_safe_validation_metrics",
        lambda *_args: (0.5, 0.5),
    )
    common = {
        "model_config": _model_config(),
        "learning_rate": 0.01,
        "batch_size": 8,
        "max_epochs": 5,
        "patience": 1,
        "reconstruction_seed": 42,
        "feature_sha256": _sha("track-a-features"),
        "split_sha256": _sha("track-a-split"),
        "code_revision": "task4-patience-test",
        "device": torch.device("cpu"),
        "mode": "corrected_reference",
        "fold": 0,
    }
    full = _track_a_corrected_fit_no_scaler(
        artifacts.features[training],
        artifacts.author_labels[training],
        artifacts.features[validation],
        artifacts.covid_labels[validation],
        checkpoint_dir=tmp_path / "full",
        resume=False,
        interrupt_after_epoch=None,
        **common,
    )
    assert calls["full"] == 2

    active = "resumed"
    with pytest.raises(PlannedInterruption):
        _track_a_corrected_fit_no_scaler(
            artifacts.features[training],
            artifacts.author_labels[training],
            artifacts.features[validation],
            artifacts.covid_labels[validation],
            checkpoint_dir=tmp_path / "resumed",
            resume=False,
            interrupt_after_epoch=2,
            **common,
        )
    assert calls["resumed"] == 2
    resumed = _track_a_corrected_fit_no_scaler(
        artifacts.features[training],
        artifacts.author_labels[training],
        artifacts.features[validation],
        artifacts.covid_labels[validation],
        checkpoint_dir=tmp_path / "resumed",
        resume=True,
        interrupt_after_epoch=None,
        **common,
    )

    assert calls["resumed"] == 2
    assert resumed.best_epoch == full.best_epoch == 1
    assert resumed.history == full.history
    full_probability = _predict_probabilities(
        full.model,
        artifacts.features[validation],
        device=torch.device("cpu"),
        batch_size=8,
    )
    resumed_probability = _predict_probabilities(
        resumed.model,
        artifacts.features[validation],
        device=torch.device("cpu"),
        batch_size=8,
    )
    np.testing.assert_allclose(resumed_probability, full_probability, rtol=0.0, atol=0.0)
    latest = _resolve_checkpoint_manifest(
        tmp_path / "resumed",
        role="latest_recovery",
        map_location="cpu",
    )
    assert latest.payload["completed_epoch"] == 2
    assert latest.payload["no_improvement"] == 1


def test_author_fold_batch_resume_matches_uninterrupted_and_writes_atomic_receipts(
    tmp_path: Path,
) -> None:
    artifacts = _tiny_track_a_artifacts()
    cache = _tiny_track_a_cache(tmp_path)
    config = _tiny_track_a_config(tmp_path)
    common = dict(
        artifacts=artifacts,
        feature_cache=cache,
        fold_batch=(0, 1),
        modes=("author_behaviour_audit",),
        model_names=("dndt",),
        code_revision="task4-resume-test",
        device="cpu",
    )
    full = run_track_a(config, run_id="full", **common)

    with pytest.raises(PlannedInterruption):
        run_track_a(
            config,
            run_id="resumed",
            interrupt_after=("author_behaviour_audit", "dndt", 0, 1),
            **common,
        )
    resumed = run_track_a(config, run_id="resumed", resume=True, **common)
    completed_resume = run_track_a(config, run_id="resumed", resume=True, **common)

    full_predictions = pd.DataFrame(full["predictions"]).sort_values(
        ["model_name", "fold", "analysis_id"]
    ).reset_index(drop=True)
    resumed_predictions = pd.DataFrame(resumed["predictions"]).sort_values(
        ["model_name", "fold", "analysis_id"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        full_predictions.drop(columns="run_id"),
        resumed_predictions.drop(columns="run_id"),
    )
    assert len(completed_resume["predictions"]) == len(resumed["predictions"])
    receipt_paths = list((tmp_path / "runs" / "resumed").rglob("receipt.json"))
    assert len(receipt_paths) == 2
    for receipt_path in receipt_paths:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["status"] == "complete"
        assert receipt["feature_sha256"] == cache.selected_features_sha256
        assert receipt["author_commit"] == artifacts.author_commit
    assert not list((tmp_path / "runs" / "resumed").rglob("*.tmp"))

    prediction_path = receipt_paths[0].parent / "predictions.csv"
    prediction_path.write_text(
        prediction_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="authenticated receipt chain.*invalid"):
        run_track_a(config, run_id="resumed", resume=True, **common)


def test_author_receipt_crash_resume_never_repeats_outer_test_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_rars.dndt_dndf_experiment as experiment

    artifacts = _tiny_track_a_artifacts()
    full_root = tmp_path / "full"
    resumed_root = tmp_path / "resumed"
    full_cache = _tiny_track_a_cache(full_root)
    resumed_cache = _tiny_track_a_cache(resumed_root)
    calls = {"full": 0, "resumed": 0}
    threshold_calls = {"full": 0, "resumed": 0}
    active_run = "full"
    original = experiment._predict_track_a_outer_test
    original_threshold_sweep = experiment.author_threshold_sweep

    def counting_outer_test(*args: object, **kwargs: object) -> np.ndarray:
        calls[active_run] += 1
        return original(*args, **kwargs)

    def counting_threshold_sweep(*args: object, **kwargs: object) -> dict[str, float | int]:
        threshold_calls[active_run] += 1
        return original_threshold_sweep(*args, **kwargs)

    monkeypatch.setattr(
        experiment,
        "_predict_track_a_outer_test",
        counting_outer_test,
    )
    monkeypatch.setattr(
        experiment,
        "author_threshold_sweep",
        counting_threshold_sweep,
    )
    common = {
        "run_id": "receipt-window",
        "artifacts": artifacts,
        "fold_batch": (0,),
        "modes": ("author_behaviour_audit",),
        "model_names": ("dndt",),
        "code_revision": "task4-receipt-window-test",
        "device": "cpu",
    }
    full = run_track_a(
        _tiny_track_a_config(full_root),
        feature_cache=full_cache,
        **common,
    )
    assert calls["full"] == 1
    assert threshold_calls["full"] == 1

    active_run = "resumed"
    with pytest.raises(PlannedInterruption, match="after fold receipt"):
        run_track_a(
            _tiny_track_a_config(resumed_root),
            feature_cache=resumed_cache,
            interrupt_after_receipt=("author_behaviour_audit", "dndt", 0),
            **common,
        )
    assert calls["resumed"] == 1
    assert threshold_calls["resumed"] == 1
    interrupted_state = _resolve_checkpoint_manifest(
        resumed_root
        / "runs"
        / "receipt-window"
        / "track_a"
        / "author_behaviour_audit"
        / "dndt"
        / "state",
        role="latest_recovery",
        map_location="cpu",
    )
    assert interrupted_state.payload["phase"] == "trained"
    assert interrupted_state.payload["next_fold"] == 0

    resumed = run_track_a(
        _tiny_track_a_config(resumed_root),
        feature_cache=resumed_cache,
        resume=True,
        **common,
    )
    assert calls["resumed"] == 1
    assert threshold_calls["resumed"] == 1
    pd.testing.assert_frame_equal(
        pd.DataFrame(full["predictions"]),
        pd.DataFrame(resumed["predictions"]),
    )
    pd.testing.assert_frame_equal(
        pd.DataFrame(full["metrics"]),
        pd.DataFrame(resumed["metrics"]),
        check_like=True,
    )

    relative_receipt = (
        Path("track_a")
        / "author_behaviour_audit"
        / "dndt"
        / "fold_00"
        / "receipt.json"
    )
    full_run = full_root / "runs" / "receipt-window"
    resumed_run = resumed_root / "runs" / "receipt-window"
    assert (full_run / relative_receipt).read_bytes() == (
        resumed_run / relative_receipt
    ).read_bytes()
    full_state = _resolve_checkpoint_manifest(
        full_run / "track_a" / "author_behaviour_audit" / "dndt" / "state",
        role="latest_recovery",
        map_location="cpu",
    )
    resumed_state = _resolve_checkpoint_manifest(
        resumed_run / "track_a" / "author_behaviour_audit" / "dndt" / "state",
        role="latest_recovery",
        map_location="cpu",
    )
    assert full_state.payload["phase"] == resumed_state.payload["phase"] == "between_folds"
    assert full_state.payload["next_fold"] == resumed_state.payload["next_fold"] == 1
    assert full_state.descriptor["sha256"] == resumed_state.descriptor["sha256"]


def test_track_a_predictions_carry_complete_fold_provenance(tmp_path: Path) -> None:
    artifacts = _tiny_track_a_artifacts()
    cache = _tiny_track_a_cache(tmp_path)
    result = run_track_a(
        _tiny_track_a_config(tmp_path),
        run_id="prediction-provenance",
        artifacts=artifacts,
        feature_cache=cache,
        fold_batch=(0,),
        modes=("corrected_reference",),
        model_names=("dndt",),
        code_revision="task4-prediction-provenance-test",
        device="cpu",
    )

    predictions = pd.DataFrame(result["predictions"])
    expected = {
        "mode": "corrected_reference",
        "author_commit": artifacts.author_commit,
        "feature_indices_sha256": cache.selected_indices_sha256,
        "feature_cache_key": cache.cache_key,
        "code_revision": "task4-prediction-provenance-test",
    }
    for field, value in expected.items():
        assert field in predictions
        assert predictions[field].eq(value).all()


@pytest.mark.parametrize(
    "field",
    (
        "status",
        "run_id",
        "track",
        "mode",
        "protocol",
        "fold",
        "model_name",
        "author_commit",
        "feature_sha256",
        "feature_indices_sha256",
        "feature_cache_key",
        "configuration_sha256",
        "split_sha256",
        "code_revision",
        "execution_backend",
        "threshold_source",
        "threshold_comparator",
        "predictions_path",
        "metrics_path",
        "checkpoint_path",
        "checkpoint_sha256",
        "predictions_sha256",
        "metrics_sha256",
        "global_feature_selection_retained",
        "released_preprocessed_array_retained",
        "author_training_order",
        "author_randomness_unseeded",
        "reconstruction_seed",
    ),
)
def test_completed_fold_resume_rejects_each_tampered_receipt_provenance_field(
    tmp_path: Path,
    completed_corrected_track_a_template: tuple[
        Path,
        AuthorTrackAArtifacts,
        TrackAFeatureCache,
        dict[str, object],
    ],
    field: str,
) -> None:
    run_dir, artifacts, cache, config = _copied_corrected_track_a_case(
        tmp_path,
        completed_corrected_track_a_template,
    )
    receipt_path = (
        run_dir
        / "track_a"
        / "corrected_reference"
        / "dndt"
        / "fold_00"
        / "receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = 999 if field == "fold" else "tampered"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="authenticated fold"):
        _resume_copied_corrected_track_a(config, artifacts, cache)


@pytest.mark.parametrize("mutation", ("extra", "missing"))
def test_completed_fold_resume_rejects_nonexact_receipt_schema(
    tmp_path: Path,
    completed_corrected_track_a_template: tuple[
        Path,
        AuthorTrackAArtifacts,
        TrackAFeatureCache,
        dict[str, object],
    ],
    mutation: str,
) -> None:
    run_dir, artifacts, cache, config = _copied_corrected_track_a_case(
        tmp_path,
        completed_corrected_track_a_template,
    )
    receipt_path = (
        run_dir
        / "track_a"
        / "corrected_reference"
        / "dndt"
        / "fold_00"
        / "receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "extra":
        receipt["unexpected"] = "not-allowed"
    else:
        del receipt["author_training_order"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="authenticated fold receipt schema"):
        _resume_copied_corrected_track_a(config, artifacts, cache)


@pytest.mark.parametrize(
    ("target", "field"),
    (
        *(('prediction', field) for field in (
            "run_id",
            "track",
            "mode",
            "protocol",
            "fold",
            "model_name",
            "author_commit",
            "threshold_source",
            "configuration_sha256",
            "split_sha256",
            "feature_sha256",
            "feature_indices_sha256",
            "feature_cache_key",
            "code_revision",
            "execution_backend",
            "checkpoint_sha256",
            "threshold_comparator",
            "predicted_label",
        )),
        *(('metric', field) for field in (
            "run_id",
            "track",
            "protocol",
            "mode",
            "fold",
            "model_name",
            "author_commit",
            "feature_sha256",
            "feature_indices_sha256",
            "feature_cache_key",
            "configuration_sha256",
            "split_sha256",
            "code_revision",
            "execution_backend",
            "threshold_source",
            "threshold_comparator",
            "checkpoint_sha256",
        )),
    ),
)
def test_completed_fold_resume_rejects_tampered_output_identity_even_with_new_hash(
    tmp_path: Path,
    completed_corrected_track_a_template: tuple[
        Path,
        AuthorTrackAArtifacts,
        TrackAFeatureCache,
        dict[str, object],
    ],
    target: str,
    field: str,
) -> None:
    run_dir, artifacts, cache, config = _copied_corrected_track_a_case(
        tmp_path,
        completed_corrected_track_a_template,
    )
    fold_dir = (
        run_dir / "track_a" / "corrected_reference" / "dndt" / "fold_00"
    )
    receipt_path = fold_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if target == "prediction":
        output_path = fold_dir / "predictions.csv"
        output = pd.read_csv(output_path)
        output[field] = 999 if field == "fold" else "tampered"
        output.to_csv(output_path, index=False)
        receipt["predictions_sha256"] = checkpoint_sha256(output_path)
    else:
        output_path = fold_dir / "metrics.json"
        output = json.loads(output_path.read_text(encoding="utf-8"))
        output[field] = 999 if field == "fold" else "tampered"
        output_path.write_text(json.dumps(output), encoding="utf-8")
        receipt["metrics_sha256"] = checkpoint_sha256(output_path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="authenticated fold"):
        _resume_copied_corrected_track_a(config, artifacts, cache)


@pytest.mark.parametrize("checkpoint_change", ("missing", "tampered"))
def test_completed_fold_resume_rejects_missing_or_tampered_checkpoint_bytes(
    tmp_path: Path,
    completed_corrected_track_a_template: tuple[
        Path,
        AuthorTrackAArtifacts,
        TrackAFeatureCache,
        dict[str, object],
    ],
    checkpoint_change: str,
) -> None:
    run_dir, artifacts, cache, config = _copied_corrected_track_a_case(
        tmp_path,
        completed_corrected_track_a_template,
    )
    fold_dir = (
        run_dir / "track_a" / "corrected_reference" / "dndt" / "fold_00"
    )
    receipt = json.loads((fold_dir / "receipt.json").read_text(encoding="utf-8"))
    assert "checkpoint_path" in receipt
    checkpoint_path = fold_dir / str(receipt["checkpoint_path"])
    if checkpoint_change == "missing":
        checkpoint_path.unlink()
    else:
        checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="checkpoint"):
        _resume_copied_corrected_track_a(config, artifacts, cache)


@pytest.mark.parametrize(
    "change",
    ("delete_fold_0", "tamper_fold_0", "delete_fold_1", "tamper_fold_1"),
)
def test_author_later_batch_requires_contiguous_authenticated_receipt_chain(
    tmp_path: Path,
    change: str,
) -> None:
    artifacts = _tiny_track_a_artifacts()
    cache = _tiny_track_a_cache(tmp_path)
    config = _tiny_track_a_config(tmp_path)
    common = dict(
        artifacts=artifacts,
        feature_cache=cache,
        modes=("author_behaviour_audit",),
        model_names=("dndt",),
        code_revision="task4-chain-test",
        device="cpu",
    )
    run_track_a(
        config,
        run_id="chain",
        fold_batch=(0, 1),
        **common,
    )
    fold = 0 if change.endswith("_0") else 1
    receipt_path = (
        tmp_path
        / "runs"
        / "chain"
        / "track_a"
        / "author_behaviour_audit"
        / "dndt"
        / f"fold_{fold:02d}"
        / "receipt.json"
    )
    if change.startswith("delete"):
        receipt_path.unlink()
    else:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["model_name"] = "tampered"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="contiguous authenticated receipt chain"):
        run_track_a(
            config,
            run_id="chain",
            resume=True,
            fold_batch=(2,),
            **common,
        )


@pytest.mark.parametrize(
    "binding_field",
    (
        "completed_receipt_path",
        "completed_receipt_sha256",
        "completed_receipt_identity",
        "completed_receipt_identity_sha256",
    ),
)
def test_author_between_fold_state_is_bound_to_immediately_preceding_receipt(
    tmp_path: Path,
    binding_field: str,
) -> None:
    artifacts = _tiny_track_a_artifacts()
    cache = _tiny_track_a_cache(tmp_path)
    config = _tiny_track_a_config(tmp_path)
    common = dict(
        artifacts=artifacts,
        feature_cache=cache,
        modes=("author_behaviour_audit",),
        model_names=("dndt",),
        code_revision="task4-state-binding-test",
        device="cpu",
    )
    run_track_a(
        config,
        run_id="state-binding",
        fold_batch=(0, 1),
        **common,
    )
    state_dir = (
        tmp_path
        / "runs"
        / "state-binding"
        / "track_a"
        / "author_behaviour_audit"
        / "dndt"
        / "state"
    )
    resolved = _resolve_checkpoint_manifest(
        state_dir,
        role="latest_recovery",
        map_location="cpu",
    )
    state = dict(resolved.payload)
    assert "completed_receipt_identity" in state
    state[binding_field] = (
        {"fold": 999}
        if binding_field == "completed_receipt_identity"
        else "tampered"
    )
    _publish_checkpoint_generation(
        state,
        state_dir,
        role="latest_recovery",
        epoch=999,
    )

    with pytest.raises(ValueError, match="not bound"):
        run_track_a(
            config,
            run_id="state-binding",
            resume=True,
            fold_batch=(2,),
            **common,
        )


def test_corrected_outer_test_is_touched_once_after_validation_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_rars.dndt_dndf_experiment as experiment

    artifacts = _tiny_track_a_artifacts()
    cache = _tiny_track_a_cache(tmp_path)
    calls: list[np.ndarray] = []
    original = experiment._predict_track_a_outer_test

    def sentinel(
        model: torch.nn.Module,
        selected_features: np.ndarray,
        test_indices: np.ndarray,
        *,
        device: torch.device,
    ) -> np.ndarray:
        assert model.training is False
        calls.append(test_indices.copy())
        return original(
            model,
            selected_features,
            test_indices,
            device=device,
        )

    monkeypatch.setattr(experiment, "_predict_track_a_outer_test", sentinel)
    run_track_a(
        _tiny_track_a_config(tmp_path),
        run_id="outer-test-sentinel",
        artifacts=artifacts,
        feature_cache=cache,
        fold_batch=(0,),
        modes=("corrected_reference",),
        model_names=("dndt",),
        code_revision="task4-sentinel-test",
        device="cpu",
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], artifacts.test_folds[0])


def test_track_a_batched_aggregate_retains_all_authenticated_prior_folds(
    tmp_path: Path,
) -> None:
    artifacts = _tiny_track_a_artifacts()
    cache = _tiny_track_a_cache(tmp_path)
    config = _tiny_track_a_config(tmp_path)
    common = {
        "run_id": "batched-aggregate",
        "artifacts": artifacts,
        "feature_cache": cache,
        "modes": ("corrected_reference",),
        "model_names": ("dndt",),
        "code_revision": "task4-aggregate-test",
        "device": "cpu",
    }

    first = run_track_a(config, fold_batch=(0,), **common)
    assert first["status"] == "partial"
    assert first["completed_units"] == 1
    assert first["total_units"] == 3

    second = run_track_a(config, resume=True, fold_batch=(1,), **common)
    assert second["status"] == "partial"
    assert second["completed_units"] == 2
    assert second["total_units"] == 3
    second_metrics = pd.DataFrame(second["metrics"])
    assert second_metrics["fold"].tolist() == [0, 1]
    run_dir = tmp_path / "runs" / "batched-aggregate"
    persisted = pd.read_csv(run_dir / "track_a_metrics.csv")
    assert persisted["fold"].tolist() == [0, 1]
    manifest = json.loads(
        (run_dir / "track_a_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "partial"
    assert len(manifest["authenticated_receipts"]) == 2

    idempotent = run_track_a(config, resume=True, fold_batch=(1,), **common)
    assert pd.DataFrame(idempotent["metrics"])["fold"].tolist() == [0, 1]

    prior_receipt = (
        run_dir
        / "track_a"
        / "corrected_reference"
        / "dndt"
        / "fold_00"
        / "receipt.json"
    )
    receipt = json.loads(prior_receipt.read_text(encoding="utf-8"))
    receipt["fold"] = 99
    prior_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="authenticated fold"):
        run_track_a(config, resume=True, fold_batch=(1,), **common)
    prepare_track_a_rfecv_cache,
    run_track_a,
