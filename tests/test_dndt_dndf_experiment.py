from __future__ import annotations

import hashlib
import inspect
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
    aggregate_participant_probabilities,
    balance_training_rows,
    checkpoint_sha256,
    deterministic_batch_indices,
    fit_model,
    fit_preprocessor,
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
        def __init__(self, *, random_state: int) -> None:
            observed["seed"] = np.array([random_state])

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
    assert (tmp_path / "resumed" / "latest.pt").exists()
    assert (tmp_path / "resumed" / "best.pt").exists()
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
    assert checkpoint_sha256(tmp_path / "full-stop" / "latest.pt") == checkpoint_sha256(
        tmp_path / "resumed-stop" / "latest.pt"
    )
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
    (tmp_path / "best.pt").unlink()
    (tmp_path / "best.pt.sha256").unlink()

    result = _fit(tmp_path, resume=True)

    assert result.checkpoint_path.is_file()
    assert checkpoint_sha256(result.checkpoint_path) == (
        tmp_path / "best.pt.sha256"
    ).read_text(encoding="ascii").strip()


@pytest.mark.parametrize("mismatch", ["feature", "split", "config", "code"])
def test_resume_rejects_provenance_mismatch_before_checkpoint_mutation(
    tmp_path: Path, mismatch: str
) -> None:
    run_dir = tmp_path / mismatch
    with pytest.raises(PlannedInterruption):
        _fit(run_dir, interrupt_after_epoch=2)
    latest = run_dir / "latest.pt"
    digest_before = checkpoint_sha256(latest)
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
    assert checkpoint_sha256(latest) == digest_before


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

    conflicted = frame.copy()
    conflicted.loc[0, "participant_id"] = "p1"
    with pytest.raises(ValueError, match="conflicting labels"):
        aggregate_participant_probabilities(conflicted)

    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate recording"):
        aggregate_participant_probabilities(duplicated)


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
        tmp_path / "cuda-resume", train_config=config, device="cuda", resume=True
    )
    np.testing.assert_allclose(
        resumed.validation_probability, full.validation_probability, rtol=0.0, atol=1e-6
    )
    assert checkpoint_sha256(resumed.checkpoint_path) == checkpoint_sha256(
        full.checkpoint_path
    )
