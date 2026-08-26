from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import sys
import pytest
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from covid_rars.dndt_dndf_evidence import (
    _model_selection_table,
    _validate_complete_run_manifests,
    bootstrap_metric_delta,
    calibration_bin_table,
    complete_metric_bundle,
    decision_curve_analysis,
    generate_evidence,
    participant_bootstrap_ci,
    participant_bootstrap_indices,
    select_sensitivity_threshold,
)


def _prediction_frame(
    probabilities: list[float],
    labels: list[int],
    participants: list[str] | None = None,
) -> pd.DataFrame:
    if participants is None:
        participants = [f"p{index}" for index in range(len(labels))]
    return pd.DataFrame(
        {
            "participant_id": participants,
            "label_binary": labels,
            "probability": probabilities,
        }
    )


def test_complete_metric_bundle_matches_known_binary_case() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_prob = np.array([0.1, 0.8, 0.7, 0.4], dtype=np.float64)

    result = complete_metric_bundle(y_true, y_prob, threshold=0.5, n_bins=2)

    assert result["n_samples"] == 4
    assert result["tp"] == result["fp"] == result["tn"] == result["fn"] == 1
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["sensitivity"] == pytest.approx(0.5)
    assert result["specificity"] == pytest.approx(0.5)
    assert result["ppv"] == pytest.approx(0.5)
    assert result["npv"] == pytest.approx(0.5)
    assert result["prevalence"] == pytest.approx(0.5)
    assert result["brier"] == pytest.approx(0.275)
    assert result["ece"] == pytest.approx(0.25)
    assert result["mce"] == pytest.approx(0.25)
    assert result["nll"] == pytest.approx(log_loss(y_true, y_prob))
    assert result["auroc"] == pytest.approx(roc_auc_score(y_true, y_prob))
    assert result["auprc"] == pytest.approx(average_precision_score(y_true, y_prob))
    assert result["auprc_lift"] == pytest.approx(result["auprc"] / 0.5)


def test_complete_metric_bundle_honors_threshold_comparator() -> None:
    y_true = np.array([0, 1], dtype=np.int64)
    y_prob = np.array([0.5, 0.5], dtype=np.float64)

    greater_equal = complete_metric_bundle(
        y_true,
        y_prob,
        threshold=0.5,
        threshold_comparator="ge",
    )
    strict_greater = complete_metric_bundle(
        y_true,
        y_prob,
        threshold=0.5,
        threshold_comparator="gt",
    )

    assert greater_equal["tn"] == 0
    assert greater_equal["tp"] == 1
    assert strict_greater["tn"] == 1
    assert strict_greater["tp"] == 0
    with pytest.raises(ValueError, match="threshold_comparator"):
        complete_metric_bundle(
            y_true,
            y_prob,
            threshold=0.5,
            threshold_comparator="invalid",
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.2, float("nan")],
        [0.2, float("inf")],
        [-0.1, 0.2],
        [0.2, 1.1],
    ],
)
def test_complete_metric_bundle_rejects_invalid_probabilities(
    probabilities: list[float],
) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        complete_metric_bundle(np.array([0, 1]), np.array(probabilities), threshold=0.5)


def test_calibration_bins_are_left_closed_and_last_bin_is_right_closed() -> None:
    bins = calibration_bin_table(
        np.array([0, 1, 1]),
        np.array([0.0, 0.5, 1.0]),
        n_bins=2,
    )

    assert bins["count"].tolist() == [1, 2]
    assert bins["lower"].tolist() == [0.0, 0.5]
    assert bins["upper"].tolist() == [0.5, 1.0]


def test_participant_bootstrap_indices_resample_whole_clusters() -> None:
    participants = np.array(["a", "a", "b", "c", "c", "c"])
    sampled = participant_bootstrap_indices(
        participants,
        np.random.default_rng(7),
    )

    sampled_ids = participants[sampled]
    for participant in np.unique(sampled_ids):
        expected_cluster_size = int(np.sum(participants == participant))
        observed_count = int(np.sum(sampled_ids == participant))
        assert observed_count % expected_cluster_size == 0


def test_participant_bootstrap_ci_is_deterministic() -> None:
    frame = _prediction_frame(
        [0.05, 0.20, 0.35, 0.65, 0.80, 0.95],
        [0, 0, 0, 1, 1, 1],
    )

    first = participant_bootstrap_ci(
        frame,
        metric="auroc",
        n_bootstraps=100,
        seed=17,
    )
    second = participant_bootstrap_ci(
        frame,
        metric="auroc",
        n_bootstraps=100,
        seed=17,
    )

    assert first == second
    assert first["design"] == "participant_cluster"
    assert first["valid_replicates"] <= 100
    assert first["seed"] == 17


def test_paired_delta_rejects_nonidentical_participant_sets() -> None:
    left = _prediction_frame([0.1, 0.9], [0, 1], ["a", "b"])
    right = _prediction_frame([0.2, 0.8], [0, 1], ["a", "c"])

    with pytest.raises(ValueError, match="identical participant sets"):
        bootstrap_metric_delta(
            left,
            right,
            metric="auroc",
            paired=True,
            n_bootstraps=20,
            seed=3,
        )


def test_unpaired_delta_uses_independent_two_sample_design() -> None:
    source = _prediction_frame(
        [0.05, 0.15, 0.85, 0.95],
        [0, 0, 1, 1],
        ["s0", "s1", "s2", "s3"],
    )
    target = _prediction_frame(
        [0.45, 0.55, 0.40, 0.60],
        [0, 0, 1, 1],
        ["t0", "t1", "t2", "t3"],
    )

    result = bootstrap_metric_delta(
        source,
        target,
        metric="auroc",
        paired=False,
        n_bootstraps=100,
        seed=11,
    )

    assert result["design"] == "independent_two_sample_participant_cluster"
    assert result["point"] == pytest.approx(0.5)
    assert result["valid_replicates"] > 0


def test_fixed_sensitivity_threshold_is_selected_on_validation_only() -> None:
    validation = _prediction_frame(
        [0.05, 0.10, 0.20, 0.70, 0.80, 0.90],
        [0, 0, 0, 1, 1, 1],
    )
    threshold = select_sensitivity_threshold(
        validation["label_binary"].to_numpy(),
        validation["probability"].to_numpy(),
        minimum_sensitivity=0.90,
    )

    assert threshold == pytest.approx(0.70)


def test_decision_curve_reports_model_and_reference_strategies() -> None:
    frame = _prediction_frame([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
    thresholds = np.array([0.1, 0.5])

    result = decision_curve_analysis(frame, thresholds=thresholds)

    assert result["threshold_probability"].tolist() == [0.1, 0.5]
    assert set(
        [
            "model_net_benefit",
            "treat_all_net_benefit",
            "treat_none_net_benefit",
        ]
    ).issubset(result.columns)
    assert result["treat_none_net_benefit"].eq(0.0).all()


def _track_b_rows(
    *,
    stage: str,
    protocol: str,
    model_name: str,
    split: str,
    dataset: str,
    participant_prefix: str,
    probabilities: list[float],
    labels: list[int],
) -> list[dict[str, object]]:
    return [
        {
            "run_id": "fixture-run",
            "track": "B",
            "stage": stage,
            "protocol": protocol,
            "seed": 42,
            "dataset": dataset,
            "split": split,
            "model_name": model_name,
            "modality": "cough",
            "participant_id": f"{participant_prefix}{index}",
            "label_binary": label,
            "probability": probability,
            "threshold": 0.5,
            "threshold_source": "source_validation_balanced_accuracy",
            "selected_configuration_sha256": "a" * 64,
            "configuration_sha256": "b" * 64,
            "feature_sha256": "c" * 64,
            "split_sha256": "d" * 64,
            "checkpoint_sha256": "e" * 64,
            "code_revision": "f" * 40,
            "execution_backend": "cpu",
        }
        for index, (probability, label) in enumerate(zip(probabilities, labels))
    ]


_REQUIRED_MANIFEST_FIXTURES = (
    (Path("track_a_manifest.json"), "A", None),
    (Path("track_b/candidates_manifest.json"), "B", "candidates"),
    (Path("track_b/final_manifest.json"), "B", "final"),
    (Path("track_b/fusion_manifest.json"), "B", "fusion"),
    (
        Path("track_b/prespecified_ladder_v2_manifest.json"),
        "B",
        "prespecified_ladder_v2",
    ),
    (Path("track_b/shuffle_manifest.json"), "B", "shuffle"),
)


def _write_complete_manifest_fixtures(run_dir: Path) -> list[Path]:
    receipt_dir = run_dir / "fixture_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    selected_path = run_dir / "track_b" / "selected_configurations.json"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    if not selected_path.is_file():
        selected_path.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "modalities": {
                        "cough": {
                            "overall_winner": "dndf",
                            "selected_configuration_sha256": "a" * 64,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
    receipts: list[Path] = []
    for index, (relative_path, track, stage) in enumerate(
        _REQUIRED_MANIFEST_FIXTURES
    ):
        receipt_path = receipt_dir / f"receipt_{index}.json"
        receipt_path.write_text(
            json.dumps(
                {
                    "run_id": run_dir.name,
                    "track": track,
                    "stage": stage,
                    "status": "complete",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        payload: dict[str, object] = {
            "run_id": run_dir.name,
            "track": track,
            "status": "complete",
            "completed_units": 1,
            "total_units": 1,
            "authenticated_receipts": [
                {"path": str(receipt_path), "sha256": receipt_sha}
            ],
        }
        if track == "A":
            payload["required_design"] = {
                "modes": [
                    "author_behaviour_audit",
                    "fresh_fold_author_protocol",
                    "corrected_reference",
                ],
                "model_names": ["dndf"],
                "folds": list(range(10)),
            }
            payload["design_contract_complete"] = True
        if stage is not None:
            payload["stage"] = stage
        if stage == "shuffle":
            payload["control_scope"] = "single_permutation_fit_stage_sanity_check"
            payload["permutation_seed"] = 42
        artifact_names = {
            "track_a": ("predictions", "metrics", "rfecv_outer_test_exposure"),
            "final": ("recording_predictions", "participant_predictions", "metrics"),
            "prespecified_ladder_v2": (
                "recording_predictions",
                "participant_predictions",
                "metrics",
                "feature_lineage",
            ),
            "shuffle": ("recording_predictions", "participant_predictions", "metrics"),
            "fusion": ("participant_predictions", "metrics", "weights"),
            "candidates": ("selected_configurations",),
        }.get("track_a" if track == "A" else str(stage), ())
        if artifact_names:
            artifacts: dict[str, dict[str, str]] = {}
            for artifact_name in artifact_names:
                suffix = "csv"
                preferred = (
                    selected_path
                    if stage == "candidates" and artifact_name == "selected_configurations"
                    else run_dir / "track_b" / f"{stage}_{artifact_name}.{suffix}"
                )
                artifact_path = (
                    preferred
                    if preferred.is_file()
                    else receipt_dir / f"{stage}_{artifact_name}.bin"
                )
                if not artifact_path.exists():
                    artifact_path.write_bytes(f"{stage}:{artifact_name}".encode("ascii"))
                artifacts[artifact_name] = {
                    "path": str(artifact_path),
                    "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                }
            payload["artifacts"] = artifacts
        manifest_path = run_dir / relative_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        receipts.append(receipt_path)
    return receipts


def test_model_selection_table_reads_nested_modality_payload(tmp_path: Path) -> None:
    selected_path = tmp_path / "selected_configurations.json"
    selected_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "modalities": {
                    modality: {
                        "overall_winner": "dndf",
                        "selected_configuration_sha256": character * 64,
                    }
                    for modality, character in (
                        ("breath", "a"),
                        ("cough", "b"),
                        ("speech", "c"),
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    table = _model_selection_table(selected_path)

    assert table["modality"].tolist() == ["breath", "cough", "speech"]
    assert "format_version" not in set(table["modality"])
    assert "modalities" not in set(table["modality"])


def test_generate_evidence_rejects_partial_required_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "partial-run"
    track_b = run_dir / "track_b"
    track_b.mkdir(parents=True)
    frame = pd.DataFrame(
        _track_b_rows(
            stage="final",
            protocol="existing",
            model_name="dndf",
            split="test",
            dataset="coswara",
            participant_prefix="p",
            probabilities=[0.1, 0.2, 0.8, 0.9],
            labels=[0, 0, 1, 1],
        )
    )
    frame["run_id"] = run_dir.name
    frame.to_csv(track_b / "final_participant_predictions.csv", index=False)
    (run_dir / "track_a_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "track": "A",
                "status": "partial",
                "completed_units": 1,
                "total_units": 40,
                "authenticated_receipts": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="track_a_manifest.*complete"):
        generate_evidence(run_dir, n_bootstraps=5, seed=1)


def test_complete_manifest_validation_requires_authenticated_candidate_selection(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "candidate-artifact-required"
    run_dir.mkdir()
    _write_complete_manifest_fixtures(run_dir)
    manifest_path = run_dir / "track_b" / "candidates_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("artifacts", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="candidates_manifest.*artifact schema"):
        _validate_complete_run_manifests(run_dir)


def test_complete_manifest_validation_rejects_reduced_track_a_design(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "reduced-track-a"
    run_dir.mkdir()
    _write_complete_manifest_fixtures(run_dir)
    manifest_path = run_dir / "track_a_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["design_contract_complete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Track A required design contract"):
        _validate_complete_run_manifests(run_dir)


def test_complete_manifest_validation_requires_precise_shuffle_scope(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "shuffle-scope"
    run_dir.mkdir()
    _write_complete_manifest_fixtures(run_dir)
    manifest_path = run_dir / "track_b" / "shuffle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["control_scope"] = "permutation_test"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="shuffle control scope"):
        _validate_complete_run_manifests(run_dir)


def test_complete_manifest_validation_requires_ladder_feature_lineage(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "ladder-feature-lineage"
    run_dir.mkdir()
    _write_complete_manifest_fixtures(run_dir)
    manifest_path = (
        run_dir / "track_b" / "prespecified_ladder_v2_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].pop("feature_lineage", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError, match="prespecified_ladder_v2_manifest.*artifact schema"
    ):
        _validate_complete_run_manifests(run_dir)


def test_generate_evidence_rejects_tampered_authenticated_receipt(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "tampered-run"
    run_dir.mkdir()
    receipts = _write_complete_manifest_fixtures(run_dir)
    receipts[2].write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="authenticated receipt.*tampered"):
        generate_evidence(run_dir, n_bootstraps=5, seed=1)


def test_generate_evidence_rejects_tampered_track_a_aggregate_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "track-a-aggregate-tamper"
    run_dir.mkdir()
    _write_complete_manifest_fixtures(run_dir)
    manifest = json.loads(
        (run_dir / "track_a_manifest.json").read_text(encoding="utf-8")
    )
    descriptor = manifest["artifacts"]["rfecv_outer_test_exposure"]
    artifact_path = Path(descriptor["path"])
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="track_a_manifest.*artifact.*tampered"):
        generate_evidence(run_dir, n_bootstraps=5, seed=1)


def test_generate_evidence_rejects_tampered_aggregate_prediction(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "aggregate-tamper-run"
    track_b = run_dir / "track_b"
    track_b.mkdir(parents=True)
    prediction_path = track_b / "final_participant_predictions.csv"
    frame = pd.DataFrame(
        _track_b_rows(
            stage="final",
            protocol="existing",
            model_name="dndf",
            split="test",
            dataset="coswara",
            participant_prefix="p",
            probabilities=[0.1, 0.2, 0.8, 0.9],
            labels=[0, 0, 1, 1],
        )
    )
    frame["run_id"] = run_dir.name
    frame.to_csv(prediction_path, index=False)
    _write_complete_manifest_fixtures(run_dir)
    prediction_path.write_bytes(prediction_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="manifest artifact.*tampered"):
        generate_evidence(run_dir, n_bootstraps=5, seed=1)


def test_generate_evidence_writes_complete_tables_and_recomputable_metrics(
    tmp_path,
) -> None:
    run_dir = tmp_path / "fixture-run"
    track_b = run_dir / "track_b"
    track_b.mkdir(parents=True)
    validation_probabilities = [0.05, 0.20, 0.70, 0.90]
    test_probabilities = [0.10, 0.60, 0.80, 0.40]
    labels = [0, 0, 1, 1]
    final_rows = []
    for model_name, offset in (("dndf", 0.0), ("dndt", -0.02)):
        final_rows.extend(
            _track_b_rows(
                stage="final",
                protocol="existing",
                model_name=model_name,
                split="validation",
                dataset="coswara",
                participant_prefix="v",
                probabilities=[value + offset for value in validation_probabilities],
                labels=labels,
            )
        )
        final_rows.extend(
            _track_b_rows(
                stage="final",
                protocol="existing",
                model_name=model_name,
                split="test",
                dataset="coswara",
                participant_prefix="p",
                probabilities=[value + offset for value in test_probabilities],
                labels=labels,
            )
        )
    final_frame = pd.DataFrame(final_rows)
    final_frame["label_binary"] = final_frame["label_binary"].map(
        {0: "negative", 1: "positive"}
    )
    final_frame.to_csv(track_b / "final_participant_predictions.csv", index=False)
    ladder_rows = _track_b_rows(
        stage="prespecified_ladder_v2",
        protocol="external_cough",
        model_name="dndf",
        split="validation",
        dataset="coswara",
        participant_prefix="v",
        probabilities=validation_probabilities,
        labels=labels,
    ) + _track_b_rows(
        stage="prespecified_ladder_v2",
        protocol="external_cough",
        model_name="dndf",
        split="external",
        dataset="coughvid",
        participant_prefix="x",
        probabilities=[0.55, 0.45, 0.50, 0.60],
        labels=labels,
    )
    ladder_frame = pd.DataFrame(ladder_rows)
    ladder_frame["label_binary"] = ladder_frame["label_binary"].map(
        {0: "negative", 1: "positive"}
    )
    ladder_frame.to_csv(
        track_b / "prespecified_ladder_v2_participant_predictions.csv", index=False
    )
    _write_complete_manifest_fixtures(run_dir)
    (track_b / "selected_configurations.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "modalities": {
                    "cough": {
                        "overall_winner": "dndf",
                        "selected_configuration_sha256": "a" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = generate_evidence(
        run_dir,
        n_bootstraps=50,
        seed=23,
    )

    expected_outputs = {
        "metrics.csv",
        "final_summary.csv",
        "bootstrap_ci.csv",
        "paired_comparisons.csv",
        "external_deltas.csv",
        "fixed_sensitivity_operating_points.csv",
        "decision_curve.csv",
        "model_selection.csv",
        "fusion_weights.csv",
        "run_manifest.json",
    }
    evidence_dir = run_dir / "evidence"
    assert expected_outputs == {path.name for path in evidence_dir.iterdir()}
    metrics = pd.read_csv(evidence_dir / "metrics.csv")
    selected = metrics[
        metrics["stage"].eq("final")
        & metrics["protocol"].eq("existing")
        & metrics["model_name"].eq("dndf")
        & metrics["split"].eq("test")
    ].iloc[0]
    assert selected["auroc"] == pytest.approx(
        roc_auc_score(labels, test_probabilities), abs=1e-12
    )
    assert selected["auprc"] == pytest.approx(
        average_precision_score(labels, test_probabilities), abs=1e-12
    )
    assert int(selected["tp"]) == 1
    assert int(selected["fp"]) == 1
    deltas = pd.read_csv(evidence_dir / "external_deltas.csv")
    assert len(deltas) == 4
    assert set(deltas["metric"]) == {"auroc", "auprc", "brier", "ece"}
    assert manifest["run_id"] == "fixture-run"
    assert manifest["n_bootstraps"] == 50
    assert set(manifest["inputs"]) == {
        "final_participant_predictions.csv",
        "prespecified_ladder_v2_participant_predictions.csv",
        "selected_configurations.json",
        "track_a_manifest.json",
        "candidates_manifest.json",
        "final_manifest.json",
        "fusion_manifest.json",
        "prespecified_ladder_v2_manifest.json",
        "shuffle_manifest.json",
    }


def test_evidence_cli_generates_outputs_from_configured_run_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "cli-run"
    track_b = run_dir / "track_b"
    track_b.mkdir(parents=True)
    rows = _track_b_rows(
        stage="final",
        protocol="existing",
        model_name="dndf",
        split="validation",
        dataset="coswara",
        participant_prefix="v",
        probabilities=[0.1, 0.2, 0.8, 0.9],
        labels=[0, 0, 1, 1],
    ) + _track_b_rows(
        stage="final",
        protocol="existing",
        model_name="dndf",
        split="test",
        dataset="coswara",
        participant_prefix="t",
        probabilities=[0.2, 0.4, 0.6, 0.8],
        labels=[0, 0, 1, 1],
    )
    frame = pd.DataFrame(rows)
    frame["run_id"] = "cli-run"
    frame.to_csv(track_b / "final_participant_predictions.csv", index=False)
    _write_complete_manifest_fixtures(run_dir)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"run_root": str(tmp_path)}), encoding="utf-8")
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "82_make_dndt_dndf_evidence.py"),
            "--config",
            str(config),
            "--run-id",
            "cli-run",
            "--n-bootstraps",
            "20",
            "--seed",
            "5",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "complete"
    assert payload["run_id"] == "cli-run"
    assert (run_dir / "evidence" / "run_manifest.json").is_file()
