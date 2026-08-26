from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import sys
import pytest
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from covid_rars.dndt_dndf_evidence import (
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
    pd.DataFrame(final_rows).to_csv(
        track_b / "final_participant_predictions.csv", index=False
    )
    ladder_rows = _track_b_rows(
        stage="ladder",
        protocol="external_cough",
        model_name="dndf",
        split="validation",
        dataset="coswara",
        participant_prefix="v",
        probabilities=validation_probabilities,
        labels=labels,
    ) + _track_b_rows(
        stage="ladder",
        protocol="external_cough",
        model_name="dndf",
        split="external",
        dataset="coughvid",
        participant_prefix="x",
        probabilities=[0.55, 0.45, 0.50, 0.60],
        labels=labels,
    )
    pd.DataFrame(ladder_rows).to_csv(
        track_b / "ladder_participant_predictions.csv", index=False
    )
    (track_b / "selected_configurations.json").write_text(
        '{"cough":{"overall_winner":"dndf","selected_configuration_sha256":"'
        + "a" * 64
        + '"}}',
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
        "ladder_participant_predictions.csv",
        "selected_configurations.json",
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
