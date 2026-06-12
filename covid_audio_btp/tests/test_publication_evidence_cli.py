from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "34_make_publication_evidence_matrix.py"
    spec = importlib.util.spec_from_file_location("publication_evidence_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_input_paths_include_publication_upgrade_artifacts(tmp_path) -> None:
    module = _load_script_module()

    paths = module.default_input_paths(project_root=tmp_path)

    assert paths["external_model_grid_metrics"] == tmp_path / "data" / "outputs" / "metrics" / "external_model_grid_metrics.csv"
    assert paths["clinical_operating_points"] == tmp_path / "reports" / "tables" / "clinical_operating_points.csv"
    assert paths["calibration_under_shift_summary"] == tmp_path / "reports" / "tables" / "calibration_under_shift_summary.csv"


def test_publication_evidence_cli_writes_csv_and_markdown(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    metrics = tmp_path / "data" / "outputs" / "metrics"
    tables = tmp_path / "reports" / "tables"
    metrics.mkdir(parents=True)
    tables.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "model_name": "logistic_regression",
                "modality": "cough",
                "feature_strategy": "all",
                "auroc": 0.53,
                "auprc": 0.04,
                "n_samples": 8331,
            }
        ]
    ).to_csv(metrics / "external_model_grid_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "audit_model": "full_safe_metadata",
                "modality": "metadata",
                "auroc": 0.96,
                "auprc": 0.93,
                "n_samples": 2862,
            }
        ]
    ).to_csv(metrics / "metadata_confounding_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "control_method": "ipw_label_propensity",
                "fusion_method": "quality_weighted_auprc",
                "auroc": 0.78,
                "auprc": 0.54,
                "n_samples": 318,
            }
        ]
    ).to_csv(metrics / "confounding_controlled_audio_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "table_source": "quality_weighted_fusion_predictions",
                "fusion_method": "quality_weighted_auprc",
                "operating_constraint": "specificity>=0.900",
                "sensitivity": 0.70,
                "specificity": 0.91,
                "precision": 0.78,
                "f1": 0.74,
                "n_samples": 318,
            }
        ]
    ).to_csv(tables / "clinical_operating_points.csv", index=False)
    pd.DataFrame(
        [
            {
                "prediction_source": "quality_weighted_fusion_predictions",
                "fusion_method": "quality_weighted_auprc",
                "n_samples": 318,
                "observed_prevalence": 0.324,
                "mean_probability": 0.331,
                "calibration_gap": 0.007,
                "ece": 0.15,
                "mce": 0.53,
                "brier": 0.19,
                "nll": 0.56,
            }
        ]
    ).to_csv(tables / "calibration_under_shift_summary.csv", index=False)

    output = tmp_path / "evidence.csv"
    markdown_output = tmp_path / "evidence.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "34_make_publication_evidence_matrix.py",
            "--project-root",
            str(tmp_path),
            "--output",
            str(output),
            "--markdown-output",
            str(markdown_output),
        ],
    )

    module.main()

    evidence = pd.read_csv(output)
    assert {"external_transfer_mfcc_best", "metadata_confounding_full_safe_metadata"}.issubset(set(evidence["claim_id"]))
    assert "# Publication Evidence Matrix" in markdown_output.read_text()
