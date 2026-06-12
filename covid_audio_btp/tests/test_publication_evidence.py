from __future__ import annotations

import pandas as pd


def _metric_row(**kwargs) -> dict[str, object]:
    row: dict[str, object] = {
        "model_name": "logistic_regression",
        "modality": "cough",
        "feature_strategy": "all",
        "auroc": 0.5,
        "auprc": 0.05,
        "balanced_accuracy": 0.5,
        "f1": 0.05,
        "n_samples": 100,
    }
    row.update(kwargs)
    return row


def test_publication_evidence_matrix_summarizes_core_claims() -> None:
    from covid_audio_btp.publication_evidence import build_publication_evidence_matrix

    tables = {
        "external_model_grid_metrics": pd.DataFrame(
            [
                _metric_row(model_name="logistic_regression", auroc=0.53, auprc=0.04, n_samples=8331),
                _metric_row(model_name="random_forest", auroc=0.49, auprc=0.03, n_samples=8331),
            ]
        ),
        "external_model_grid_beats_metrics": pd.DataFrame(
            [
                _metric_row(model_name="logistic_regression", feature_strategy="drop_high_shift", auroc=0.55, auprc=0.039, n_samples=8331),
                _metric_row(model_name="catboost", auroc=0.50, auprc=0.033, n_samples=8331),
            ]
        ),
        "coughvid_internal_beats_metrics": pd.DataFrame(
            [
                _metric_row(model_name="lightgbm", dataset="coughvid", auroc=0.76, auprc=0.14, n_samples=1667),
                _metric_row(model_name="logistic_regression", dataset="coughvid", auroc=0.70, auprc=0.10, n_samples=1667),
            ]
        ),
        "metadata_confounding_metrics": pd.DataFrame(
            [
                _metric_row(audit_model="full_safe_metadata", modality="metadata", auroc=0.96, auprc=0.93, n_samples=2862),
                _metric_row(audit_model="symptoms_only", modality="metadata", auroc=0.93, auprc=0.90, n_samples=2862),
            ]
        ),
        "confounding_controlled_audio_metrics": pd.DataFrame(
            [
                _metric_row(control_method="unweighted", fusion_method="quality_weighted_auprc", auroc=0.88, auprc=0.83, n_samples=318),
                _metric_row(control_method="ipw_label_propensity", fusion_method="quality_weighted_auprc", auroc=0.78, auprc=0.54, n_samples=318),
            ]
        ),
        "clinical_operating_points": pd.DataFrame(
            [
                {
                    "table_source": "quality_weighted_fusion_predictions",
                    "fusion_method": "quality_weighted_auprc",
                    "operating_constraint": "specificity>=0.900",
                    "threshold": 0.35,
                    "sensitivity": 0.70,
                    "specificity": 0.91,
                    "precision": 0.78,
                    "f1": 0.74,
                    "n_samples": 318,
                }
            ]
        ),
        "calibration_under_shift_summary": pd.DataFrame(
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
                },
                {
                    "prediction_source": "external_model_grid_beats_predictions",
                    "model_name": "logistic_regression",
                    "feature_strategy": "top_stable_50",
                    "n_samples": 8331,
                    "observed_prevalence": 0.034,
                    "mean_probability": 0.317,
                    "calibration_gap": 0.283,
                    "ece": 0.283,
                    "mce": 0.70,
                    "brier": 0.14,
                    "nll": 0.45,
                },
            ]
        ),
    }

    matrix = build_publication_evidence_matrix(tables)

    claim_ids = set(matrix["claim_id"])
    assert "external_transfer_mfcc_best" in claim_ids
    assert "external_transfer_beats_best" in claim_ids
    assert "coughvid_internal_beats_best" in claim_ids
    assert "metadata_confounding_full_safe_metadata" in claim_ids
    assert "confounding_controlled_audio_ipw" in claim_ids
    assert "clinical_fusion_specificity_0_900" in claim_ids
    assert "calibration_quality_weighted_fusion" in claim_ids
    assert "calibration_external_transfer_worst" in claim_ids

    beats = matrix[matrix["claim_id"] == "external_transfer_beats_best"].iloc[0]
    assert beats["primary_metric"] == "auroc"
    assert beats["primary_value"] == 0.55
    assert beats["evidence_direction"] == "cautionary"
    assert "BEATs" in beats["claim"]

    required_columns = {
        "claim_id",
        "claim",
        "evidence_type",
        "artifact",
        "comparison",
        "primary_metric",
        "primary_value",
        "secondary_metrics",
        "n_samples",
        "evidence_direction",
        "paper_use",
    }
    assert required_columns.issubset(matrix.columns)


def test_evidence_markdown_contains_claim_table() -> None:
    from covid_audio_btp.publication_evidence import evidence_matrix_to_markdown

    matrix = pd.DataFrame(
        [
            {
                "claim_id": "example_claim",
                "claim": "Example claim",
                "evidence_type": "example",
                "artifact": "example.csv",
                "comparison": "A vs B",
                "primary_metric": "auroc",
                "primary_value": 0.75,
                "secondary_metrics": "auprc=0.50",
                "n_samples": 100,
                "evidence_direction": "supportive",
                "paper_use": "Use in Results.",
            }
        ]
    )

    markdown = evidence_matrix_to_markdown(matrix)

    assert "# Publication Evidence Matrix" in markdown
    assert "| claim_id | claim | evidence_type |" in markdown
    assert "example_claim" in markdown
    assert "0.750" in markdown
