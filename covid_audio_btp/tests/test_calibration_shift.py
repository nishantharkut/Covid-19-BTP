from __future__ import annotations

import numpy as np
import pandas as pd

from covid_audio_btp.calibration_shift import (
    build_calibration_shift_report,
    calibration_bin_table,
    calibration_summary,
)


def test_calibration_bin_table_reports_weighted_gaps() -> None:
    predictions = pd.DataFrame(
        {
            "label_binary": ["negative", "negative", "positive", "positive"],
            "probability": [0.05, 0.15, 0.85, 0.95],
        }
    )

    bins = calibration_bin_table(predictions, n_bins=2)

    assert bins["n_samples"].tolist() == [2, 2]
    assert bins["observed_positive_rate"].tolist() == [0.0, 1.0]
    assert np.allclose(bins["mean_probability"], [0.10, 0.90])
    assert np.allclose(bins["abs_calibration_gap"], [0.10, 0.10])


def test_calibration_summary_uses_bin_weighted_ece_and_mce() -> None:
    predictions = pd.DataFrame(
        {
            "label_binary": ["negative", "negative", "positive", "positive"],
            "probability": [0.05, 0.15, 0.85, 0.95],
        }
    )

    summary = calibration_summary(predictions, n_bins=2)

    assert summary["n_samples"] == 4
    assert summary["observed_prevalence"] == 0.5
    assert summary["mean_probability"] == 0.5
    assert np.isclose(summary["ece"], 0.10)
    assert np.isclose(summary["mce"], 0.10)
    assert summary["nll"] < 0.2
    assert summary["brier"] < 0.03


def test_build_calibration_shift_report_preserves_groups() -> None:
    predictions = pd.DataFrame(
        {
            "label_binary": ["negative", "positive", "negative", "positive"],
            "probability": [0.1, 0.9, 0.4, 0.6],
            "prediction_source": ["internal", "internal", "external", "external"],
            "model_name": ["lr", "lr", "lr", "lr"],
        }
    )

    summary, bins = build_calibration_shift_report(
        predictions,
        group_columns=["prediction_source", "model_name"],
        n_bins=2,
    )

    assert set(summary["prediction_source"]) == {"internal", "external"}
    assert set(bins["prediction_source"]) == {"internal", "external"}
    assert summary["model_name"].unique().tolist() == ["lr"]
    assert bins.groupby("prediction_source")["n_samples"].sum().to_dict() == {
        "external": 2,
        "internal": 2,
    }
