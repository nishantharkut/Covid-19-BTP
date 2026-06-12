from __future__ import annotations

import time

import numpy as np
import pandas as pd

from covid_audio_btp.clinical_operating_points import (
    build_clinical_operating_points,
    operating_point_at_sensitivity,
    operating_point_at_specificity,
)


def test_operating_point_at_specificity_maximizes_sensitivity_under_constraint() -> None:
    frame = pd.DataFrame(
        {
            "label_binary": ["negative", "negative", "negative", "positive", "positive"],
            "probability": [0.10, 0.20, 0.70, 0.60, 0.80],
        }
    )

    row = operating_point_at_specificity(frame, target_specificity=2 / 3)

    assert row["operating_constraint"] == "specificity>=0.667"
    assert row["threshold"] == 0.60
    assert row["sensitivity"] == 1.0
    assert row["specificity"] == 2 / 3


def test_operating_point_at_sensitivity_maximizes_specificity_under_constraint() -> None:
    frame = pd.DataFrame(
        {
            "label_binary": ["negative", "negative", "negative", "positive", "positive"],
            "probability": [0.10, 0.20, 0.70, 0.60, 0.80],
        }
    )

    row = operating_point_at_sensitivity(frame, target_sensitivity=1.0)

    assert row["operating_constraint"] == "sensitivity>=1.000"
    assert row["threshold"] == 0.60
    assert row["sensitivity"] == 1.0
    assert row["specificity"] == 2 / 3


def test_build_clinical_operating_points_preserves_groups_and_counts() -> None:
    predictions = pd.DataFrame(
        {
            "label_binary": [
                "negative",
                "negative",
                "positive",
                "positive",
                "negative",
                "positive",
            ],
            "probability": [0.05, 0.40, 0.35, 0.90, 0.20, 0.80],
            "model_name": ["a", "a", "a", "a", "b", "b"],
            "feature_strategy": ["all", "all", "all", "all", "all", "all"],
        }
    )

    table = build_clinical_operating_points(
        predictions,
        group_columns=["model_name", "feature_strategy"],
        target_specificities=[0.5],
        target_sensitivities=[1.0],
    )

    assert set(table["model_name"]) == {"a", "b"}
    assert set(table["operating_constraint"]) == {
        "specificity>=0.500",
        "sensitivity>=1.000",
    }
    assert table.loc[table["model_name"].eq("a"), "n_positive"].unique().tolist() == [2]
    assert table.loc[table["model_name"].eq("a"), "n_negative"].unique().tolist() == [2]


def test_large_unique_probability_group_uses_fast_threshold_search() -> None:
    n = 6000
    labels = np.array(["negative"] * (n // 2) + ["positive"] * (n // 2))
    probabilities = np.linspace(0.001, 0.999, n)
    frame = pd.DataFrame({"label_binary": labels, "probability": probabilities})

    started = time.perf_counter()
    row = operating_point_at_specificity(frame, target_specificity=0.9)
    elapsed = time.perf_counter() - started

    assert row["specificity"] >= 0.9
    assert elapsed < 0.5
