from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _toy_metadata_and_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    preds = []
    labels = ["positive"] * 12 + ["negative"] * 12
    countries = ["India"] * 9 + ["Canada"] * 3 + ["India"] * 3 + ["Canada"] * 9
    for i, (label, country) in enumerate(zip(labels, countries)):
        pid = f"p{i}"
        rows.append(
            {
                "participant_id": pid,
                "recording_id": f"r{i}",
                "label_binary": label,
                "split": "test",
                "age": 55 if label == "positive" else 30,
                "gender": "male" if i % 2 else "female",
                "country": country,
                "recording_date": f"2020-06-{(i % 28) + 1:02d}",
                "duration_sec": 8.0 if label == "positive" else 5.0,
                "sample_rate_original": 48000 if i % 2 else 44100,
                "quality_flag": "ok",
            }
        )
        preds.append(
            {
                "participant_id": pid,
                "label_binary": label,
                "split": "test",
                "probability": 0.70 if label == "positive" else 0.30,
                "fusion_method": "quality_weighted_auprc",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(preds)


def test_bootstrap_controlled_metrics_returns_ci_rows_for_each_control_method():
    from covid_audio_btp.confounding_controlled_eval import bootstrap_confounding_controlled_metrics

    metadata, predictions = _toy_metadata_and_predictions()

    table = bootstrap_confounding_controlled_metrics(
        predictions,
        metadata,
        covariates=["country", "age", "gender"],
        group_columns=["fusion_method"],
        metrics=["auroc", "auprc", "balanced_accuracy"],
        n_bootstraps=50,
        random_state=7,
    )

    assert set(table["control_method"]) == {"unweighted", "ipw_label_propensity"}
    assert set(table["metric"]) == {"auroc", "auprc", "balanced_accuracy"}
    assert set(table["fusion_method"]) == {"quality_weighted_auprc"}
    assert (table["n_bootstraps"] == 50).all()
    assert (table["ci_low"] <= table["point"]).all()
    assert (table["point"] <= table["ci_high"]).all()
    assert table["effective_sample_size"].notna().all()


def test_confounding_controlled_bootstrap_cli_writes_output(tmp_path, monkeypatch):
    metadata, predictions = _toy_metadata_and_predictions()
    metadata_path = tmp_path / "metadata.csv"
    predictions_path = tmp_path / "predictions.csv"
    output_path = tmp_path / "bootstrap.csv"
    metadata.to_csv(metadata_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    script_path = Path(__file__).parents[1] / "scripts" / "31_bootstrap_confounding_controlled_metrics.py"
    spec = importlib.util.spec_from_file_location("bootstrap_confounding_controlled_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "31_bootstrap_confounding_controlled_metrics.py",
            "--predictions",
            str(predictions_path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(output_path),
            "--covariates",
            "country",
            "age",
            "gender",
            "--group-columns",
            "fusion_method",
            "--metrics",
            "auroc",
            "auprc",
            "--n-bootstraps",
            "25",
        ],
    )

    module.main()

    written = pd.read_csv(output_path)
    assert not written.empty
    assert set(written["metric"]) == {"auroc", "auprc"}
    assert set(written["control_method"]) == {"unweighted", "ipw_label_propensity"}
