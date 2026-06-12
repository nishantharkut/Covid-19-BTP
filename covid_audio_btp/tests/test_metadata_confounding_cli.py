from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "29_metadata_confounding_audit.py"
    spec = importlib.util.spec_from_file_location("metadata_confounding_audit_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata_rows() -> pd.DataFrame:
    rows = []
    labels = ["negative", "positive"] * 18
    splits = ["train"] * 24 + ["validation"] * 6 + ["test"] * 6
    for i, (label, split) in enumerate(zip(labels, splits)):
        positive = label == "positive"
        rows.append(
            {
                "recording_id": f"rec_{i}",
                "participant_id": f"p_{i}",
                "dataset": "coswara",
                "modality": "cough",
                "label_binary": label,
                "split": split,
                "age": 52 if positive else 24,
                "gender": "male" if i % 2 else "female",
                "country": "India" if i % 3 else "United States",
                "recording_date": f"2020-05-{(i % 28) + 1:02d}",
                "symptoms_json": '{"cough": true}' if positive else '{"cough": false}',
                "comorbidities_json": '{"asthma": true}' if positive else '{"asthma": false}',
                "duration_sec": 8.0 if positive else 4.0,
                "sample_rate_original": 48000,
                "quality_flag": "ok",
            }
        )
    return pd.DataFrame(rows)


def test_metadata_confounding_cli_writes_all_outputs(tmp_path, monkeypatch):
    module = _load_script_module()
    metadata_path = tmp_path / "metadata.csv"
    metrics_path = tmp_path / "metrics.csv"
    predictions_path = tmp_path / "predictions.csv"
    importance_path = tmp_path / "importance.csv"
    group_summary_path = tmp_path / "group_summary.csv"
    _metadata_rows().to_csv(metadata_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "29_metadata_confounding_audit.py",
            "--metadata",
            str(metadata_path),
            "--metrics-output",
            str(metrics_path),
            "--predictions-output",
            str(predictions_path),
            "--feature-importance-output",
            str(importance_path),
            "--group-summary-output",
            str(group_summary_path),
        ],
    )

    module.main()

    assert set(pd.read_csv(metrics_path)["audit_model"]) == {
        "symptoms_only",
        "demographic_protocol_only",
        "full_safe_metadata",
    }
    assert not pd.read_csv(predictions_path).empty
    assert not pd.read_csv(importance_path).empty
    assert not pd.read_csv(group_summary_path).empty
