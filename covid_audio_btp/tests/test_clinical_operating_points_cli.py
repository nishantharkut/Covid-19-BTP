from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "32_clinical_operating_points.py"
    spec = importlib.util.spec_from_file_location("clinical_operating_points_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_prediction_paths_discovers_representation_predictions(tmp_path) -> None:
    module = _load_script_module()
    metrics_dir = tmp_path / "data" / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "external_model_grid_beats_predictions.csv").write_text("probability,label_binary\n0.2,negative\n")
    (metrics_dir / "coughvid_internal_panns_predictions.csv").write_text("probability,label_binary\n0.8,positive\n")

    paths = module.default_prediction_paths(metrics_dir=metrics_dir)

    assert metrics_dir / "external_model_grid_beats_predictions.csv" in paths
    assert metrics_dir / "coughvid_internal_panns_predictions.csv" in paths


def test_build_table_from_prediction_paths_skips_non_prediction_csvs(tmp_path) -> None:
    module = _load_script_module()
    good = tmp_path / "predictions.csv"
    bad = tmp_path / "lfs_pointer.csv"
    good.write_text(
        "label_binary,probability,model_name\n"
        "negative,0.1,lr\n"
        "negative,0.2,lr\n"
        "positive,0.7,lr\n"
        "positive,0.8,lr\n"
    )
    bad.write_text("version https://git-lfs.github.com/spec/v1\n")

    table = module.build_table_from_prediction_paths(
        [good, bad],
        group_columns=["model_name"],
        target_specificities=[0.5],
        target_sensitivities=[],
    )

    assert len(table) == 1
    assert table["table_source"].tolist() == ["predictions"]
    assert table["model_name"].tolist() == ["lr"]
    assert table["operating_constraint"].tolist() == ["specificity>=0.500"]
