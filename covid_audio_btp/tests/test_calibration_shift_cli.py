from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "33_calibration_under_shift.py"
    spec = importlib.util.spec_from_file_location("calibration_under_shift_cli", script_path)
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


def test_build_reports_from_prediction_paths_skips_non_prediction_csvs(tmp_path) -> None:
    module = _load_script_module()
    good = tmp_path / "predictions.csv"
    bad = tmp_path / "lfs_pointer.csv"
    good.write_text(
        "label_binary,probability,model_name\n"
        "negative,0.1,lr\n"
        "positive,0.9,lr\n"
        "negative,0.2,lr\n"
        "positive,0.8,lr\n"
    )
    bad.write_text("version https://git-lfs.github.com/spec/v1\n")

    summary, bins = module.build_reports_from_prediction_paths(
        [good, bad],
        group_columns=["model_name"],
        n_bins=2,
    )

    assert summary["prediction_source"].tolist() == ["predictions"]
    assert summary["model_name"].tolist() == ["lr"]
    assert bins["prediction_source"].unique().tolist() == ["predictions"]
    assert bins["n_samples"].sum() == 4
