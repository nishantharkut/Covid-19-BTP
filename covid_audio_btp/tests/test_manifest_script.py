from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "24_make_experiment_manifest.py"
    spec = importlib.util.spec_from_file_location("make_manifest_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_artifact_paths_discovers_representation_outputs(tmp_path) -> None:
    module = _load_script_module()
    metrics_dir = tmp_path / "data" / "outputs" / "metrics"
    tables_dir = tmp_path / "reports" / "tables"
    metrics_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)
    (metrics_dir / "external_model_grid_beats_metrics.csv").write_text("auroc\\n0.5\\n")
    (metrics_dir / "coughvid_internal_panns_cnn14_bootstrap_ci.csv").write_text("metric\\nauroc\\n")
    (tables_dir / "feature_shift_beats_cough.csv").write_text("feature\\nsome_dim\\n")

    artifacts = module.default_artifact_paths(project_root=tmp_path)

    assert tmp_path / "reports" / "tables" / "calibration_under_shift_bins.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "calibration_under_shift_summary.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "clinical_operating_points.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "publication_evidence_matrix.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "publication_evidence_matrix.md" in artifacts
    assert metrics_dir / "external_model_grid_beats_metrics.csv" in artifacts
    assert metrics_dir / "coughvid_internal_panns_cnn14_bootstrap_ci.csv" in artifacts
    assert tables_dir / "feature_shift_beats_cough.csv" in artifacts
