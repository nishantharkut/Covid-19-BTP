#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from covid_audio_btp.manifest import build_experiment_manifest


DEFAULT_ARTIFACTS = [
    Path("data/interim/coswara_index.csv"),
    Path("data/processed/metadata_clean.csv"),
    Path("data/processed/audio_quality.csv"),
    Path("data/processed/features_mfcc.csv"),
    Path("data/outputs/metrics/ml_baseline_metrics.csv"),
    Path("data/outputs/metrics/calibration_metrics.csv"),
    Path("data/outputs/metrics/fusion_metrics.csv"),
    Path("data/outputs/metrics/quality_weighted_fusion_metrics.csv"),
    Path("data/outputs/metrics/cross_dataset_metrics.csv"),
    Path("data/outputs/metrics/external_model_grid_metrics.csv"),
    Path("data/outputs/metrics/external_model_grid_bootstrap_ci.csv"),
    Path("data/outputs/metrics/coughvid_internal_metrics.csv"),
    Path("data/outputs/metrics/coughvid_internal_bootstrap_ci.csv"),
    Path("data/outputs/metrics/metadata_confounding_metrics.csv"),
    Path("data/outputs/metrics/metadata_confounding_predictions.csv"),
    Path("reports/tables/metadata_confounding_feature_importance.csv"),
    Path("reports/tables/metadata_confounding_group_summary.csv"),
    Path("data/outputs/metrics/confounding_controlled_audio_metrics.csv"),
    Path("data/outputs/metrics/confounding_controlled_audio_weights.csv"),
    Path("data/outputs/metrics/confounding_controlled_audio_bootstrap_ci.csv"),
    Path("reports/tables/confounding_controlled_balance.csv"),
    Path("reports/tables/calibration_under_shift_bins.csv"),
    Path("reports/tables/calibration_under_shift_summary.csv"),
    Path("reports/tables/clinical_operating_points.csv"),
    Path("reports/tables/publication_evidence_matrix.csv"),
    Path("reports/tables/publication_evidence_matrix.md"),
    Path("reports/final/BTP_PUBLICATION_RESULTS_REPORT.md"),
    Path("reports/final/BTP_PUBLICATION_RESULTS_SUMMARY.md"),
    Path("reports/tables/feature_shift_report.csv"),
    Path("reports/tables/feature_shift_summary.csv"),
    Path("reports/tables/paper_metric_table.csv"),
]


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def default_artifact_paths(project_root: Path | None = None) -> list[Path]:
    root = Path(".") if project_root is None else Path(project_root)
    base = [root / path for path in DEFAULT_ARTIFACTS]
    metrics_dir = root / "data" / "outputs" / "metrics"
    tables_dir = root / "reports" / "tables"
    patterns = [
        (metrics_dir, "external_model_grid_*_metrics.csv"),
        (metrics_dir, "external_model_grid_*_bootstrap_ci.csv"),
        (metrics_dir, "coughvid_internal_*_metrics.csv"),
        (metrics_dir, "coughvid_internal_*_bootstrap_ci.csv"),
        (tables_dir, "feature_shift_*_cough.csv"),
        (tables_dir, "feature_shift_*_summary.csv"),
    ]
    discovered: list[Path] = []
    for directory, pattern in patterns:
        discovered.extend(sorted(directory.glob(pattern)))
    return _dedupe_paths([*base, *discovered])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write reproducibility manifest with config, package versions, and artifact hashes.")
    parser.add_argument("--output", type=Path, default=Path("reports/experiment_manifest.json"))
    parser.add_argument("--artifacts", nargs="*", type=Path, default=None)
    parser.add_argument("--run-name", default="covid_audio_publication_run")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts if args.artifacts is not None else default_artifact_paths()
    manifest = build_experiment_manifest(
        config={"run_name": args.run_name, "seed": args.seed},
        artifact_paths=artifacts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote experiment manifest: {args.output}")


if __name__ == "__main__":
    main()
