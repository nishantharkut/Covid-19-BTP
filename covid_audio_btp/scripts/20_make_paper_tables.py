#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_audio_btp.reporting import build_paper_metric_table, read_existing_csvs


DEFAULT_METRIC_PATHS = [
    Path("data/outputs/metrics/ml_baseline_metrics.csv"),
    Path("data/outputs/metrics/cnn_metrics.csv"),
    Path("data/outputs/metrics/calibration_metrics.csv"),
    Path("data/outputs/metrics/fusion_metrics.csv"),
    Path("data/outputs/metrics/quality_weighted_fusion_metrics.csv"),
    Path("data/outputs/metrics/metadata_baseline_metrics.csv"),
    Path("data/outputs/metrics/cross_dataset_metrics.csv"),
    Path("data/outputs/metrics/external_model_grid_metrics.csv"),
    Path("data/outputs/metrics/coughvid_internal_metrics.csv"),
    Path("data/outputs/metrics/metadata_confounding_metrics.csv"),
    Path("data/outputs/metrics/confounding_controlled_audio_metrics.csv"),
    Path("reports/tables/calibration_under_shift_summary.csv"),
]

DEFAULT_CI_PATHS = [
    Path("data/outputs/metrics/quality_weighted_fusion_bootstrap_ci.csv"),
    Path("data/outputs/metrics/cross_dataset_bootstrap_ci.csv"),
    Path("data/outputs/metrics/external_model_grid_bootstrap_ci.csv"),
    Path("data/outputs/metrics/coughvid_internal_bootstrap_ci.csv"),
    Path("data/outputs/metrics/confounding_controlled_audio_bootstrap_ci.csv"),
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


def default_metric_paths(metrics_dir: Path = Path("data/outputs/metrics")) -> list[Path]:
    representation_patterns = [
        "external_model_grid_*_metrics.csv",
        "coughvid_internal_*_metrics.csv",
    ]
    discovered: list[Path] = []
    for pattern in representation_patterns:
        discovered.extend(sorted(metrics_dir.glob(pattern)))
    return _dedupe_paths([*DEFAULT_METRIC_PATHS, *discovered])


def default_ci_paths(metrics_dir: Path = Path("data/outputs/metrics")) -> list[Path]:
    representation_patterns = [
        "external_model_grid_*_bootstrap_ci.csv",
        "coughvid_internal_*_bootstrap_ci.csv",
    ]
    discovered: list[Path] = []
    for pattern in representation_patterns:
        discovered.extend(sorted(metrics_dir.glob(pattern)))
    return _dedupe_paths([*DEFAULT_CI_PATHS, *discovered])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready metric tables from experiment CSVs.")
    parser.add_argument("--metrics", nargs="*", type=Path, default=None)
    parser.add_argument("--ci", nargs="*", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/tables/paper_metric_table.csv"))
    parser.add_argument("--raw-output", type=Path, default=Path("reports/tables/paper_metric_table_raw.csv"))
    return parser.parse_args()


def _group_columns(metrics: pd.DataFrame) -> list[str]:
    candidates = [
        "table_source",
        "model_name",
        "model",
        "modality",
        "feature_strategy",
        "fusion_method",
        "dataset",
        "split",
        "audit_model",
        "control_method",
        "prediction_source",
        "calibration_method",
    ]
    return [col for col in candidates if col in metrics.columns]


def main() -> None:
    args = parse_args()
    metric_paths = args.metrics if args.metrics is not None else default_metric_paths()
    ci_paths = args.ci if args.ci is not None else default_ci_paths()
    metrics = read_existing_csvs(metric_paths)
    ci_table = read_existing_csvs(ci_paths)
    if metrics.empty:
        checked = ", ".join(str(path) for path in metric_paths)
        raise FileNotFoundError(f"No metric CSVs found. Checked: {checked}")
    group_columns = _group_columns(metrics)
    paper = build_paper_metric_table(metrics, ci_table=ci_table, group_columns=group_columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.raw_output, index=False)
    paper.to_csv(args.output, index=False)
    print(f"Wrote paper metric table: {args.output} ({len(paper)} rows)")
    print(f"Wrote raw combined metric table: {args.raw_output} ({len(metrics)} rows)")


if __name__ == "__main__":
    main()
