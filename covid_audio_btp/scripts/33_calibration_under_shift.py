#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_audio_btp.calibration_shift import DEFAULT_CALIBRATION_BINS, build_calibration_shift_report


DEFAULT_PREDICTION_PATHS = [
    Path("data/outputs/metrics/quality_weighted_fusion_predictions.csv"),
    Path("data/outputs/metrics/cross_dataset_predictions.csv"),
    Path("data/outputs/metrics/external_model_grid_predictions.csv"),
    Path("data/outputs/metrics/coughvid_internal_predictions.csv"),
]


DEFAULT_GROUP_COLUMNS = [
    "prediction_source",
    "model_name",
    "modality",
    "feature_strategy",
    "fusion_method",
    "dataset",
    "calibration_method",
    "split",
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


def default_prediction_paths(metrics_dir: Path = Path("data/outputs/metrics")) -> list[Path]:
    patterns = [
        "external_model_grid_*_predictions.csv",
        "coughvid_internal_*_predictions.csv",
    ]
    discovered: list[Path] = []
    for pattern in patterns:
        discovered.extend(sorted(metrics_dir.glob(pattern)))
    return _dedupe_paths([*DEFAULT_PREDICTION_PATHS, *discovered])


def _read_prediction_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    required = {"label_binary", "probability"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    frame["prediction_source"] = path.stem
    return frame


def build_reports_from_prediction_paths(
    prediction_paths: list[Path],
    group_columns: list[str] | None = None,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [_read_prediction_csv(Path(path)) for path in prediction_paths if Path(path).exists()]
    predictions = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False) if frames else pd.DataFrame()
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    groups = group_columns if group_columns is not None else DEFAULT_GROUP_COLUMNS
    if "prediction_source" in predictions.columns and "prediction_source" not in groups:
        groups = ["prediction_source", *groups]
    return build_calibration_shift_report(predictions, group_columns=groups, n_bins=n_bins)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build calibration-under-shift summary and reliability-bin tables.")
    parser.add_argument("--predictions", nargs="*", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=Path("reports/tables/calibration_under_shift_summary.csv"))
    parser.add_argument("--bins-output", type=Path, default=Path("reports/tables/calibration_under_shift_bins.csv"))
    parser.add_argument("--group-columns", nargs="*", default=DEFAULT_GROUP_COLUMNS)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_CALIBRATION_BINS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_paths = args.predictions if args.predictions is not None else default_prediction_paths()
    summary, bins = build_reports_from_prediction_paths(
        prediction_paths,
        group_columns=args.group_columns,
        n_bins=args.n_bins,
    )
    if summary.empty:
        checked = ", ".join(str(path) for path in prediction_paths)
        raise FileNotFoundError(f"No usable prediction CSVs found. Checked: {checked}")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.bins_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    bins.to_csv(args.bins_output, index=False)
    print(f"Wrote calibration-under-shift summary: {args.summary_output} ({len(summary)} rows)")
    print(f"Wrote calibration-under-shift bins: {args.bins_output} ({len(bins)} rows)")


if __name__ == "__main__":
    main()
