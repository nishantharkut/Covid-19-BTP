#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_audio_btp.clinical_operating_points import (
    DEFAULT_TARGET_SENSITIVITIES,
    DEFAULT_TARGET_SPECIFICITIES,
    build_clinical_operating_points,
)


DEFAULT_PREDICTION_PATHS = [
    Path("data/outputs/metrics/quality_weighted_fusion_predictions.csv"),
    Path("data/outputs/metrics/cross_dataset_predictions.csv"),
    Path("data/outputs/metrics/external_model_grid_predictions.csv"),
    Path("data/outputs/metrics/coughvid_internal_predictions.csv"),
]


DEFAULT_GROUP_COLUMNS = [
    "table_source",
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
    frame["table_source"] = path.stem
    return frame


def build_table_from_prediction_paths(
    prediction_paths: list[Path],
    group_columns: list[str] | None = None,
    target_specificities: list[float] | None = None,
    target_sensitivities: list[float] | None = None,
) -> pd.DataFrame:
    frames = [_read_prediction_csv(Path(path)) for path in prediction_paths if Path(path).exists()]
    predictions = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False) if frames else pd.DataFrame()
    if predictions.empty:
        return pd.DataFrame()
    groups = group_columns if group_columns is not None else DEFAULT_GROUP_COLUMNS
    if "table_source" in predictions.columns and "table_source" not in groups:
        groups = ["table_source", *groups]
    return build_clinical_operating_points(
        predictions,
        group_columns=groups,
        target_specificities=target_specificities,
        target_sensitivities=target_sensitivities,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-sensitivity and fixed-specificity operating point tables.")
    parser.add_argument("--predictions", nargs="*", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/tables/clinical_operating_points.csv"))
    parser.add_argument("--group-columns", nargs="*", default=DEFAULT_GROUP_COLUMNS)
    parser.add_argument("--target-specificities", nargs="*", type=float, default=DEFAULT_TARGET_SPECIFICITIES)
    parser.add_argument("--target-sensitivities", nargs="*", type=float, default=DEFAULT_TARGET_SENSITIVITIES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_paths = args.predictions if args.predictions is not None else default_prediction_paths()
    table = build_table_from_prediction_paths(
        prediction_paths,
        group_columns=args.group_columns,
        target_specificities=args.target_specificities,
        target_sensitivities=args.target_sensitivities,
    )
    if table.empty:
        checked = ", ".join(str(path) for path in prediction_paths)
        raise FileNotFoundError(f"No usable prediction CSVs found. Checked: {checked}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote clinical operating points: {args.output} ({len(table)} rows)")


if __name__ == "__main__":
    main()
