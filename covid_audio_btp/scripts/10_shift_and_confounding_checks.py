#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_audio_btp.metrics import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate quality/subgroup/confounding metric tables.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/outputs/metrics/subgroup_metrics.csv"))
    return parser.parse_args()


def _age_bucket(value: object) -> str:
    try:
        age = float(value)
    except Exception:
        return "unknown"
    if age < 30:
        return "<30"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    return "60+"


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    metadata = pd.read_csv(args.metadata)
    cols = ["recording_id", "participant_id", "quality_flag", "age", "gender", "country"]
    merge_key = "recording_id" if "recording_id" in predictions.columns else "participant_id"
    merged = predictions.merge(metadata[[c for c in cols if c in metadata.columns]], on=merge_key, how="left")
    if "age" in merged.columns:
        merged["age_bucket"] = merged["age"].map(_age_bucket)
    group_columns = [c for c in ["quality_flag", "gender", "age_bucket"] if c in merged.columns]
    frames = []
    for group_col in group_columns:
        metrics = evaluate_predictions(merged.dropna(subset=[group_col]), group_columns=[group_col])
        metrics["analysis_group"] = group_col
        frames.append(metrics)
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote subgroup metrics: {args.output}")


if __name__ == "__main__":
    main()

