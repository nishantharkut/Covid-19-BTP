#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_audio_btp.confounding_controlled_eval import (
    DEFAULT_CONFOUNDERS,
    bootstrap_confounding_controlled_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for confounding-controlled audio metrics."
    )
    parser.add_argument("--predictions", type=Path, default=Path("data/outputs/metrics/quality_weighted_fusion_predictions.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_clean.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/metrics/confounding_controlled_audio_bootstrap_ci.csv"),
    )
    parser.add_argument("--covariates", nargs="+", default=DEFAULT_CONFOUNDERS)
    parser.add_argument("--group-columns", nargs="*", default=None)
    parser.add_argument("--metrics", nargs="+", default=["auroc", "auprc", "balanced_accuracy", "f1", "brier", "ece"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _default_group_columns(predictions: pd.DataFrame) -> list[str]:
    candidates = ["fusion_method", "model_name", "modality", "feature_strategy", "dataset"]
    return [col for col in candidates if col in predictions.columns]


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    metadata = pd.read_csv(args.metadata)
    group_columns = args.group_columns if args.group_columns is not None else _default_group_columns(predictions)
    table = bootstrap_confounding_controlled_metrics(
        predictions,
        metadata,
        covariates=args.covariates,
        group_columns=group_columns,
        metrics=args.metrics,
        split=args.split,
        threshold=args.threshold,
        n_bootstraps=args.n_bootstraps,
        confidence=args.confidence,
        random_state=args.random_state,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote confounding-controlled bootstrap CIs: {args.output} ({len(table)} rows)")
    if not table.empty:
        display_cols = [
            *group_columns,
            "control_method",
            "metric",
            "point",
            "ci_low",
            "ci_high",
            "effective_sample_size",
            "n_bootstraps",
        ]
        print(table[[col for col in display_cols if col in table.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
