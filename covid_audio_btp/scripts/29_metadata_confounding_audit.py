#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_audio_btp.metadata_confounding import FEATURE_SETS, run_metadata_confounding_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit how much COVID-label signal is recoverable from metadata, symptoms, and protocol variables."
    )
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_clean.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/metadata_confounding_metrics.csv"))
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/metadata_confounding_predictions.csv"),
    )
    parser.add_argument(
        "--feature-importance-output",
        type=Path,
        default=Path("reports/tables/metadata_confounding_feature_importance.csv"),
    )
    parser.add_argument(
        "--group-summary-output",
        type=Path,
        default=Path("reports/tables/metadata_confounding_group_summary.csv"),
    )
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    result = run_metadata_confounding_audit(
        metadata,
        feature_sets=args.feature_sets,
        random_state=args.random_state,
    )
    for path in [
        args.metrics_output,
        args.predictions_output,
        args.feature_importance_output,
        args.group_summary_output,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    result.metrics.to_csv(args.metrics_output, index=False)
    result.predictions.to_csv(args.predictions_output, index=False)
    result.feature_importance.to_csv(args.feature_importance_output, index=False)
    result.group_summary.to_csv(args.group_summary_output, index=False)

    print(f"Wrote metadata confounding metrics: {args.metrics_output} ({len(result.metrics)} rows)")
    print(f"Wrote metadata confounding predictions: {args.predictions_output} ({len(result.predictions)} rows)")
    print(f"Wrote metadata feature importance: {args.feature_importance_output} ({len(result.feature_importance)} rows)")
    print(f"Wrote metadata group summary: {args.group_summary_output} ({len(result.group_summary)} rows)")
    if not result.metrics.empty:
        columns = [
            "audit_model",
            "auroc",
            "auprc",
            "balanced_accuracy",
            "f1",
            "n_features",
            "test_positive_prevalence",
        ]
        print(result.metrics[[col for col in columns if col in result.metrics.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
