#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_audio_btp.models_ml import save_model, train_single_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical ML baselines by modality.")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--models-dir", type=Path, default=Path("data/outputs/models"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/ml_baseline_metrics.csv"))
    parser.add_argument("--validation-output", type=Path, default=Path("data/outputs/metrics/ml_predictions_validation.csv"))
    parser.add_argument("--test-output", type=Path, default=Path("data/outputs/metrics/ml_predictions_test.csv"))
    parser.add_argument("--modalities", nargs="+", default=["cough", "breath", "speech"])
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=["dummy_most_frequent", "dummy_stratified", "logistic_regression", "random_forest"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    validation_frames = []
    test_frames = []
    for modality in args.modalities:
        for model_name in args.model_names:
            try:
                result = train_single_model(features, model_name=model_name, modality=modality)
            except Exception as exc:
                print(f"SKIP model={model_name} modality={modality}: {exc}")
                continue
            metric_rows.append(result.metrics)
            validation_frames.append(result.validation_predictions)
            test_frames.append(result.test_predictions)
            save_model(result.model, args.models_dir / f"{model_name}_{modality}.joblib")
            print(f"Trained {model_name} / {modality}: AUROC={result.metrics.get('auroc')}")

    if not metric_rows:
        counts = (
            features.groupby(["modality", "label_binary", "split"], dropna=False)
            .size()
            .reset_index(name="n")
            .to_string(index=False)
            if {"modality", "label_binary", "split"}.issubset(features.columns)
            else "feature table is missing modality/label_binary/split columns"
        )
        raise RuntimeError(
            "No ML models were trained successfully. Check supervised label coverage, splits, "
            "and feature columns. Feature counts:\n"
            f"{counts}"
        )

    pd.DataFrame(metric_rows).to_csv(args.metrics_output, index=False)
    pd.concat(validation_frames, ignore_index=True).to_csv(args.validation_output, index=False)
    pd.concat(test_frames, ignore_index=True).to_csv(args.test_output, index=False)
    print(f"Wrote metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()

