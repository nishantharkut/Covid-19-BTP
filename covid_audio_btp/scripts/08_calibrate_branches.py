#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from covid_audio_btp.calibration import IsotonicCalibrator, PlattCalibrator, TemperatureScaler
from covid_audio_btp.metrics import evaluate_predictions, labels_to_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate branch predictions using validation split only.")
    parser.add_argument("--validation-predictions", required=True, type=Path)
    parser.add_argument("--test-predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/outputs/metrics/calibrated_branch_predictions.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/calibration_metrics.csv"))
    parser.add_argument("--method", choices=["platt", "isotonic", "temperature"], default="platt")
    return parser.parse_args()


def _fit_calibrator(method: str, validation: pd.DataFrame):
    y_val = labels_to_binary(validation["label_binary"])
    if len(np.unique(y_val)) < 2:
        return None
    if method == "temperature":
        if "logit" not in validation.columns:
            raise ValueError("Temperature scaling requires a logit column")
        return TemperatureScaler().fit(validation["logit"].to_numpy(), y_val)
    if method == "isotonic":
        return IsotonicCalibrator().fit(validation["probability"].to_numpy(), y_val)
    return PlattCalibrator().fit(validation["probability"].to_numpy(), y_val)


def _transform(calibrator, method: str, test: pd.DataFrame) -> np.ndarray:
    if calibrator is None:
        return test["probability"].astype(float).to_numpy()
    if method == "temperature":
        return calibrator.transform_logits(test["logit"].to_numpy())
    return calibrator.transform(test["probability"].to_numpy())


def main() -> None:
    args = parse_args()
    validation = pd.read_csv(args.validation_predictions)
    test = pd.read_csv(args.test_predictions)
    output_frames = []
    metric_frames = []

    group_cols = ["model_name", "modality"]
    for key, val_group in validation.groupby(group_cols, dropna=False):
        model_name, modality = key
        test_group = test[(test["model_name"] == model_name) & (test["modality"] == modality)].copy()
        if test_group.empty:
            continue
        calibrator = _fit_calibrator(args.method, val_group)
        test_group["raw_probability"] = test_group["probability"]
        test_group["probability"] = _transform(calibrator, args.method, test_group)
        test_group["calibration_method"] = args.method if calibrator is not None else "identity_single_class_validation"
        output_frames.append(test_group)
        metrics = evaluate_predictions(test_group)
        metrics["model_name"] = model_name
        metrics["modality"] = modality
        metrics["calibration_method"] = args.method
        metric_frames.append(metrics)

    if not output_frames:
        raise RuntimeError("No calibrated prediction groups were produced. Check ML validation/test prediction files.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(output_frames, ignore_index=True).to_csv(args.output, index=False)
    if metric_frames:
        pd.concat(metric_frames, ignore_index=True).to_csv(args.metrics_output, index=False)
    else:
        pd.DataFrame().to_csv(args.metrics_output, index=False)
    print(f"Wrote calibrated predictions: {args.output}")
    print(f"Wrote calibration metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()

