from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


EVIDENCE_COLUMNS = [
    "claim_id",
    "claim",
    "evidence_type",
    "artifact",
    "comparison",
    "primary_metric",
    "primary_value",
    "secondary_metrics",
    "n_samples",
    "evidence_direction",
    "paper_use",
]


REPRESENTATION_SOURCES = {
    "mfcc": {
        "name": "MFCC",
        "external": "external_model_grid_metrics",
        "internal": "coughvid_internal_metrics",
        "external_artifact": "data/outputs/metrics/external_model_grid_metrics.csv",
        "internal_artifact": "data/outputs/metrics/coughvid_internal_metrics.csv",
    },
    "opensmile_egemaps": {
        "name": "OpenSMILE eGeMAPSv02",
        "external": "external_model_grid_opensmile_egemaps_metrics",
        "internal": "coughvid_internal_opensmile_egemaps_metrics",
        "external_artifact": "data/outputs/metrics/external_model_grid_opensmile_egemaps_metrics.csv",
        "internal_artifact": "data/outputs/metrics/coughvid_internal_opensmile_egemaps_metrics.csv",
    },
    "beats": {
        "name": "BEATs",
        "external": "external_model_grid_beats_metrics",
        "internal": "coughvid_internal_beats_metrics",
        "external_artifact": "data/outputs/metrics/external_model_grid_beats_metrics.csv",
        "internal_artifact": "data/outputs/metrics/coughvid_internal_beats_metrics.csv",
    },
    "panns": {
        "name": "PANNs CNN14",
        "external": "external_model_grid_panns_metrics",
        "internal": "coughvid_internal_panns_metrics",
        "external_artifact": "data/outputs/metrics/external_model_grid_panns_metrics.csv",
        "internal_artifact": "data/outputs/metrics/coughvid_internal_panns_metrics.csv",
    },
}


def _table(tables: Mapping[str, pd.DataFrame], name: str) -> pd.DataFrame:
    frame = tables.get(name)
    return frame if frame is not None else pd.DataFrame()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _best_row(frame: pd.DataFrame, metric: str = "auroc", filters: Mapping[str, object] | None = None) -> pd.Series | None:
    if frame.empty or metric not in frame.columns:
        return None
    filtered = frame.copy()
    for col, value in (filters or {}).items():
        if col not in filtered.columns:
            return None
        filtered = filtered[filtered[col].astype(str).eq(str(value))]
    if filtered.empty:
        return None
    values = _numeric(filtered[metric])
    if values.dropna().empty:
        return None
    return filtered.loc[values.idxmax()]


def _worst_row(frame: pd.DataFrame, metric: str, mask: pd.Series | None = None) -> pd.Series | None:
    if frame.empty or metric not in frame.columns:
        return None
    filtered = frame[mask].copy() if mask is not None else frame.copy()
    if filtered.empty:
        return None
    values = _numeric(filtered[metric])
    if values.dropna().empty:
        return None
    return filtered.loc[values.idxmax()]


def _value(row: pd.Series, col: str, default: object = "") -> object:
    if col not in row.index or pd.isna(row[col]):
        return default
    return row[col]


def _float_value(row: pd.Series, col: str) -> float | None:
    if col not in row.index:
        return None
    try:
        value = float(row[col])
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _int_value(row: pd.Series, col: str) -> int | str:
    value = _float_value(row, col)
    return int(value) if value is not None else ""


def _format_metric_value(value: object) -> str:
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.3f}"


def _secondary_metrics(row: pd.Series, columns: list[str]) -> str:
    parts: list[str] = []
    for col in columns:
        value = _float_value(row, col)
        if value is not None:
            parts.append(f"{col}={value:.3f}")
    return "; ".join(parts)


def _append_metric_row(
    rows: list[dict[str, object]],
    *,
    claim_id: str,
    claim: str,
    evidence_type: str,
    artifact: str,
    comparison: str,
    row: pd.Series,
    primary_metric: str,
    secondary_columns: list[str],
    evidence_direction: str,
    paper_use: str,
) -> None:
    primary_value = _float_value(row, primary_metric)
    if primary_value is None:
        return
    rows.append(
        {
            "claim_id": claim_id,
            "claim": claim,
            "evidence_type": evidence_type,
            "artifact": artifact,
            "comparison": comparison,
            "primary_metric": primary_metric,
            "primary_value": primary_value,
            "secondary_metrics": _secondary_metrics(row, secondary_columns),
            "n_samples": _int_value(row, "n_samples"),
            "evidence_direction": evidence_direction,
            "paper_use": paper_use,
        }
    )


def _safe_constraint_id(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace(">=", "_").replace("<=", "_").replace(">", "_").replace("<", "_")
    text = text.replace("=", "_").replace(".", "_").replace(" ", "_")
    return "_".join(part for part in text.split("_") if part)


def _add_external_representation_rows(rows: list[dict[str, object]], tables: Mapping[str, pd.DataFrame]) -> None:
    for slug, source in REPRESENTATION_SOURCES.items():
        frame = _table(tables, str(source["external"]))
        best = _best_row(frame, metric="auroc")
        if best is None:
            continue
        representation = str(source["name"])
        model = _value(best, "model_name", "model")
        strategy = _value(best, "feature_strategy", "all features")
        _append_metric_row(
            rows,
            claim_id=f"external_transfer_{slug}_best",
            claim=f"Best {representation} external transfer remains weak under Coswara-to-COUGHVID shift.",
            evidence_type="external_transfer",
            artifact=str(source["external_artifact"]),
            comparison=f"{representation} / {model} / {strategy} evaluated on COUGHVID external labels",
            row=best,
            primary_metric="auroc",
            secondary_columns=["auprc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier", "ece", "nll"],
            evidence_direction="cautionary",
            paper_use="Use in Results and Discussion as external-validity evidence; avoid presenting this as deployable screening performance.",
        )


def _add_internal_representation_rows(rows: list[dict[str, object]], tables: Mapping[str, pd.DataFrame]) -> None:
    for slug, source in REPRESENTATION_SOURCES.items():
        frame = _table(tables, str(source["internal"]))
        best = _best_row(frame, metric="auroc")
        if best is None:
            continue
        representation = str(source["name"])
        model = _value(best, "model_name", "model")
        _append_metric_row(
            rows,
            claim_id=f"coughvid_internal_{slug}_best",
            claim=f"{representation} is substantially stronger in COUGHVID internal validation than in external transfer.",
            evidence_type="internal_baseline",
            artifact=str(source["internal_artifact"]),
            comparison=f"{representation} / {model} trained and tested within COUGHVID split",
            row=best,
            primary_metric="auroc",
            secondary_columns=["auprc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier", "ece", "nll"],
            evidence_direction="context",
            paper_use="Use as domain-shift context, not as the main claim of generalization.",
        )


def _add_quality_fusion_row(rows: list[dict[str, object]], tables: Mapping[str, pd.DataFrame]) -> None:
    frame = _table(tables, "quality_weighted_fusion_metrics")
    best = _best_row(frame, metric="auroc")
    if best is None:
        return
    _append_metric_row(
        rows,
        claim_id="internal_quality_weighted_fusion",
        claim="Quality-weighted fusion is the strongest internal audio model family in the current pipeline.",
        evidence_type="internal_audio",
        artifact="data/outputs/metrics/quality_weighted_fusion_metrics.csv",
        comparison=str(_value(best, "fusion_method", "quality-weighted fusion")),
        row=best,
        primary_metric="auroc",
        secondary_columns=["auprc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier", "ece", "nll"],
        evidence_direction="supportive",
        paper_use="Use as the main internal performance result, with explicit non-clinical disclaimer.",
    )


def _add_metadata_rows(rows: list[dict[str, object]], tables: Mapping[str, pd.DataFrame]) -> None:
    frame = _table(tables, "metadata_confounding_metrics")
    for audit_model in ["full_safe_metadata", "symptoms_only", "demographic_protocol_only"]:
        row = _best_row(frame, metric="auroc", filters={"audit_model": audit_model})
        if row is None:
            continue
        _append_metric_row(
            rows,
            claim_id=f"metadata_confounding_{audit_model}",
            claim=f"Non-audio metadata alone predicts COVID label for the {audit_model} audit.",
            evidence_type="metadata_confounding",
            artifact="data/outputs/metrics/metadata_confounding_metrics.csv",
            comparison=str(audit_model),
            row=row,
            primary_metric="auroc",
            secondary_columns=["auprc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier", "ece", "nll"],
            evidence_direction="cautionary",
            paper_use="Use to justify confounding analysis and conservative claims about audio-only causality.",
        )


def _add_controlled_audio_rows(rows: list[dict[str, object]], tables: Mapping[str, pd.DataFrame]) -> None:
    frame = _table(tables, "confounding_controlled_audio_metrics")
    row = _best_row(frame, metric="auroc", filters={"control_method": "ipw_label_propensity"})
    if row is None:
        return
    _append_metric_row(
        rows,
        claim_id="confounding_controlled_audio_ipw",
        claim="The quality-weighted audio signal persists after inverse-propensity confounder control, but with reduced performance.",
        evidence_type="confounding_controlled_audio",
        artifact="data/outputs/metrics/confounding_controlled_audio_metrics.csv",
        comparison=str(_value(row, "fusion_method", "quality_weighted_auprc")),
        row=row,
        primary_metric="auroc",
        secondary_columns=["auprc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier", "ece", "nll", "effective_sample_size"],
        evidence_direction="qualified_supportive",
        paper_use="Use as the strongest defensible audio-result claim because it directly addresses measured confounding.",
    )


def _add_clinical_rows(rows: list[dict[str, object]], tables: Mapping[str, pd.DataFrame]) -> None:
    frame = _table(tables, "clinical_operating_points")
    if frame.empty:
        return
    if "table_source" in frame.columns:
        frame = frame[frame["table_source"].astype(str).eq("quality_weighted_fusion_predictions")]
    if "operating_constraint" not in frame.columns:
        return
    constraints = ["specificity>=0.800", "specificity>=0.900", "specificity>=0.950", "sensitivity>=0.900"]
    for constraint in constraints:
        matches = frame[frame["operating_constraint"].astype(str).eq(constraint)]
        if matches.empty:
            continue
        row = matches.iloc[0]
        metric = "sensitivity" if str(constraint).startswith("specificity") else "specificity"
        _append_metric_row(
            rows,
            claim_id=f"clinical_fusion_{_safe_constraint_id(constraint)}",
            claim=f"At {constraint}, quality-weighted fusion has a concrete operating-point tradeoff.",
            evidence_type="clinical_operating_point",
            artifact="reports/tables/clinical_operating_points.csv",
            comparison=str(constraint),
            row=row,
            primary_metric=metric,
            secondary_columns=["threshold", "sensitivity", "specificity", "precision", "npv", "f1", "balanced_accuracy"],
            evidence_direction="operational_context",
            paper_use="Use to discuss threshold tradeoffs; state that this is a research prototype and not a diagnostic threshold.",
        )


def _add_calibration_rows(rows: list[dict[str, object]], tables: Mapping[str, pd.DataFrame]) -> None:
    frame = _table(tables, "calibration_under_shift_summary")
    if frame.empty:
        return
    internal = _best_row(frame, metric="ece", filters={"prediction_source": "quality_weighted_fusion_predictions"})
    if internal is not None:
        _append_metric_row(
            rows,
            claim_id="calibration_quality_weighted_fusion",
            claim="Internal quality-weighted fusion has small global calibration gap but non-trivial bin-level calibration error.",
            evidence_type="calibration",
            artifact="reports/tables/calibration_under_shift_summary.csv",
            comparison="quality_weighted_fusion_predictions",
            row=internal,
            primary_metric="ece",
            secondary_columns=["observed_prevalence", "mean_probability", "calibration_gap", "mce", "brier", "nll"],
            evidence_direction="mixed",
            paper_use="Use in reliability/calibration discussion to avoid overclaiming calibrated clinical probabilities.",
        )
    if "prediction_source" not in frame.columns:
        return
    external_mask = frame["prediction_source"].astype(str).str.contains("external_model_grid", na=False)
    worst = _worst_row(frame, metric="ece", mask=external_mask)
    if worst is None:
        return
    _append_metric_row(
        rows,
        claim_id="calibration_external_transfer_worst",
        claim="External transfer probabilities are strongly over-confident relative to COUGHVID prevalence.",
        evidence_type="calibration_under_shift",
        artifact="reports/tables/calibration_under_shift_summary.csv",
        comparison=str(_value(worst, "prediction_source", "external_model_grid")),
        row=worst,
        primary_metric="ece",
        secondary_columns=["observed_prevalence", "mean_probability", "calibration_gap", "mce", "brier", "nll"],
        evidence_direction="cautionary",
        paper_use="Use as direct evidence that external probabilities should not be interpreted as calibrated risks.",
    )


def build_publication_evidence_matrix(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    _add_quality_fusion_row(rows, tables)
    _add_external_representation_rows(rows, tables)
    _add_internal_representation_rows(rows, tables)
    _add_metadata_rows(rows, tables)
    _add_controlled_audio_rows(rows, tables)
    _add_clinical_rows(rows, tables)
    _add_calibration_rows(rows, tables)
    matrix = pd.DataFrame(rows)
    if matrix.empty:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    return matrix[EVIDENCE_COLUMNS]


def evidence_matrix_to_markdown(matrix: pd.DataFrame) -> str:
    columns = EVIDENCE_COLUMNS
    lines = [
        "# Publication Evidence Matrix",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in matrix.iterrows():
        values: list[str] = []
        for col in columns:
            value = row[col] if col in row.index else ""
            if isinstance(value, float):
                text = _format_metric_value(value)
            else:
                text = "" if pd.isna(value) else str(value)
            text = text.replace("|", "\\|").replace("\n", " ")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def write_evidence_matrix(matrix: pd.DataFrame, csv_output: Path, markdown_output: Path) -> None:
    csv_output = Path(csv_output)
    markdown_output = Path(markdown_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(csv_output, index=False)
    markdown_output.write_text(evidence_matrix_to_markdown(matrix), encoding="utf-8")
