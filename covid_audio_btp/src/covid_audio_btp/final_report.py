from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_EVIDENCE_COLUMNS = {
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
}


def _validate_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    if evidence is None or evidence.empty:
        raise ValueError("Final report requires a non-empty publication evidence matrix")
    missing = REQUIRED_EVIDENCE_COLUMNS - set(evidence.columns)
    if missing:
        raise ValueError(f"Evidence matrix is missing required columns: {sorted(missing)}")
    frame = evidence.copy()
    frame["primary_value"] = pd.to_numeric(frame["primary_value"], errors="coerce")
    return frame


def _format_number(value: object, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def _rows_by_type(evidence: pd.DataFrame, evidence_type: str) -> pd.DataFrame:
    return evidence[evidence["evidence_type"].astype(str).eq(evidence_type)].copy()


def _row_by_claim(evidence: pd.DataFrame, claim_id: str) -> pd.Series | None:
    rows = evidence[evidence["claim_id"].astype(str).eq(claim_id)]
    return rows.iloc[0] if not rows.empty else None


def _best_by_type(evidence: pd.DataFrame, evidence_type: str, metric: str | None = None) -> pd.Series | None:
    rows = _rows_by_type(evidence, evidence_type)
    if metric is not None:
        rows = rows[rows["primary_metric"].astype(str).eq(metric)]
    if rows.empty:
        return None
    values = pd.to_numeric(rows["primary_value"], errors="coerce")
    if values.dropna().empty:
        return None
    return rows.loc[values.idxmax()]


def _metric_phrase(row: pd.Series | None) -> str:
    if row is None:
        return "not available"
    return f"{row['primary_metric']}={_format_number(row['primary_value'])}"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "No rows available.\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col] if col in row.index else ""
            if isinstance(value, float):
                text = _format_number(value)
            else:
                text = "" if pd.isna(value) else str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _evidence_table(evidence: pd.DataFrame) -> str:
    columns = [
        "claim_id",
        "evidence_type",
        "primary_metric",
        "primary_value",
        "n_samples",
        "evidence_direction",
    ]
    return _markdown_table(evidence[columns], columns)


def _architecture_section() -> str:
    return """## Pipeline Architecture

The pipeline is organized as an evidence-first audio ML workflow:

1. Build cleaned Coswara metadata, quality labels, participant-level splits, and audio features.
2. Train internal audio baselines and calibrated fusion models on leakage-controlled splits.
3. Compare handcrafted and learned representations: MFCC, OpenSMILE eGeMAPSv02, BEATs, and PANNs CNN14.
4. Evaluate source-trained models on COUGHVID as external transfer validation.
5. Run COUGHVID internal baselines to separate representation capacity from dataset shift.
6. Audit metadata confounding, then evaluate quality-weighted audio after inverse-propensity weighting.
7. Report clinical operating points and calibration-under-shift to avoid relying only on AUROC.

This architecture supports a robustness and external-validation study. It does not support a clinical-deployment claim.
"""


def _decision_log_section() -> str:
    return """## Decision Log

- MFCC was retained as the traditional handcrafted baseline because it anchors the project against the original pipeline.
- OpenSMILE eGeMAPSv02 was added as a stronger handcrafted clinical-acoustic baseline.
- BEATs was selected as the main general-audio learned embedding model because it directly tests whether modern transformer audio representations help.
- PANNs CNN14 was added as the CNN audio-embedding comparison, so representation architecture is not reduced to a single transformer result.
- wav2vec2 was treated as a lower-priority speech-biased option because the current scientific question is cough/audio representation robustness, not speech recognition transfer.
- External validation was prioritized over leaderboard tuning because Coswara-to-COUGHVID transfer is the central generalization test.
- Metadata confounding, IPW control, operating points, and calibration shift were added because medical-audio claims require reliability and bias checks, not only accuracy metrics.
"""


def _results_section(evidence: pd.DataFrame) -> str:
    external = _rows_by_type(evidence, "external_transfer")
    internal = _rows_by_type(evidence, "internal_baseline")
    metadata = _rows_by_type(evidence, "metadata_confounding")
    controlled = _row_by_claim(evidence, "confounding_controlled_audio_ipw")
    clinical = _rows_by_type(evidence, "clinical_operating_point")
    calibration = _rows_by_type(evidence, "calibration_under_shift")

    best_external = _best_by_type(evidence, "external_transfer", metric="auroc")
    best_internal = _best_by_type(evidence, "internal_baseline", metric="auroc")
    best_metadata = _best_by_type(evidence, "metadata_confounding", metric="auroc")

    parts = [
        "## Quantitative Results",
        "",
        f"- Best external transfer: {_metric_phrase(best_external)}.",
        f"- Best COUGHVID internal baseline: {_metric_phrase(best_internal)}.",
        f"- Strongest metadata-only confounding audit: {_metric_phrase(best_metadata)}.",
        f"- IPW-controlled audio: {_metric_phrase(controlled)}.",
        "",
        "### External Transfer",
        _markdown_table(external, ["claim_id", "comparison", "primary_metric", "primary_value", "secondary_metrics", "n_samples"]),
        "### Internal COUGHVID Baselines",
        _markdown_table(internal, ["claim_id", "comparison", "primary_metric", "primary_value", "secondary_metrics", "n_samples"]),
        "### Confounding And Controlled Audio",
        _markdown_table(pd.concat([metadata, controlled.to_frame().T if controlled is not None else pd.DataFrame()], ignore_index=True), ["claim_id", "primary_metric", "primary_value", "secondary_metrics", "n_samples", "evidence_direction"]),
        "### Clinical Operating Points",
        _markdown_table(clinical, ["claim_id", "comparison", "primary_metric", "primary_value", "secondary_metrics", "n_samples"]),
        "### Calibration Under Shift",
        _markdown_table(calibration, ["claim_id", "comparison", "primary_metric", "primary_value", "secondary_metrics", "n_samples"]),
    ]
    return "\n".join(parts)


def _interpretation_section() -> str:
    return """## Interpretation

The current evidence does not show a deployable COVID screening model. The stronger and more defensible interpretation is that internal audio models can learn label-associated signal, but external transfer and calibration degrade substantially across datasets. The contrast between internal COUGHVID performance and Coswara-to-COUGHVID transfer supports domain shift as a central finding.

The metadata audits are especially important: symptoms, demographics, recording protocol, and related metadata predict the label strongly. Therefore, any audio-only claim must be framed as association under measured controls rather than as a causal COVID biomarker.
"""


def _novelty_section() -> str:
    return """## Novelty

The novelty is not a new neural architecture. The contribution is a controlled, evidence-driven evaluation layer around COVID audio classification:

- Direct comparison of traditional handcrafted, stronger handcrafted, transformer audio, and CNN audio representations.
- Explicit internal versus external validation contrast on Coswara and COUGHVID.
- Metadata-only confounding audit showing how non-audio variables can explain labels.
- Inverse-propensity weighted controlled audio evaluation.
- Clinical operating-point reporting instead of AUROC-only reporting.
- Calibration-under-shift analysis showing that external probabilities should not be interpreted as calibrated risk.
"""


def _comparison_section() -> str:
    return """## Related-Paper Comparison Position

The final paper should compare against the exact related papers listed in the original project document. That comparison should be filled from a structured related-paper table, not from memory. Until that table is added, the correct qualitative comparison is:

- Many COVID audio papers emphasize internal validation or single-dataset performance.
- This work emphasizes external validation, measured confounding, calibration, and operating-point behavior.
- Therefore the paper should not claim higher accuracy than prior work; it should claim a stronger reliability and robustness analysis.
"""


def _publication_readiness_section() -> str:
    return """## Publication Readiness

This is BTP-ready if presented as a rigorous negative/robustness study. It is not ready as a clinical diagnostic or high-accuracy detector paper.

Defensible claim: COVID audio models can perform well internally, but measured metadata confounding and cross-dataset shift substantially weaken external validity; robust reporting requires external validation, confounding audits, operating points, and calibration analysis.

Non-defensible claim: this system is ready for clinical diagnosis, screening deployment, or individual health decision-making.
"""


def _limitations_section() -> str:
    return """## Limitations

- External transfer remains weak, so deployment claims are not supported.
- IPW control addresses measured confounders only; unmeasured confounding may remain.
- COUGHVID and Coswara labels, collection protocols, and class prevalences differ.
- Related-paper quantitative comparison still needs a structured table based on the exact papers from the original document.
- The current system is a research prototype, not a medical device.
"""


def _next_steps_section() -> str:
    return """## Remaining Work

1. Add a structured related-paper comparison table from the original project document.
2. Convert this report into the final BTP manuscript sections.
3. Keep the central claim conservative: robustness analysis, not clinical deployment.
4. Archive final artifacts and Git tags after the final report and related-paper table are frozen.
"""


def build_final_report(evidence: pd.DataFrame) -> str:
    evidence = _validate_evidence(evidence)
    sections = [
        "# COVID Audio BTP Final Results Report",
        "",
        "## Executive Position",
        "",
        "The implementation supports a defensible BTP and potential robustness-oriented publication story. It is not a clinically deployable diagnostic model. It does not support claiming clinical deployment. The strongest finding is that internal audio performance can look promising while external transfer, confounding, and calibration analyses reveal major reliability limits.",
        "",
        _architecture_section(),
        _decision_log_section(),
        _results_section(evidence),
        "## Quantitative Evidence Matrix",
        _evidence_table(evidence),
        _interpretation_section(),
        _novelty_section(),
        _comparison_section(),
        _publication_readiness_section(),
        _limitations_section(),
        _next_steps_section(),
    ]
    return "\n".join(sections).rstrip() + "\n"


def build_summary_report(evidence: pd.DataFrame) -> str:
    evidence = _validate_evidence(evidence)
    best_external = _best_by_type(evidence, "external_transfer", metric="auroc")
    controlled = _row_by_claim(evidence, "confounding_controlled_audio_ipw")
    metadata = _best_by_type(evidence, "metadata_confounding", metric="auroc")
    calibration = _row_by_claim(evidence, "calibration_external_transfer_worst")
    return (
        "# COVID Audio BTP Results Summary\n\n"
        f"- Best external transfer: {_metric_phrase(best_external)}; this remains weak and cautionary.\n"
        f"- IPW-controlled audio: {_metric_phrase(controlled)}; audio signal persists but is reduced after measured confounder control.\n"
        f"- Strongest metadata confounding result: {_metric_phrase(metadata)}; non-audio variables strongly predict labels.\n"
        f"- Worst external calibration shift: {_metric_phrase(calibration)}; external probabilities are not reliable calibrated risks.\n\n"
        "Conclusion: the project is defensible as a robustness and external-validation BTP study, not as a clinically deployable diagnostic model.\n"
    )


def write_final_reports(evidence: pd.DataFrame, report_output: Path, summary_output: Path) -> None:
    report_output = Path(report_output)
    summary_output = Path(summary_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(build_final_report(evidence), encoding="utf-8")
    summary_output.write_text(build_summary_report(evidence), encoding="utf-8")
