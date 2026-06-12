#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_audio_btp.publication_evidence import build_publication_evidence_matrix, write_evidence_matrix


def default_input_paths(project_root: Path = Path(".")) -> dict[str, Path]:
    root = Path(project_root)
    metrics = root / "data" / "outputs" / "metrics"
    tables = root / "reports" / "tables"
    return {
        "quality_weighted_fusion_metrics": metrics / "quality_weighted_fusion_metrics.csv",
        "external_model_grid_metrics": metrics / "external_model_grid_metrics.csv",
        "external_model_grid_opensmile_egemaps_metrics": metrics / "external_model_grid_opensmile_egemaps_metrics.csv",
        "external_model_grid_beats_metrics": metrics / "external_model_grid_beats_metrics.csv",
        "external_model_grid_panns_metrics": metrics / "external_model_grid_panns_metrics.csv",
        "coughvid_internal_metrics": metrics / "coughvid_internal_metrics.csv",
        "coughvid_internal_opensmile_egemaps_metrics": metrics / "coughvid_internal_opensmile_egemaps_metrics.csv",
        "coughvid_internal_beats_metrics": metrics / "coughvid_internal_beats_metrics.csv",
        "coughvid_internal_panns_metrics": metrics / "coughvid_internal_panns_metrics.csv",
        "metadata_confounding_metrics": metrics / "metadata_confounding_metrics.csv",
        "confounding_controlled_audio_metrics": metrics / "confounding_controlled_audio_metrics.csv",
        "clinical_operating_points": tables / "clinical_operating_points.csv",
        "calibration_under_shift_summary": tables / "calibration_under_shift_summary.csv",
    }


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_input_tables(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {name: frame for name, path in paths.items() if not (frame := _read_table(path)).empty}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manuscript-ready evidence matrix from publication result artifacts.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("reports/tables/publication_evidence_matrix.csv"))
    parser.add_argument("--markdown-output", type=Path, default=Path("reports/tables/publication_evidence_matrix.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = default_input_paths(project_root=args.project_root)
    tables = read_input_tables(input_paths)
    matrix = build_publication_evidence_matrix(tables)
    if matrix.empty:
        checked = ", ".join(str(path) for path in input_paths.values())
        raise FileNotFoundError(f"No usable evidence artifacts found. Checked: {checked}")
    write_evidence_matrix(matrix, args.output, args.markdown_output)
    print(f"Wrote publication evidence matrix: {args.output} ({len(matrix)} rows)")
    print(f"Wrote publication evidence markdown: {args.markdown_output}")


if __name__ == "__main__":
    main()
