#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_audio_btp.final_report import write_final_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final evidence-driven BTP publication report.")
    parser.add_argument("--evidence", type=Path, default=Path("reports/tables/publication_evidence_matrix.csv"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/final/BTP_PUBLICATION_RESULTS_REPORT.md"))
    parser.add_argument("--summary-output", type=Path, default=Path("reports/final/BTP_PUBLICATION_RESULTS_SUMMARY.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.evidence.exists() or args.evidence.stat().st_size == 0:
        raise FileNotFoundError(f"Publication evidence matrix not found: {args.evidence}")
    evidence = pd.read_csv(args.evidence)
    write_final_reports(evidence, args.report_output, args.summary_output)
    print(f"Wrote final publication report: {args.report_output}")
    print(f"Wrote final publication summary: {args.summary_output}")


if __name__ == "__main__":
    main()
