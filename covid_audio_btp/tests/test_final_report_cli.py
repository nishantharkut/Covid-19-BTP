from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "35_make_final_publication_report.py"
    spec = importlib.util.spec_from_file_location("final_report_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_report_cli_writes_report_and_summary(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    tables_dir = tmp_path / "reports" / "tables"
    final_dir = tmp_path / "reports" / "final"
    tables_dir.mkdir(parents=True)
    evidence_path = tables_dir / "publication_evidence_matrix.csv"
    report_path = final_dir / "BTP_PUBLICATION_RESULTS_REPORT.md"
    summary_path = final_dir / "BTP_PUBLICATION_RESULTS_SUMMARY.md"

    pd.DataFrame(
        [
            {
                "claim_id": "external_transfer_beats_best",
                "claim": "Best BEATs external transfer remains weak under Coswara-to-COUGHVID shift.",
                "evidence_type": "external_transfer",
                "artifact": "external.csv",
                "comparison": "BEATs",
                "primary_metric": "auroc",
                "primary_value": 0.553,
                "secondary_metrics": "auprc=0.039",
                "n_samples": 8331,
                "evidence_direction": "cautionary",
                "paper_use": "Use in Results.",
            },
            {
                "claim_id": "metadata_confounding_full_safe_metadata",
                "claim": "Non-audio metadata alone predicts COVID label.",
                "evidence_type": "metadata_confounding",
                "artifact": "metadata.csv",
                "comparison": "full_safe_metadata",
                "primary_metric": "auroc",
                "primary_value": 0.964,
                "secondary_metrics": "auprc=0.928",
                "n_samples": 2862,
                "evidence_direction": "cautionary",
                "paper_use": "Use in Discussion.",
            },
        ]
    ).to_csv(evidence_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "35_make_final_publication_report.py",
            "--evidence",
            str(evidence_path),
            "--report-output",
            str(report_path),
            "--summary-output",
            str(summary_path),
        ],
    )

    module.main()

    assert report_path.exists()
    assert summary_path.exists()
    assert "COVID Audio BTP Final Results Report" in report_path.read_text()
    assert "COVID Audio BTP Results Summary" in summary_path.read_text()
