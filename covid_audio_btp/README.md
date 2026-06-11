# COVID/Respiratory Audio BTP

Reliability-aware and confounding-controlled respiratory audio screening from cough, breath, and speech.

This project is a research prototype only. It is not a clinical diagnostic tool.

## Master Record

Read this first for copy-paste local commands:

```text
LOCAL_RUN_EXACT_COMMANDS.md
```

Read this for the complete first-run-to-final-BTP workflow:

```text
BTP_FULL_LOCAL_RUNBOOK.md
```

Read this audit before expecting SOTA-level results:

```text
DEEP_LOGIC_AUDIT_2026-06-10.md
```

Read this to understand the locked grading/publication strategy:

```text
BTP_A_GRADE_AND_PUBLICATION_TARGET.md
```

Then read this for the current local-run decisions:

```text
DECISION_LOCK_AND_LOCAL_RUN_PROTOCOL.md
```

The advanced Gemini/PDF-derived extension is recorded separately and should be used only after baseline results exist:

```text
ADVANCED_EXTENSION_FROM_GEMINI_PDFS.md
```

Then read the full master record:

```text
MASTER_PROJECT_GUIDE.md
```

It contains the project status, decisions, code map, dataset notes, workflow, redundancy audit, and remaining risks.

## Main Notebook

Run only this notebook for the full workflow:

```text
notebooks/00_RUN_EVERYTHING_PUBLICATION.ipynb
```

It has 20 cells total: 10 code and 10 markdown.

Other notebooks are optional review/debug notebooks.

## Local Setup

On your local machine, replace `<PROJECT_ROOT>` with the extracted project path.

```bash
cd <PROJECT_ROOT>
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts/00_local_preflight.py --coswara-dir data/raw/coswara
jupyter lab
```

`requirements.txt` is intentionally core-only for the first Coswara run. GPU/CNN, XGBoost, Streamlit, and pytest extras are separated into `requirements-gpu.txt`, `requirements-optional.txt`, and `requirements-dev.txt`.

Place Coswara under:

```text
data/raw/coswara/
```

Keep these first-run notebook settings:

```python
COUGHVID_RAW = None
RUN_CNN = False
RUN_PUBLICATION_EXTRAS = False
RUN_COUGHVID_FEATURES = False
```

Optional COUGHVID path can be configured after the Coswara baseline works.
