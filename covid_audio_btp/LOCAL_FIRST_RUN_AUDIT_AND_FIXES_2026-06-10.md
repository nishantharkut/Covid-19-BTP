# Local First-Run Audit And Fixes - 2026-06-10

This note records the re-audit after the first local notebook run failed.

## Correct Local Dataset Flow

```text
Covid-19-BTP/
  data/raw/coswara/          # clone https://github.com/iiscleap/Coswara-Data here
    extract_data.py
    combined_data.csv
    annotations/
    Extracted_data/          # produced by python extract_data.py
```

The main notebook default is:

```python
RAW_COSWARA_DIR = PROJECT_ROOT / "data/raw/coswara"
```

The indexer recursively scans that directory, so files under `Extracted_data/` are included.

## Fixes Applied

- Removed EC2 absolute-path fallback from notebooks and replaced it with project-root discovery.
- Added `scripts/00_local_preflight.py` to fail fast before the full notebook.
- Changed `requirements.txt` to core first-run dependencies only.
- Moved pytest to `requirements-dev.txt`.
- Moved xgboost/streamlit to `requirements-optional.txt`.
- Kept PyTorch/Torchaudio out of first-run install; use `requirements-gpu.txt` notes only when enabling CNN.
- Made `scripts/00_check_environment.py` require only runtime core packages.
- Changed `RUN_PUBLICATION_EXTRAS` default to `False` for the first run.
- Changed default CNN batch size to 8 for an 8 GB GPU when CNN is enabled later.
- Made XGBoost opt-in rather than a default ML baseline.
- Added a clear failure if no ML model trains successfully.
- Added calibration fallback for validation branches containing only one class.
- Made validation-weighted fusion choose the best available metric per modality deterministically.

## Verification Run On Codex EC2

Full ML execution was not run here because this EC2 Python does not have pandas/librosa/sklearn installed and does not contain your raw dataset. The deterministic gates passed:

```text
Python py_compile over src/scripts/tests: passed
All notebook code-cell syntax compile: passed
00_local_preflight.py --skip-imports --skip-coswara: passed
Coswara status normalization spot-check: passed
```

## Required Local Gate

Before opening the notebook locally, run:

```bash
cd "$PROJECT_ROOT"
source .venv/bin/activate
python scripts/00_local_preflight.py --coswara-dir data/raw/coswara
```

Do not run the full notebook until this passes.
