# Notebook First Run Guide

Start with `notebooks/00_RUN_EVERYTHING_PUBLICATION.ipynb`. It is the single main notebook for core Coswara plus publication extras. The older `00_MASTER_RUN_ALL.ipynb` and notebooks `01` through `08` are optional review/debug notebooks.

Before running locally, read `DECISION_LOCK_AND_LOCAL_RUN_PROTOCOL.md`. It records the current decisions: Coswara first, COUGHVID later, `RUN_CNN = False`, no SSL/GRL until baseline evidence exists, and datasets live on your local machine rather than this EC2 workspace.

## First Run Guide

This project is designed to be run from Jupyter, with reusable implementation code under `src/covid_audio_btp`.

## Setup

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

## Put Dataset Here

```text
<PROJECT_ROOT>/data/raw/coswara
```

If you use a different path, edit `RAW_COSWARA_DIR` in `notebooks/00_RUN_EVERYTHING_PUBLICATION.ipynb`.

## First Notebook

Open:

```text
notebooks/00_RUN_EVERYTHING_PUBLICATION.ipynb
```

Run it section by section. Do not blindly run all cells on the first attempt; stop at each gate and inspect output.

## Mandatory Review Order

After the master notebook creates each artifact group, inspect:

```text
notebooks/01_dataset_audit.ipynb
notebooks/02_quality_review.ipynb
notebooks/03_feature_review.ipynb
notebooks/04_ml_baseline_review.ipynb
notebooks/06_calibration_fusion_review.ipynb
notebooks/07_shift_confounding_review.ipynb
```

`05_cnn_review.ipynb` is only for after optional GPU CNN training.

## CNN Switch

In the master notebook:

```python
RUN_CNN = False
```

Keep it false until:

- dataset audit passes;
- participant leakage check passes;
- quality audit passes;
- features are healthy;
- classical ML models beat dummy baselines for at least the main modality.

## What To Share If A Cell Fails

Share the failing cell output and, if created, these files:

```text
reports/tables/coswara_layout_audit.csv
reports/tables/validation_issues.csv
reports/tables/nb01_label_by_modality.csv
reports/tables/nb02_quality_summary.csv
reports/tables/nb03_feature_health.csv
```
