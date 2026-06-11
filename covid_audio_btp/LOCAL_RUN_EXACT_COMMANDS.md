# Local Run Exact Commands

Last updated: 2026-06-10

Use this file when running the project on your own local machine. The EC2 workspace does not contain your raw datasets.

These commands assume Linux, macOS, or Windows WSL. If you use native Windows, run through WSL or adapt paths in Anaconda Prompt.

## 0. Goal For The First Run

First run only the Coswara baseline pipeline.

Do not enable CNN, COUGHVID feature extraction, Wav2Vec, HuBERT, SSL, or GRL yet.

The first run settings must be:

```python
COUGHVID_RAW = None
RUN_CNN = False
RUN_COUGHVID_FEATURES = False
RUN_FEATURE_SHIFT_REPORT = False
```

## 1. Choose Local Project Path

Set your local project path once.

Example:

```bash
export PROJECT_ROOT="$HOME/covid_audio_btp"
```

If your extracted project is somewhere else, change the path:

```bash
export PROJECT_ROOT="/absolute/path/to/covid_audio_btp"
```

Check:

```bash
cd "$PROJECT_ROOT"
pwd
ls
```

You should see files like:

```text
README.md
requirements.txt
pyproject.toml
notebooks/
scripts/
src/
```

Approximate time: less than 1 minute.

## 2. Put Datasets In The Correct Local Folders

Create raw-data folders:

```bash
cd "$PROJECT_ROOT"
mkdir -p data/raw/coswara
mkdir -p data/raw/coughvid
```

Coswara must be here:

```text
$PROJECT_ROOT/data/raw/coswara
```

COUGHVID must stay optional for now:

```text
$PROJECT_ROOT/data/raw/coughvid
```

Approximate time: less than 1 minute if already downloaded.

## 3. Verify Coswara Folder Shape

Run:

```bash
cd "$PROJECT_ROOT"
find data/raw/coswara -maxdepth 2 -type f | head -30
find data/raw/coswara -maxdepth 2 -type d | head -30
```

You should see at least:

```text
data/raw/coswara/combined_data.csv
data/raw/coswara/csv_labels_legend.json
data/raw/coswara/annotations/
data/raw/coswara/<date folders>/
```

If you cloned Coswara but did not extract it, run:

```bash
cd "$PROJECT_ROOT/data/raw/coswara"
python extract_data.py
```

Approximate time:

```text
Folder check: less than 1 minute
Coswara extraction: 5-30 minutes depending disk and CPU
```

## 4. Keep COUGHVID Disabled For First Run

For now, only verify it exists somewhere. Do not run it yet.

If you have the official zip:

```bash
cd "$PROJECT_ROOT"
ls -lh data/raw/coughvid
```

Expected optional file:

```text
data/raw/coughvid/public_dataset_v3.zip
```

Do not set `COUGHVID_RAW` yet.

Approximate time: less than 1 minute.

## 5. Create Python Environment

Run:

```bash
cd "$PROJECT_ROOT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` is now the core first-run environment only. Do not install GPU/CNN/demo extras until the Coswara baseline passes.

Optional later:

```bash
pip install -r requirements-dev.txt       # pytest only, for tests
pip install -r requirements-optional.txt  # xgboost and streamlit
```

If `python3` is not available, try:

```bash
python -m venv .venv
source .venv/bin/activate
```

Approximate time:

```text
Fast internet: 15-30 minutes
Slow internet or source builds: 30-60 minutes
```

If dependency installation fails, stop and send the exact error.

## 6. Register Jupyter Kernel

Run:

```bash
cd "$PROJECT_ROOT"
source .venv/bin/activate
python -m ipykernel install --user --name covid-audio-btp --display-name "COVID Audio BTP"
```

Approximate time: less than 1 minute.

## 7. Mandatory Fast Preflight

Run this before opening the notebook:

```bash
cd "$PROJECT_ROOT"
source .venv/bin/activate
python scripts/00_local_preflight.py --coswara-dir data/raw/coswara
```

This checks notebook syntax, Python script syntax, required imports, and whether Coswara contains extracted audio files.

Optional, if you have time:

```bash
python scripts/00_check_environment.py
pip install -r requirements-dev.txt
pytest tests/test_labels.py tests/test_data_index.py tests/test_quality.py tests/test_validation.py -q
```

Approximate time:

```text
Preflight: less than 1 minute
Environment check: less than 1 minute
Small tests: 1-5 minutes
```

If preflight fails, stop and send the exact output. Do not start the full notebook until preflight passes.

## 8. Start Jupyter

Run:

```bash
cd "$PROJECT_ROOT"
source .venv/bin/activate
jupyter lab
```

Open:

```text
notebooks/00_RUN_EVERYTHING_PUBLICATION.ipynb
```

Select kernel:

```text
COVID Audio BTP
```

Approximate time to start: less than 2 minutes.

## 9. Set Notebook Values Before Running

In the first configuration cell of `00_RUN_EVERYTHING_PUBLICATION.ipynb`, set:

```python
RAW_COSWARA_DIR = PROJECT_ROOT / "data/raw/coswara"
COUGHVID_RAW = None

FORCE_REBUILD = False

RUN_CORE_COSWARA = True
RUN_LAYOUT_AUDIT = True
RUN_VALIDATION = True
RUN_CNN = False

RUN_PUBLICATION_EXTRAS = False
RUN_METADATA_BASELINE = True
RUN_QUALITY_WEIGHTED_FUSION = True
RUN_ABSTENTION = True
RUN_BOOTSTRAP_CI = True

RUN_COUGHVID_INDEX = False
RUN_COUGHVID_FEATURES = False
RUN_CROSS_DATASET = False
RUN_FEATURE_SHIFT_REPORT = False

RUN_PAIRED_MODEL_COMPARISON = True
RUN_CONFOUNDING_MATCHING = True
RUN_EXPERIMENT_MANIFEST = True
```

Important:

```text
COUGHVID_RAW must be None for the first run.
RUN_CNN must be False for the first run.
```

Approximate time: 1-2 minutes.

## 10. Run Notebook Section By Section

Do not blindly run every cell on the first attempt.

Recommended order:

1. Run setup/config cells.
2. Run raw Coswara layout audit.
3. Run Coswara index.
4. Run metadata cleaning.
5. Run participant split.
6. Run quality audit.
7. Run validation gate.
8. Run feature extraction.
9. Run classical ML baselines.
10. Run branch calibration.
11. Run fusion.
12. Run publication extras that do not require COUGHVID.
13. Run paper tables and experiment manifest.

Approximate time on 4-8 CPU cores, 16-32 GB RAM:

```text
Setup/config cells: less than 2 minutes
Raw layout audit: 1-5 minutes
Coswara index + metadata cleaning: 2-10 minutes
Participant split: less than 2 minutes
Quality audit: 20-90 minutes
Feature extraction: 30-120 minutes
Classical ML baselines: 5-30 minutes
Calibration + fusion: 5-20 minutes
Metadata baseline + abstention + bootstrap + paper tables: 10-45 minutes
Total first Coswara baseline run: 1.5-5 hours typical
Slow laptop worst case: 5-8 hours
```

If the notebook appears stuck during quality audit or feature extraction, check CPU activity before interrupting. Those stages read many audio files.

## 11. First Success Gate

After the first run, these files should exist:

```bash
cd "$PROJECT_ROOT"
ls -lh reports/tables/coswara_layout_audit.csv
ls -lh reports/tables/validation_issues.csv
ls -lh data/interim/coswara_index.csv
ls -lh data/processed/metadata_clean.csv
ls -lh data/processed/audio_quality.csv
ls -lh data/outputs/metrics/ml_baseline_metrics.csv
ls -lh data/outputs/metrics/calibration_metrics.csv
ls -lh data/outputs/metrics/fusion_metrics.csv
```

Quick preview:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

paths = [
    "reports/tables/coswara_layout_audit.csv",
    "reports/tables/validation_issues.csv",
    "data/interim/coswara_index.csv",
    "data/processed/metadata_clean.csv",
    "data/processed/audio_quality.csv",
    "data/outputs/metrics/ml_baseline_metrics.csv",
    "data/outputs/metrics/calibration_metrics.csv",
    "data/outputs/metrics/fusion_metrics.csv",
]

for path in paths:
    p = Path(path)
    print("\\n==", path, "==")
    if not p.exists():
        print("MISSING")
        continue
    df = pd.read_csv(p)
    print("shape:", df.shape)
    print(df.head(3).to_string(index=False))
PY
```

Approximate time: less than 2 minutes.

## 12. Stop Conditions

Stop and send output if any of these happen:

```text
Coswara folder not found
combined_data.csv not found
labels mostly become unknown
positive or negative labels are missing
participant leakage assertion fails
validation gate raises an error
most audio files are corrupt/silent
feature extraction crashes on codec/audio loading
ML baseline metrics file is missing
calibration/fusion metrics file is missing
```

Do not enable COUGHVID, CNN, SSL, or GRL to "fix" these. These are baseline data/code issues.

## 13. What To Send Back To Codex

If the first run succeeds, send/upload:

```text
reports/tables/coswara_layout_audit.csv
reports/tables/validation_issues.csv
data/interim/coswara_index.csv
data/processed/metadata_clean.csv
data/processed/audio_quality.csv
data/outputs/metrics/ml_baseline_metrics.csv
data/outputs/metrics/calibration_metrics.csv
data/outputs/metrics/fusion_metrics.csv
data/outputs/metrics/paper_metric_table.csv
reports/tables/confounding_balance.csv
data/outputs/metrics/abstention_coverage_curve.csv
```

If the first run fails, send:

```text
1. failed notebook cell screenshot or copied cell text
2. full traceback
3. any generated CSVs from the list above
4. your local OS, Python version, CPU/RAM, and whether a GPU is present
```

Commands for local system info:

```bash
python --version
python - <<'PY'
import os, platform
print("platform:", platform.platform())
print("cpu_count:", os.cpu_count())
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_check_error:", exc)
PY
```

## 14. When To Enable COUGHVID

Only after Coswara baseline succeeds.

Then change notebook settings:

```python
COUGHVID_RAW = PROJECT_ROOT / "data/raw/coughvid/public_dataset_v3.zip"
RUN_COUGHVID_INDEX = True
RUN_COUGHVID_FEATURES = False
RUN_CROSS_DATASET = False
RUN_FEATURE_SHIFT_REPORT = False
```

Run only the COUGHVID index step first.

Approximate time:

```text
COUGHVID zip index: 5-30 minutes depending zip size and disk speed
```

Then smoke test COUGHVID feature extraction:

```python
RUN_COUGHVID_FEATURES = True
COUGHVID_FEATURE_MAX_ROWS = 25
```

Approximate time:

```text
25-row COUGHVID feature smoke test: 5-20 minutes
Full COUGHVID feature extraction: several hours, sometimes overnight
```

Only after the smoke test works, set:

```python
COUGHVID_FEATURE_MAX_ROWS = None
```

## 15. When To Enable CNN

Only after classical Coswara baseline is clean.

Minimum GPU:

```text
6-8 GB VRAM
```

Notebook setting:

```python
RUN_CNN = True
```

Approximate time:

```text
Small GPU: 1-4 hours
T4/RTX 3060 or better: 30 minutes-2 hours
CPU-only CNN: not recommended
```

## 16. Advanced SSL/GRL Is Not For First Run

Do not run or implement Wav2Vec/HuBERT/GRL before baseline evidence exists.

Future requirements:

```text
GPU: 12-16 GB VRAM minimum
RAM: 32 GB preferred
Disk: 80-150 GB free
```

Future scripts, only after baseline review:

```text
25_acoustic_domain_proxy.py
26_extract_ssl_embeddings.py
27_train_adversarial_ssl.py
28_compare_adversarial_vs_baselines.py
```

These are advanced ablations, not replacements for the current pipeline.

