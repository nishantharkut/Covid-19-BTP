# BTP Full Local Runbook

Last updated: 2026-06-11

## Goal

This is the single end-to-end local execution guide for the COVID/respiratory-audio BTP project.

Primary goal:

```text
Get an A-grade BTP with a reproducible, working, leakage-safe, quality-aware, calibrated respiratory-audio research pipeline.
```

Secondary goal:

```text
Keep the work strong enough for a later tier-2 conference or journal submission after real results and ablations exist.
```

Do not skip the gates. If a gate fails, stop and send the exact output back instead of manually changing code for hours.

## Lab Machine Assumption

```text
CPU: Intel i7-14700, 24 exposed cores
RAM: 19 GB
GPU: NVIDIA T1000 8 GB
Disk: about 430 GB
OS: Linux VM
```

This machine is suitable for the Coswara baseline, classical ML, calibration, fusion, confounding analysis, compact CNN, and carefully batched frozen SSL embeddings later. It is not the first-choice machine for full Wav2Vec/HuBERT fine-tuning or large GRL/adversarial SSL training.

## Phase Map

| Phase | Task | Approx Time | Required |
|---|---|---:|---|
| 0 | Clone project and Coswara | 10-60 min | yes |
| 1 | Environment and preflight | 20-70 min | yes |
| 2 | First Coswara baseline notebook run | 1.5-5 hr typical, 5-8 hr slow | yes |
| 3 | BTP strong extras without COUGHVID | 20-90 min | strongly recommended |
| 4 | Compact CNN | 30 min-4 hr | optional |
| 5 | COUGHVID cough-only external validation | 1 hr to overnight | publication-track |
| 6 | Final BTP packaging | 1-3 hr | yes |
| 7 | SSL/domain-proxy ablations | later | publication-track only |

## Phase 0: Project And Dataset Setup

### 0.1 Clone Your Project

```bash
cd "$HOME"
git clone https://github.com/nishantharkut/Covid-19-BTP.git
cd Covid-19-BTP
```

Expected files:

```bash
ls
```

You should see:

```text
README.md
requirements.txt
pyproject.toml
notebooks/
scripts/
src/
```

If these are missing, you are not in the project root or the latest code is not present in the repo.

### 0.2 Ignore Datasets And Generated Artifacts

Run this once:

```bash
printf "\n# Local datasets and generated artifacts\ndata/raw/\ndata/interim/\ndata/processed/\ndata/outputs/\nreports/\n.cache/\n.ipynb_checkpoints/\n" >> .gitignore
git status --short
```

Expected:

```text
.gitignore may be modified.
No raw dataset files should appear as tracked files.
```

If many files under `data/raw/` appear in `git status`, stop before committing.

### 0.3 Clone Coswara

```bash
mkdir -p data/raw
git clone https://github.com/iiscleap/Coswara-Data.git data/raw/coswara
```

Expected time:

```text
5-30 minutes depending internet speed
```

Check:

```bash
ls data/raw/coswara | head
```

Expected items include:

```text
combined_data.csv
extract_data.py
```

### 0.4 Extract Coswara

```bash
cd data/raw/coswara
python3 extract_data.py
cd ../../..
```

Expected time:

```text
5-30 minutes typical
30-60 minutes if disk/network is slow
```

Check that audio exists:

```bash
find data/raw/coswara -type f \( -name "*.wav" -o -name "*.flac" -o -name "*.webm" -o -name "*.ogg" \) | head
```

If nothing prints, extraction did not complete.

## Phase 1: Environment And Mandatory Preflight

### 1.1 Create Environment

```bash
cd "$HOME/Covid-19-BTP"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Expected time:

```text
15-30 minutes with good internet
30-60 minutes with slow internet or source builds
```

If `python3 -m venv` fails:

```bash
sudo apt update
sudo apt install python3-venv
python3 -m venv .venv
```

If dependency installation fails, stop and send the exact pip error.

### 1.2 Register Jupyter Kernel

```bash
source .venv/bin/activate
python -m ipykernel install --user --name covid-audio-btp --display-name "COVID Audio BTP"
```

Expected time:

```text
less than 1 minute
```

### 1.3 Run Mandatory Preflight

```bash
source .venv/bin/activate
python scripts/00_local_preflight.py --coswara-dir data/raw/coswara
```

Expected success:

```text
Notebook syntax: OK
Python syntax: OK
Required imports: OK
Coswara audio files discovered: non-zero
Coswara CSV files discovered: non-zero
Preflight passed. It is safe to start the notebook pipeline.
```

Warnings for these optional packages are acceptable in the first run:

```text
xgboost
torch
torchaudio
streamlit
```

They are not required for the first Coswara baseline.

### 1.4 Preflight Failures

| Failure | Meaning | Fix |
|---|---|---|
| `Coswara path does not exist` | wrong path | clone Coswara into `data/raw/coswara` |
| `Coswara audio files discovered: 0` | extraction missing | run `python3 extract_data.py` inside `data/raw/coswara` |
| `No module named covid_audio_btp` | package not installed | run `pip install -e .` |
| `No module named pandas/librosa/sklearn` | dependency install failed | rerun `pip install -r requirements.txt` |
| notebook syntax error | code mismatch | send full preflight output |

Do not open Jupyter until preflight passes.

## Phase 2: First Coswara Baseline Notebook Run

### 2.1 Start Jupyter

```bash
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

### 2.2 First-Run Notebook Settings

Use exactly these conservative settings:

```python
RAW_COSWARA_DIR = PROJECT_ROOT / "data/raw/coswara"
COUGHVID_RAW = None

FORCE_REBUILD = False
RUN_ENV_CHECK = True
RUN_LAYOUT_AUDIT = True

RUN_CORE_COSWARA = True
RUN_FEATURES = True
RUN_ML_BASELINES = True
RUN_CALIBRATION = True
RUN_FUSION = True
RUN_CNN = False
RUN_SHIFT_CHECKS = True
RUN_REPORT_ASSETS = True

RUN_PUBLICATION_EXTRAS = False
RUN_COUGHVID_INDEX = False
RUN_COUGHVID_FEATURES = False
RUN_CROSS_DATASET = False
RUN_FEATURE_SHIFT_REPORT = False

COUGHVID_FEATURE_MAX_ROWS = 25
MIN_COUGH_DETECTED = 0.8
CNN_EPOCHS = 10
CNN_BATCH_SIZE = 8
```

If preflight passed, you can do:

```text
Kernel -> Restart Kernel and Clear Outputs
Run -> Run All Cells
```

If you want a safer first attempt, run section by section through the validation gate first, then continue.

### 2.3 Expected Time

```text
Setup/config: less than 2 minutes
Layout/index/metadata/split: 5-20 minutes
Quality audit: 20-90 minutes
Feature extraction: 30-120 minutes
ML baselines: 5-30 minutes
Calibration/fusion/report assets: 5-20 minutes
Total typical: 1.5-5 hours
Slow worst case: 5-8 hours
```

If it seems stuck during quality audit or feature extraction, check activity:

```bash
top
df -h
```

### 2.4 Phase 2 Success Files

After the notebook finishes:

```bash
ls -lh reports/tables/coswara_layout_audit.csv
ls -lh reports/tables/validation_issues.csv
ls -lh data/interim/coswara_index.csv
ls -lh data/interim/split_manifest.csv
ls -lh data/processed/metadata_clean.csv
ls -lh data/processed/audio_quality.csv
ls -lh data/processed/features_mfcc.csv
ls -lh data/outputs/metrics/ml_baseline_metrics.csv
ls -lh data/outputs/metrics/calibration_metrics.csv
ls -lh data/outputs/metrics/fusion_metrics.csv
```

Preview:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

paths = [
    "reports/tables/coswara_layout_audit.csv",
    "reports/tables/validation_issues.csv",
    "data/interim/coswara_index.csv",
    "data/interim/split_manifest.csv",
    "data/processed/metadata_clean.csv",
    "data/processed/audio_quality.csv",
    "data/processed/features_mfcc.csv",
    "data/outputs/metrics/ml_baseline_metrics.csv",
    "data/outputs/metrics/calibration_metrics.csv",
    "data/outputs/metrics/fusion_metrics.csv",
]

for path in paths:
    p = Path(path)
    print("\n==", path, "==")
    if not p.exists():
        print("MISSING")
        continue
    df = pd.read_csv(p)
    print("shape:", df.shape)
    print(df.head(3).to_string(index=False))
PY
```

### 2.5 Stop Conditions

Stop and send the failed cell/traceback if:

- labels are mostly `unknown`;
- positive or negative class is missing in train/validation/test;
- participant leakage check fails;
- most audio files are corrupt or silent;
- `features_mfcc.csv` is not produced;
- `ml_baseline_metrics.csv` is missing;
- `calibration_metrics.csv` is missing;
- `fusion_metrics.csv` is missing.

Do not enable CNN, COUGHVID, SSL, or GRL to fix Phase 2 failures.

## Phase 3: BTP Strong Extras Without COUGHVID

Run this only after Phase 2 succeeds.

Notebook settings:

```python
FORCE_REBUILD = False
RUN_PUBLICATION_EXTRAS = True

RUN_METADATA_BASELINE = True
RUN_QUALITY_WEIGHTED_FUSION = True
RUN_ABSTENTION = True
RUN_BOOTSTRAP_CI = True
RUN_PAPER_TABLES = True
RUN_PAIRED_MODEL_COMPARISON = True
RUN_CONFOUNDING_MATCHING = True
RUN_EXPERIMENT_MANIFEST = True

COUGHVID_RAW = None
RUN_COUGHVID_INDEX = False
RUN_COUGHVID_FEATURES = False
RUN_CROSS_DATASET = False
RUN_FEATURE_SHIFT_REPORT = False
RUN_CNN = False
```

Run all cells again. Because `FORCE_REBUILD = False`, existing artifacts should be reused.

Expected time:

```text
20-90 minutes
```

Success files:

```bash
ls -lh data/outputs/metrics/metadata_baseline_metrics.csv
ls -lh data/outputs/metrics/quality_weighted_fusion_metrics.csv
ls -lh data/outputs/metrics/abstention_coverage_curve.csv
ls -lh data/outputs/metrics/quality_weighted_fusion_bootstrap_ci.csv
ls -lh reports/tables/paired_model_comparison.csv
ls -lh reports/tables/confounding_balance.csv
ls -lh reports/tables/paper_metric_table.csv
ls -lh reports/manifests/
```

If metadata-only performance is close to or better than audio performance, report it honestly as confounding evidence.

## Phase 4: Optional Compact CNN

Run only after Phases 2 and 3 succeed.

Install PyTorch using the official PyTorch selector for your environment. Then verify:

```bash
source .venv/bin/activate
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

Notebook settings:

```python
RUN_CNN = True
CNN_BATCH_SIZE = 8
CNN_EPOCHS = 10
FORCE_REBUILD = False
```

Expected time:

```text
T1000 8 GB: 1-4 hours
Better GPU: 30 minutes-2 hours
CPU-only CNN: not recommended
```

If CUDA memory fails:

```python
CNN_BATCH_SIZE = 4
```

If it still fails:

```python
CNN_BATCH_SIZE = 2
```

CNN is optional. Do not let it block the BTP.

## Phase 5: COUGHVID Cough-Only External Validation

Run only after Coswara succeeds.

Important:

```text
COUGHVID is cough-only. It cannot validate cough+breath+speech multimodal fusion.
```

Create the folder:

```bash
mkdir -p data/raw/coughvid
```

For the exact Zenodo v1 record you shared, download:

```bash
curl -L -f -o data/raw/coughvid/public_dataset.zip \
  "https://zenodo.org/records/4048312/files/public_dataset.zip?download=1"
```

Zenodo also points to a newer v3 record. If you choose the latest v3 data instead, download:

```bash
curl -L -f -o data/raw/coughvid/public_dataset_v3.zip \
  "https://zenodo.org/records/7024894/files/public_dataset_v3.zip?download=1"
```

Use one of these paths in the commands below:

```text
data/raw/coughvid/public_dataset.zip
data/raw/coughvid/public_dataset_v3.zip
```

Build index first. For v1:

```bash
python scripts/13_build_coughvid_index.py \
  --raw-dir data/raw/coughvid/public_dataset.zip \
  --output data/interim/coughvid_index.csv \
  --min-cough-detected 0.8
```

For v3, use:

```bash
python scripts/13_build_coughvid_index.py \
  --raw-dir data/raw/coughvid/public_dataset_v3.zip \
  --output data/interim/coughvid_index.csv \
  --min-cough-detected 0.8
```

The code supports both direct zip sidecar JSON and `metadata_compiled.csv` inside the zip.

Expected time:

```text
5-30 minutes
```

Smoke-test features:

```bash
python scripts/19_extract_coughvid_features.py \
  --index data/interim/coughvid_index.csv \
  --features-output data/processed/coughvid_features_mfcc.csv \
  --quality-ok-only \
  --max-rows 25
```

Expected time:

```text
5-20 minutes
```

If `.webm` or `.ogg` decoding fails:

```bash
sudo apt update
sudo apt install ffmpeg libsndfile1
```

Full feature extraction after smoke test:

```bash
python scripts/19_extract_coughvid_features.py \
  --index data/interim/coughvid_index.csv \
  --features-output data/processed/coughvid_features_mfcc.csv \
  --quality-ok-only
```

Expected time:

```text
several hours to overnight
```

Cross-dataset evaluation:

```bash
python scripts/18_cross_dataset_feature_eval.py \
  --source-features data/processed/features_mfcc.csv \
  --external-features data/processed/coughvid_features_mfcc.csv \
  --modality cough \
  --model-name logistic_regression

python scripts/23_feature_shift_report.py \
  --source-features data/processed/features_mfcc.csv \
  --external-features data/processed/coughvid_features_mfcc.csv

python scripts/20_make_paper_tables.py
```

Success files:

```bash
ls -lh data/outputs/metrics/cross_dataset_metrics.csv
ls -lh data/outputs/metrics/cross_dataset_predictions.csv
ls -lh reports/tables/feature_shift_report.csv
```

If external performance collapses, do not hide it. That may be the main reliability finding.

## Phase 6: Final BTP Packaging

Generate final tables and manifest:

```bash
python scripts/20_make_paper_tables.py
python scripts/24_make_experiment_manifest.py --run-name final_btp_run
```

Required final artifacts:

```text
reports/tables/coswara_layout_audit.csv
reports/tables/validation_issues.csv
data/interim/coswara_index.csv
data/interim/split_manifest.csv
data/processed/metadata_clean.csv
data/processed/audio_quality.csv
data/processed/features_mfcc.csv
data/outputs/metrics/ml_baseline_metrics.csv
data/outputs/metrics/calibration_metrics.csv
data/outputs/metrics/fusion_metrics.csv
reports/tables/paper_metric_table.csv
reports/figures/
reports/manifests/
```

Recommended final presentation sections:

1. Problem and dataset difficulty.
2. Dataset layout and label distribution.
3. Leakage-safe participant split.
4. Quality audit results.
5. Feature extraction and baseline models.
6. Calibration and reliability.
7. Fusion comparison.
8. Metadata/confounding baseline.
9. COUGHVID cough-only external validation if completed.
10. Demo/output screenshots.
11. Limitations and future SSL/domain-proxy extension.

Allowed claim:

```text
We built a reproducible, leakage-safe, quality-aware and calibration-aware research framework for evaluating respiratory-audio COVID screening under real-world dataset limitations.
```

Do not claim:

```text
clinical diagnosis
guaranteed SOTA
COVID biomarkers without confounding analysis
full multimodal external validation on COUGHVID
```

## Phase 7: Advanced Publication Upgrade

Do not start until Phases 2-5 are reviewed.

Future modules:

```text
25_acoustic_domain_proxy.py
26_extract_ssl_embeddings.py
27_train_adversarial_ssl.py
28_compare_adversarial_vs_baselines.py
```

For the T1000:

```text
Start with frozen cough-only SSL embeddings.
Use batch size 1-4.
Train lightweight classifiers on saved embeddings.
Do not start with end-to-end GRL training.
```

Expected time:

```text
Frozen embeddings: several hours to overnight
Lightweight classifier: minutes
GRL ablation: later only if justified
```

## Common Errors And Mitigations

| Error / Symptom | Likely Cause | Action |
|---|---|---|
| `No module named covid_audio_btp` | package not installed editable | `pip install -e .` |
| `No module named pandas/librosa/sklearn` | dependencies missing | `pip install -r requirements.txt` |
| `Coswara not found` | wrong path | ensure `data/raw/coswara` exists |
| `combined_data.csv not found` | wrong folder level | Coswara repo root must be `data/raw/coswara` |
| `Coswara audio files discovered: 0` | extraction not run | run `python3 extract_data.py` in `data/raw/coswara` |
| labels mostly `unknown` | metadata mapping issue | stop and send `metadata_clean.csv` preview and traceback |
| participant leakage failure | split/index issue | stop and send `split_manifest.csv` and failed output |
| only one class in split | too few labeled samples or filtering too strict | stop; do not train |
| `features_mfcc.csv` missing | feature extraction failed | send traceback and `audio_quality.csv` |
| calibration fails | missing validation/test predictions | rerun ML baseline stage or send metrics files |
| fusion file missing | calibration output missing | check `calibrated_branch_predictions.csv` |
| Jupyter kernel missing | kernel not registered | rerun the ipykernel install command |
| wrong Python in notebook | wrong kernel selected | select `COVID Audio BTP` |
| CUDA out of memory | CNN batch too large | set `CNN_BATCH_SIZE = 4`, then `2` |
| COUGHVID codec failure | missing codec support | `sudo apt install ffmpeg libsndfile1` |
| run is slow | audio feature extraction is IO/CPU heavy | check `top`; wait if active |
| disk full | artifacts/datasets large | check `df -h`; clear only disposable cache |

## What To Send Back

After Phase 2:

```text
reports/tables/coswara_layout_audit.csv
reports/tables/validation_issues.csv
data/interim/coswara_index.csv
data/interim/split_manifest.csv
data/processed/metadata_clean.csv
data/processed/audio_quality.csv
data/outputs/metrics/ml_baseline_metrics.csv
data/outputs/metrics/calibration_metrics.csv
data/outputs/metrics/fusion_metrics.csv
```

After Phase 3:

```text
data/outputs/metrics/metadata_baseline_metrics.csv
data/outputs/metrics/quality_weighted_fusion_metrics.csv
data/outputs/metrics/abstention_coverage_curve.csv
data/outputs/metrics/quality_weighted_fusion_bootstrap_ci.csv
reports/tables/paired_model_comparison.csv
reports/tables/confounding_balance.csv
reports/tables/paper_metric_table.csv
reports/manifests/<latest_manifest_file>
```

If anything fails:

```text
1. failed notebook cell text or screenshot
2. full traceback
3. command you ran
4. current notebook settings cell
5. generated CSVs from the current phase, if any
```

## Final Exact Order

```text
1. Clone project.
2. Add dataset ignores.
3. Clone Coswara into data/raw/coswara.
4. Run Coswara extract_data.py.
5. Create venv.
6. Install requirements.
7. Install package editable.
8. Register Jupyter kernel.
9. Run mandatory preflight.
10. Run notebook with conservative first-run settings.
11. Confirm Phase 2 success files.
12. Enable publication extras without COUGHVID.
13. Confirm Phase 3 success files.
14. Optionally run compact CNN.
15. Optionally run COUGHVID cough-only validation.
16. Generate paper tables and manifest.
17. Prepare BTP presentation/report.
18. Send artifacts back for interpretation and next publication decision.
```

If you follow this order, you should either get the expected artifacts or have one precise failure point to send back.
