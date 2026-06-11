# Local GPU Handoff Checklist

This file is for the machine where experiments will actually run.

## What To Download

Required:

```text
Coswara dataset
https://github.com/iiscleap/Coswara-Data
```

Place it here after setup:

```text
<PROJECT_ROOT>/data/raw/coswara
```

Optional after the Coswara pipeline works:

```text
COUGHVID
https://zenodo.org/records/7024894
```

Place it here only for external cough validation:

```text
<PROJECT_ROOT>/data/raw/coughvid
```

Do not download mutation, genomic, wet-lab, variant-prediction, or sequence-analysis data for this audio-only build.

## First Setup

```bash
cd <PROJECT_ROOT>
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
jupyter lab
```

Open this first:

```text
notebooks/00_MASTER_RUN_ALL.ipynb
```

## Master Notebook Run Policy

Keep this flag false at first:

```python
RUN_CNN = False
```

Run the master notebook through these gates before any CNN work:

```text
0. Environment check
0A. Raw Coswara layout audit
1. Dataset index
2. Metadata cleaning and modality availability
3. Participant-level split
4. Quality and active-event audit
4A. Artifact validation gate
5. Feature and spectrogram extraction
6. Classical ML baselines
```

Only after those stages look correct, switch `RUN_CNN = True` on the GPU machine.

## Stop Conditions

Stop and inspect before continuing if any of these happen:

```text
Coswara layout audit shows unexpected folder structure
metadata labels are mostly unknown
positive or negative labels are missing
participant leakage assertion fails
validation gate raises an error
quality audit marks too many files corrupt or silent
ML models do not beat dummy baselines
fusion is worse than the best single modality
```

## Expected Early Artifacts

```text
reports/tables/coswara_layout_audit.csv
data/interim/coswara_index.csv
data/processed/metadata_clean.csv
data/interim/modality_availability.csv
data/interim/split_manifest.csv
data/processed/audio_quality.csv
reports/tables/validation_issues.csv
```

## Privacy Reminder

Use this after moving or copying the project:

```bash
chmod -R go-rwx <PRIVATE_CODEX_WORKSPACE>
```

This project now uses the persistent hidden workspace folder, not `/tmp`. Keep backups of final code, reports, and trained models when needed.
