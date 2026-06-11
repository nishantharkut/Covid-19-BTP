# Stage 1 Runbook

Run these commands after placing Coswara under `data/raw/coswara`.

```bash
cd <PROJECT_ROOT>
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts/00_check_environment.py
pytest
```

Inspect the raw dataset layout before indexing:

```bash
python scripts/00_inspect_dataset_layout.py \
  --raw-dir data/raw/coswara \
  --output reports/tables/coswara_layout_audit.csv
```

Build the first artifacts:

```bash
python scripts/01_build_coswara_index.py \
  --raw-dir data/raw/coswara \
  --output data/interim/coswara_index.csv

python scripts/02_clean_metadata.py \
  --index data/interim/coswara_index.csv \
  --output data/processed/metadata_clean.csv \
  --availability-output data/interim/modality_availability.csv

python scripts/03_create_splits.py \
  --metadata data/processed/metadata_clean.csv \
  --output data/interim/split_manifest.csv \
  --metadata-output data/processed/metadata_clean.csv

python scripts/04_quality_audit.py \
  --metadata data/processed/metadata_clean.csv \
  --output data/processed/audio_quality.csv \
  --metadata-output data/processed/metadata_clean.csv
```

Validate generated artifacts:

```bash
python scripts/12_validate_artifacts.py --strict
```

Expected files:

```text
data/interim/coswara_index.csv
data/processed/metadata_clean.csv
data/interim/modality_availability.csv
data/interim/split_manifest.csv
data/processed/audio_quality.csv
reports/tables/dataset_audit.csv
reports/tables/split_audit.csv
reports/tables/quality_summary.csv
reports/tables/validation_issues.csv
```

Do not train models until these files look correct.

