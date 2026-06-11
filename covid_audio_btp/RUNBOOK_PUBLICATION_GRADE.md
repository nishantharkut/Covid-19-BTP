# Publication-Grade Experiment Runbook

This runbook extends the notebook-first Coswara workflow with the experiments needed for the Q-CalFuse publication direction.

## Prerequisite

Install the project environment first:

```bash
cd <PROJECT_ROOT>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## New Code Added

Source modules:

- `src/covid_audio_btp/external_datasets.py` - COUGHVID metadata/index adapter.
- `src/covid_audio_btp/cross_dataset.py` - feature-column harmonization for external validation.
- `src/covid_audio_btp/statistics.py` - bootstrap confidence intervals.
- `src/covid_audio_btp/metadata_baseline.py` - metadata/symptom-only baseline.
- `src/covid_audio_btp/abstention.py` - uncertainty/quality abstention and coverage curves.
- `src/covid_audio_btp/fusion.py` - now includes quality-weighted calibrated fusion.

Tests:

- `tests/test_publication_layer.py`

Scripts:

- `scripts/13_build_coughvid_index.py`
- `scripts/14_train_metadata_baseline.py`
- `scripts/15_bootstrap_prediction_metrics.py`
- `scripts/16_run_quality_weighted_fusion.py`
- `scripts/17_abstention_analysis.py`
- `scripts/18_cross_dataset_feature_eval.py`
- `scripts/19_extract_coughvid_features.py`
- `scripts/20_make_paper_tables.py`
- `scripts/24_make_experiment_manifest.py`
- `scripts/23_feature_shift_report.py`
- `scripts/22_confounding_matching.py`
- `scripts/21_paired_model_comparison.py`


## Single Notebook Path

For normal local work, run only:

```text
notebooks/00_RUN_EVERYTHING_PUBLICATION.ipynb
```

It has 20 cells total: 10 code cells and 10 markdown cells. The notebook includes toggles for:

```text
RUN_CORE_COSWARA
RUN_FEATURES
RUN_ML_BASELINES
RUN_CALIBRATION
RUN_FUSION
RUN_CNN
RUN_PUBLICATION_EXTRAS
RUN_COUGHVID_INDEX
RUN_COUGHVID_FEATURES
RUN_CROSS_DATASET
RUN_PAPER_TABLES
```

The separate notebooks `01` through `08` are optional review/debug notebooks, not required execution steps.

## Recommended Order After Coswara Pipeline

First complete the normal notebook master pipeline through calibration/fusion.

Then run publication extras:

```bash
python scripts/14_train_metadata_baseline.py       --metadata data/processed/metadata_clean.csv

python scripts/16_run_quality_weighted_fusion.py       --predictions data/outputs/metrics/calibrated_branch_predictions.csv       --quality data/processed/audio_quality.csv       --validation-metrics data/outputs/metrics/ml_baseline_metrics.csv

python scripts/17_abstention_analysis.py       --predictions data/outputs/metrics/quality_weighted_fusion_predictions.csv       --metadata data/processed/metadata_clean.csv

python scripts/15_bootstrap_prediction_metrics.py       --predictions data/outputs/metrics/quality_weighted_fusion_predictions.csv       --output data/outputs/metrics/quality_weighted_fusion_bootstrap_ci.csv       --group-columns fusion_method
```


## Real Dataset Layouts Confirmed On 2026-05-26

A temporary inspection of official metadata/layouts was performed under `/tmp` only. The persistent project now contains the trace note:

```text
research_protocol/2026-05-26-dataset-schema-inspection.md
```

Coswara:

- Official `combined_data.csv` uses short columns such as `a` for age, `g` for gender, `l_c` for country, `record_date` for recording/submission date, and `testType` / `test_status` for testing fields.
- Date folders contain date-level CSVs and split tar archive parts such as `20200413.tar.gz.aa`.
- The indexer now preserves symptom and comorbidity fields into JSON columns for confounding analysis.

COUGHVID:

- The inspected official archive layout is `public_dataset/<uuid>.json` plus `.webm` / `.ogg` audio.
- The adapter also supports `metadata_compiled.csv` style metadata, including `status_SSL` when present.
- Direct zip indexing is supported with audio paths like `public_dataset.zip::public_dataset/<uuid>.webm`; this works through temporary materialization in `audio_io.load_audio`. For speed, extracting COUGHVID locally is still recommended for full feature extraction.

## Jupyter Publication Notebook

After the standard notebooks `00` through `07`, run:

```text
notebooks/08_publication_grade_experiments.ipynb
```

This notebook coordinates COUGHVID indexing, metadata-only baseline, quality-weighted fusion, abstention/coverage analysis, bootstrap confidence intervals, and optional cross-dataset cough evaluation.

## COUGHVID External Validation

Put COUGHVID temporarily or persistently at your chosen path. If you only want non-persistent inspection, use a temporary path outside the project, then copy only generated CSV artifacts you want to keep.

Build index:

```bash
python scripts/13_build_coughvid_index.py       --raw-dir /path/to/coughvid_or_public_dataset.zip       --output data/interim/coughvid_index.csv       --min-cough-detected 0.8
```

Extract external COUGHVID features after building the index. Start with a smoke test, then remove `--max-rows` for the full run:

```bash
python scripts/19_extract_coughvid_features.py       --index data/interim/coughvid_index.csv       --features-output data/processed/coughvid_features_mfcc.csv       --quality-ok-only       --max-rows 25
```

Full external feature extraction:

```bash
python scripts/19_extract_coughvid_features.py       --index data/interim/coughvid_index.csv       --features-output data/processed/coughvid_features_mfcc.csv       --quality-ok-only
```

After feature extraction exists for both source and external cough rows, run:

```bash
python scripts/18_cross_dataset_feature_eval.py       --source-features data/processed/features_mfcc.csv       --external-features data/processed/coughvid_features_mfcc.csv       --modality cough       --model-name logistic_regression
```


## Extra Publication-Strength Diagnostics

After the main metrics exist, these optional scripts add stronger evidence for a serious paper:

```bash
python scripts/21_paired_model_comparison.py       --predictions data/outputs/metrics/calibrated_branch_predictions.csv       --baseline-name logistic_regression       --model-column model_name       --group-columns modality

python scripts/22_confounding_matching.py       --metadata data/processed/metadata_clean.csv       --predictions data/outputs/metrics/quality_weighted_fusion_predictions.csv       --covariates age_bucket gender

python scripts/23_feature_shift_report.py       --source-features data/processed/features_mfcc.csv       --external-features data/processed/coughvid_features_mfcc.csv

python scripts/24_make_experiment_manifest.py       --run-name covid_audio_publication_run
```

These produce paired model comparison CIs, matched-cohort balance diagnostics, cross-dataset feature shift tables, and a reproducibility manifest with artifact hashes.

## Paper Tables

After metrics and bootstrap CI files exist, generate a consolidated report table:

```bash
python scripts/20_make_paper_tables.py
```

Outputs:

```text
reports/tables/paper_metric_table.csv
reports/tables/paper_metric_table_raw.csv
```

## Verification

In a prepared environment with dependencies installed:

```bash
pytest tests/test_publication_layer.py -q
```

On the current EC2 image, runtime tests could not be executed because `numpy` and `pytest` were not installed. Syntax compilation passed for all new files.
