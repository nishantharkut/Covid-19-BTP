# Implementation Status

Last updated: 2026-06-11

## Current State

The project now has a private, notebook-first but module-backed implementation scaffold.

Location:

```text
<PROJECT_ROOT>
```

Privacy:

```text
chmod -R go-rwx <PRIVATE_CODEX_WORKSPACE>
```

has been applied after code generation.

## Completed

### COUGHVID Zenodo Zip Support Hardened On 2026-06-11

Updated the COUGHVID adapter to support both the Zenodo v1 `public_dataset.zip` sidecar-JSON layout and v3-style `metadata_compiled.csv` inside `public_dataset_v3.zip`. Added a regression test for v3 metadata inside a direct zip archive. Targeted publication-layer tests passed, and full regression tests passed with 44 tests.

### Full Local Runbook Added On 2026-06-11

Added `BTP_FULL_LOCAL_RUNBOOK.md` as the canonical first-run-to-final-BTP local execution guide. It includes setup commands, Coswara clone/extract flow, preflight, notebook toggles, expected timings, success files, stop conditions, COUGHVID cough-only extension, CNN guidance, final BTP packaging, and common error mitigations.

### BTP And Publication Target Added On 2026-06-10

Added `BTP_A_GRADE_AND_PUBLICATION_TARGET.md` to lock the project priorities:

- primary goal is an A-grade BTP with a reproducible, working, defensible pipeline;
- publication strength is a staged extension after real baseline results exist;
- Gemini/PDF critiques are integrated as audit requirements and future ablations, not as a risky replacement for the baseline;
- the lab T1000 8 GB GPU is treated as suitable for baseline/CNN/frozen embeddings, not full end-to-end SSL/GRL fine-tuning;
- stop conditions are explicit so weak or confounded results are handled as research findings instead of hidden.

Notebook-first upgrade completed on 2026-05-25:

- Rebuilt `00_MASTER_RUN_ALL.ipynb` as the primary ordered execution notebook with artifact dashboard, raw data gate, validation gate, and `RUN_CNN` switch.
- Rebuilt stage notebooks `01` through `07` as real review notebooks with artifact checks, saved tables, saved figures, and explicit stop conditions.
- Expanded `src/covid_audio_btp/notebook_utils.py` with shared notebook artifact, gate, and table helpers.
- Added `tests/test_notebook_utils.py` for the new helper behavior.


- Master project guide.
- Notebook-first run guide.
- Local GPU handoff checklist.
- Verified source and code registry.
- Stage 1 and Stage 2+ runbooks.
- Source package under `src/covid_audio_btp`.
- Scripts `00` through `12`, including raw-layout inspection and artifact validation.
- Master run-all notebook with layout audit and validation gate.
- Stage review notebooks.
- Tests for core research logic.
- Demo skeleton for upload, waveform, quality, active-event window, and spectrogram.

## Most Important Design Choice

The project uses saved artifacts between stages:

```text
raw data -> index CSV -> clean metadata CSV -> split manifest -> quality CSV -> features/spectrograms -> predictions -> calibrated predictions -> fusion metrics
```

This prevents hidden notebook-state bugs and lets you restart Jupyter safely.

## Corrections Applied After Migration

- Project moved from temporary `/tmp` path to persistent hidden workspace path.
- Notebook helper now resolves project root dynamically from the installed package.
- Setup docs now include `pip install -e .` for the `src/` layout.
- Coswara indexing now prefers known participant IDs from metadata when inferring participant IDs from nested paths.
- Manual quality fields such as `cough_heavy_quality` are preserved as `manual_quality_score` and `manual_quality_label`.
- `split.py` re-exports modality availability for existing tests/notebook imports.
- `preprocess.py` exposes `crop_or_pad_audio` as a compatibility wrapper around center crop/pad.

## What Still Needs Real-Data Validation

These can only be verified after Coswara is placed under `data/raw/coswara`:

- Whether participant IDs are inferred correctly from the actual folder layout.
- Whether Coswara metadata CSV column names are detected correctly.
- Whether positive/negative labels are mapped correctly.
- Whether every modality is found with expected names.
- Whether audio files are readable by `soundfile` or `librosa`.
- Whether the final split has enough positives/negatives per split.
- Whether CNN input shapes are consistent for all modalities.

## First Real Run Gate

Do not train models until these checks pass:

```text
data/interim/coswara_index.csv exists
data/processed/metadata_clean.csv exists
data/interim/modality_availability.csv exists
data/interim/split_manifest.csv exists
data/processed/audio_quality.csv exists
participant leakage check passes
positive and negative labels both exist
quality audit has acceptable non-corrupt coverage
```

## Recommended Next Human Action

Place Coswara data under:

```text
<PROJECT_ROOT>/data/raw/coswara
```

Then run:

```bash
cd <PROJECT_ROOT>
source .venv/bin/activate
python scripts/00_check_environment.py
jupyter lab
```

If the first run fails, share the failed master-notebook cell output and these files if they exist:

```text
reports/tables/coswara_layout_audit.csv
reports/tables/validation_issues.csv
```


## Decision Lock Added On 2026-06-03

Added `DECISION_LOCK_AND_LOCAL_RUN_PROTOCOL.md` to freeze the current execution strategy:

- datasets are expected on the user's local machine, not this EC2 workspace;
- run Coswara baseline first;
- keep `RUN_CNN = False`, `COUGHVID_RAW = None`, and `RUN_COUGHVID_FEATURES = False` for the first run;
- COUGHVID is cough-only and should be enabled only after Coswara succeeds;
- SSL/HuBERT/Wav2Vec/GRL work is a later ablation layer, not a replacement for the baseline;
- compute requirements are documented for CPU baseline, optional CNN, and future SSL/GRL.

## Advanced Extension Spec Added On 2026-06-03

Added `ADVANCED_EXTENSION_FROM_GEMINI_PDFS.md` to preserve the useful parts of the two reviewed PDFs and Gemini conversation without replacing the current baseline pipeline. It records the post-baseline plan for acoustic-domain proxy labels, SSL embeddings, GRL/adversarial training, and advanced ablation comparisons.

## Publication-Grade Implementation Slice Added On 2026-05-25

Added the first Q-CalFuse publication layer:

- COUGHVID external dataset adapter.
- Cross-dataset feature harmonization utility.
- Bootstrap confidence interval utility.
- Metadata/symptom-only baseline model.
- Quality-weighted calibrated fusion.
- Uncertainty/quality abstention and coverage curve utilities.
- Scripts `13` through `18` for publication experiments.
- Focused tests in `tests/test_publication_layer.py`.
- Runbook: `RUNBOOK_PUBLICATION_GRADE.md`.

Validation performed here:

- Python syntax compilation passed for all new modules, scripts, and tests.

Validation not performed here:

- Runtime pytest execution, because this EC2 image does not currently have `numpy` or `pytest` installed. Run `pip install -r requirements.txt && pip install -e .` first, then run `pytest tests/test_publication_layer.py -q`.

## Dataset Schema Inspection And Jupyter Publication Layer Added On 2026-05-26

Temporary official dataset inspection was performed under `/tmp` only. Findings were saved to:

```text
research_protocol/2026-05-26-dataset-schema-inspection.md
```

Code updates completed:

- Coswara metadata indexing now supports official short columns (`a`, `g`, `l_c`, `record_date`, `testType`, `test_status`).
- Coswara symptom and comorbidity fields are preserved as JSON columns for metadata baselines and confounding checks.
- COUGHVID indexing now supports extracted sidecar JSON layout, direct `public_dataset.zip` inspection, CSV metadata, and v3-style `status_SSL` metadata.
- Audio loading now supports `archive.zip::member` paths through temporary local materialization.
- Added `notebooks/08_publication_grade_experiments.ipynb` for the Jupyter-first publication experiment layer.
- Added regression tests for official Coswara short metadata, COUGHVID sidecars, direct COUGHVID zip indexing, `status_SSL` metadata, and zip-member materialization.

## External Feature And Paper Table Layer Added On 2026-05-26

Added the next publication workflow slice:

- `src/covid_audio_btp/external_features.py` prepares external indexes with `split=external`, optional binary-label filtering, and optional quality filtering.
- `scripts/19_extract_coughvid_features.py` extracts COUGHVID MFCC/acoustic features from `data/interim/coughvid_index.csv`, with `--max-rows` for smoke testing.
- `src/covid_audio_btp/reporting.py` now includes `build_paper_metric_table` and `read_existing_csvs` for manuscript-style metric tables.
- `scripts/20_make_paper_tables.py` consolidates available metric and bootstrap CI CSVs into `reports/tables/paper_metric_table.csv`.
- `notebooks/08_publication_grade_experiments.ipynb` now includes COUGHVID feature extraction and paper metric table cells.
- Regression tests were added for external metadata preparation and paper metric formatting.

## Single Notebook Entrypoint Added On 2026-05-26

Added:

```text
notebooks/00_RUN_EVERYTHING_PUBLICATION.ipynb
```

This is now the recommended first notebook. It has 20 cells total: 10 code cells and 10 markdown cells. It runs the full publication workflow through toggles and skips existing artifacts unless `FORCE_REBUILD = True`.

The previous notebooks remain available as optional review/debug notebooks.

## Publication Strengthening Diagnostics Added On 2026-05-26

Added optional high-rigor analysis helpers:

- `src/covid_audio_btp/model_comparison.py` and `scripts/21_paired_model_comparison.py` for paired bootstrap model differences.
- `src/covid_audio_btp/confounding.py` and `scripts/22_confounding_matching.py` for coarsened exact matching and covariate balance tables.
- `src/covid_audio_btp/shift.py` and `scripts/23_feature_shift_report.py` for source-vs-external feature shift diagnostics.
- `src/covid_audio_btp/manifest.py` and `scripts/24_make_experiment_manifest.py` for reproducibility manifests with package versions and artifact hashes.
- `tests/test_publication_strengthening.py` for these helpers.
- `notebooks/00_RUN_EVERYTHING_PUBLICATION.ipynb` now includes these optional steps and has 20 cells total: 10 code and 10 markdown.

