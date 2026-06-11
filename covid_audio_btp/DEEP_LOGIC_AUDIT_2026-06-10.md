# Deep Logic Audit - 2026-06-10

This audit was done after the first local run failed and after the code was reworked for local-first execution.

## Bottom Line

The current codebase is now a defensible first-run research pipeline, not a guaranteed SOTA model. It is suitable for producing a rigorous Coswara baseline with leakage-safe participant splits, quality auditing, classical ML baselines, calibration, and fusion.

Current SOTA-comparable claims require evidence from real runs plus stronger model ablations. Do not claim SOTA from the baseline alone.

## Current Literature Reality Check

Recent evidence suggests that rigorous COVID audio detection is much harder than older high-accuracy claims imply. A 2025 technical report fine-tuning Audio-MAE and PANN models on Coswara/COUGHVID reports moderate intra-dataset performance and severe cross-dataset degradation under demographic controls. It reports Audio-MAE around 0.82 AUC on Coswara, COUGHVID around 0.58-0.63 AUC, and cross-dataset AUC around 0.43-0.68. Source: https://arxiv.org/abs/2511.14939

A June 2026 arXiv paper reports much higher multi-dataset cough classification numbers using Whisper/OPERA-style pretrained backbones, active-frame pooling, imbalance handling, mixup/contrastive/domain adaptation, and multi-dataset training. That is not the same setting as a Coswara-only classical baseline. Source: https://arxiv.org/abs/2606.02998

Implication: strong BTech work should frame the current pipeline as a rigorous baseline/control and add SSL/pretrained-audio ablations only after the baseline is stable.

## Local Dataset Flow Audited

Expected layout:

```text
Covid-19-BTP/
  data/raw/coswara/
    extract_data.py
    combined_data.csv
    annotations/
    Extracted_data/
```

The official Coswara `extract_data.py` creates `Extracted_data` under the Coswara repo clone. The project indexer recursively scans `data/raw/coswara`, so extracted audio files are found. Source: https://github.com/iiscleap/Coswara-Data

## Logic Fixes Applied During This Audit

- Added real runtime test environment and ran the project tests with installed core dependencies.
- Fixed pandas 3.0 incompatibility in metadata-only baseline feature conversion.
- Changed quality reason codes to stable simple codes: `duration`, `silence`, `clipping`.
- Corrected stale center-padding test expectation. Center-padding is the intended fixed-length behavior after active-event extraction.
- Added demographic-aware participant splitting. The splitter now attempts stratification by `label_binary + age_bucket + gender`, then `label_binary + age_bucket`, then label-only fallback.
- Expanded split manifest with `age_bucket`, `gender`, and `split_stratify_group`.
- Added a regression test for demographic split columns and leakage safety.

## Verification Evidence

Commands run in hidden audit virtualenv:

```text
.audit_venv/bin/python -m pytest -q
Result: 43 passed

.audit_venv/bin/python scripts/00_check_environment.py
Result: environment check passed; optional xgboost/torch/streamlit absent only as warnings

.audit_venv/bin/python scripts/00_local_preflight.py --project-root . --coswara-dir .audit_runs/synthetic_coswara
Result: preflight passed
```

Synthetic Coswara-style end-to-end CLI smoke test passed through:

```text
00_inspect_dataset_layout.py
01_build_coswara_index.py
02_clean_metadata.py
03_create_splits.py
04_quality_audit.py
12_validate_artifacts.py --strict
05_extract_features.py --skip-spectrograms
06_train_ml_baselines.py
08_calibrate_branches.py
09_run_fusion.py
```

The synthetic dataset had 24 participants, 72 wav files, 3 modalities, balanced positive/negative labels, and a Coswara-like `combined_data.csv`.

## Methodological Decisions That Are Correct

- Participant-level split prevents the same participant leaking across train/validation/test.
- Quality audit runs before modeling.
- Feature extraction filters to supervised positive/negative rows.
- Calibration is fitted on validation predictions and applied to test predictions.
- COUGHVID remains optional and cough-only. It must not be used as full multimodal external validation.
- First run keeps CNN and publication extras disabled.

## Remaining Limits

- Real Coswara run is still required. Synthetic data proves code wiring, not research performance.
- Current baseline is not SOTA architecture. It is a strong, auditable baseline.
- SOTA-comparable extension needs frozen SSL/pretrained audio embeddings or a carefully constrained Whisper/PANN/AudioMAE-style ablation.
- Cross-dataset generalization cannot be claimed until COUGHVID cough-only evaluation is run.
- Clinical or diagnostic claims must not be made. This is screening/research only.

## Recommended Next Research Upgrade After Baseline Results

1. Run Coswara baseline and send the artifact tables.
2. Enable publication extras only after the baseline passes.
3. Run COUGHVID cough-only external validation.
4. Add frozen pretrained embeddings as ablation, not replacement:
   - PANN CNN14 embeddings or AudioMAE/Whisper encoder embeddings.
   - batch size small for T1000 8GB.
   - no full fine-tuning initially.
5. Compare classical MFCC baseline versus pretrained embeddings under the same participant/demographic split policy.

## Result Target Framing

Acceptable strong project target:

```text
A rigorous, calibrated, confounding-aware baseline plus evidence showing where it succeeds/fails under domain shift.
```

Unsafe target:

```text
Claiming SOTA or clinical COVID diagnosis from a single Coswara run.
```
