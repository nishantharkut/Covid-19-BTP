# DNDT/DNDF Two-Day Experiment Design

**Date:** 2026-08-26
**Project:** COVID-RARS
**Status:** Revised for a two-day execution limit

## 1. Objective

Add Deep Neural Decision Tree (DNDT) and Deep Neural Decision Forest (DNDF) to COVID-RARS without repeating the overly broad HST implementation. The experiment must answer two separate questions:

1. Can the published ESWA method be reproduced under its own protocol and metric definitions?
2. When DNDT/DNDF are placed inside the existing COVID-RARS data and validation pipeline, do they improve the current single-modality or multimodal results, and do they remain reliable under temporal and external shift?

The design does not promise an AUROC improvement. It makes a genuine improvement measurable and prevents a weak or invalid result from being reported as an improvement.

## 2. Scope Control

### Required

- DNDT and DNDF implementations checked against the published equations and author code.
- Exact author-data reproduction using the author's released arrays and ten folds.
- Project integration for breath, cough, and speech.
- A validation-trained cough+speech fusion, matching the modality combination behind the current best internal result.
- Existing participant, time-stratified, early-to-late, and cough-only external evaluations.
- The complete established classification and calibration metric bundle.
- Prediction files, configuration, hashes, checkpoints, and a compact evidence summary.
- Resume after interruption without restarting completed folds, seeds, or protocols.

### Explicitly not required

- Re-extracting raw audio features.
- Running HST, DNDT/DNDF, or classical-model architecture searches together.
- Repeating all prior confounding, matching, subgroup, CORAL, or metadata experiments.
- Five or more seeds for every candidate.
- More than two fusion rules.
- Reverse temporal evaluation.
- New datasets.
- Any dependence on the incomplete DNDT/DNDF pull request.

This boundary is the main runtime safeguard.

## 3. Evidence Sources

The implementation will be grounded in:

- Islam, Chowdhury, and Kabir, *Robust COVID-19 detection from cough sounds using deep neural decision tree and forest: A comprehensive cross-datasets evaluation*, Expert Systems with Applications, 2026.
- Author repository pinned at the verified commit `feb0e63c790c042eaa21e9f3fc83ed64bdc8a24e`. The preflight must fail if the recorded commit does not match the actual clone.
- Existing COVID-RARS feature tables, split definitions, predictions, and metric functions.

The teammate pull request is not an implementation source. It may be retained as project history but is not trusted for model mathematics or evaluation.

## 4. Two Experimental Tracks

### 4.1 Track A: Published-protocol reproduction

Track A uses only the author's released Coswara feature arrays, labels, and ten folds. It does not mix project data with author data.

The released input has 193 acoustic variables:

- 40 mel-frequency cepstral coefficient values
- 128 Mel-spectrum values
- 6 tonal centroid values
- 12 chromagram values
- 7 spectral-contrast values

The first run reproduces the reported Coswara configurations:

| Model | Trees | Depth | Feature rate | Learning rate | Batch size | Epochs |
|---|---:|---:|---:|---:|---:|---:|
| DNDT | 1 | 11 | 0.6 | 0.01 | 16 | 14 |
| DNDF | 25 | 11 | 0.6 | 0.01 | 16 | 14 |

Two evaluation modes are retained because the author code contains evaluation choices that are unsuitable for a modern reliability study:

1. **Author-behaviour audit:** reproduces model carry-over and test-fold threshold selection where present, solely to explain the published number.
2. **Corrected reference evaluation:** creates a fresh model per fold and selects its threshold using an inner validation subset. The outer fold is touched once for evaluation.

The author's globally prepared feature arrays cannot be retrospectively unscaled. Track A therefore documents that limitation and does not present the corrected mode as a fully leakage-free raw-audio reproduction.

The paper's thresholded-label ROC quantity is written to a separate field named `paper_thresholded_auc`. The normal `auroc` field always means AUROC calculated from continuous predicted probabilities.

### 4.2 Track B: COVID-RARS integration

Track B reuses the existing top-800 ComParE2016+IS10 feature table. No raw-audio extraction or fresh 193-feature pipeline is required. This is intentional: Track A tests fidelity to the ESWA representation, while Track B isolates whether the DNDT/DNDF learner adds value to the established COVID-RARS representation.

Required branches are:

- breath DNDT and DNDF
- cough DNDT and DNDF
- speech DNDT and DNDF
- cough+speech DNDF stacked fusion
- cough+breath+speech DNDF uniform-mean sensitivity result

The cough+speech stack is the primary multimodal branch. Its logistic fusion weights and intercept are learned on validation predictions only. They are never hard-coded and are frozen before test evaluation.

The three-modality uniform mean is secondary. Its weights are fixed at one third per available modality, after aligning predictions by participant. It is included because that rule has already been used in the project and requires no target-driven fitting.

Missing modalities are handled only by the established participant-alignment rule. A participant is never duplicated to manufacture complete modality coverage.

## 5. Model Selection Without an Open-Ended Search

The selection sequence is bounded:

1. Run the published DNDT and DNDF configurations once per modality with one seed.
2. Select using validation AUROC, with validation AUPRC as the tie-breaker.
3. If the published DNDF configuration is not better than DNDT on validation, allow at most 6 Optuna trials for DNDF for that modality.
4. Search only depth, number of trees, feature rate, learning rate, and regularization. No feature extraction or preprocessing choices enter this search.
5. Use a persistent SQLite study so interrupted trials resume.
6. Never use test, temporal-test, or external labels for configuration selection.

The final DNDF configuration for each modality is then rerun with three fixed seeds. DNDT remains a one-seed comparator unless it wins validation selection.

This replaces the earlier exhaustive model-by-protocol-by-seed grid.

## 6. Validation Protocols

### Track A

- The author's ten released folds.
- Author-behaviour and corrected-reference modes reported separately.

### Track B

1. **Existing participant split:** the established train, validation, and held-out participant sets.
2. **Time-stratified participant split:** participants remain separated while calendar periods are represented across partitions.
3. **Early-to-late temporal split:** train on earlier recordings and test on later recordings.
4. **External cough transfer:** train and select using Coswara cough only, then evaluate the frozen cough branch on COUGHVID.

The same selected model lineage must be carried through a validation ladder. The implementation must not pick a different best model independently at each rung and then call their difference a reliability loss.

COUGHVID is a cough-only external target. It evaluates portability of the cough branch, not external transfer of the full multimodal fusion.

## 7. Preprocessing and Leakage Rules

- Read the existing feature tables from `G:\Covid-19-BTP\covid_audio_btp` without modifying them.
- Convert stored Ubuntu audio paths only in run-local metadata when required. Path resolution must remain inside the configured raw-data root.
- Fit imputation, scaling, and class balancing on training rows only. The project input columns remain the frozen top-800 feature set; no second feature-selection pass is introduced.
- Apply training-fitted transformations to validation and evaluation rows.
- Use participant identifiers for project split isolation.
- Use an inner validation subset for threshold and configuration selection.
- Freeze feature columns and order in the run manifest.
- Preserve the project label convention: COVID positive is `1`.
- Convert the author's `C/N` orientation explicitly and test it.
- SMOTE, if retained for parity, is applied only to transformed training data and never before splitting.

## 8. Metrics

Every evaluated prediction set reports:

- probability AUROC
- average precision, labelled AUPRC
- accuracy
- balanced accuracy
- F1 score
- sensitivity or recall
- specificity
- precision or positive predictive value
- negative predictive value
- true positives, false positives, true negatives, and false negatives
- Brier score
- expected calibration error
- maximum calibration error
- negative log likelihood
- selected threshold and threshold source
- samples, participants, positives, negatives, and prevalence
- AUPRC lift over prevalence

Definitions are fixed across scripts:

- AUPRC is `sklearn.metrics.average_precision_score`.
- Probabilities used in negative log likelihood are clipped to `[1e-6, 1-1e-6]`.
- ECE uses fixed probability bins with left-closed/right-open intervals; the last bin includes 1.0.
- Calibration gap is mean predicted probability minus observed prevalence.

For the final selected DNDF rows only:

- participant-level bootstrap 95% confidence intervals for AUROC, AUPRC, Brier score, and ECE
- paired AUROC/delta comparisons only when two models predict the same participants
- independent source-target bootstrap for Coswara-to-COUGHVID deltas
- specificity and precision at a validation-selected sensitivity target of at least 0.90
- decision-curve net benefit calculated as a post-processing analysis

Track A additionally reports `paper_thresholded_auc`. It is never substituted for probability AUROC.

## 9. Minimal Robustness Checks

Required robustness work is limited to:

- three seeds for the final DNDF modality branches and primary fusion
- one label-shuffle retraining check on the final internal DNDF configuration
- independent recomputation of final metrics from saved predictions

Existing project evidence for metadata confounding, temporal feature instability, subgroup behaviour, calibration, and support overlap is hash-registered in the final manifest but not rerun.

## 10. Runtime and Storage Design

### Local execution

- Current Windows machine is the primary runner.
- Code and the dedicated environment remain on `C:`.
- Large run artifacts are written to `G:\Covid-19-BTP\dndt_dndf_runs\<run-id>`.
- The existing project and datasets on `G:` are treated as read-only inputs.
- A new `.venv-dndt-dndf` is used so CUDA-enabled PyTorch can be installed without changing the established project environment.

Only one GPU training process runs at a time. CPU preparation, validation, and evidence generation may run alongside it if memory remains safe. Nested model-level multiprocessing is disabled to avoid oversubscription.

### Optional free cloud execution

Kaggle or Colab may run independent Track A folds or final seed jobs only after the local smoke test passes. A cloud result is accepted only if it records the same code revision, configuration hash, feature hash, and split hash. The two-day result does not depend on cloud availability.

## 11. Checkpoint and Resume Contract

Checkpointing is simple and task-oriented:

- Save `latest` and `best` state after every epoch.
- Save completion records after every fold, trial, seed, modality, and protocol.
- Write checkpoints atomically through a temporary file and rename.
- Resume only when code, configuration, features, labels, and split hashes match.
- Refuse a mismatched resume rather than silently continuing.
- Keep the latest and best checkpoint only after a task completes.

A single notebook/controller shows completed and pending units from the manifest, so one restart continues unfinished work rather than rerunning the pipeline.

## 12. Implementation Boundary

The implementation should remain small:

- `src/covid_rars/dndt_dndf_models.py` for DNDT/DNDF mathematics
- `src/covid_rars/dndt_dndf_experiment.py` for datasets, protocols, fitting, and resume
- `src/covid_rars/dndt_dndf_evidence.py` for metrics, confidence intervals, and tables
- one script for Track A
- one script for Track B
- one script for final evidence generation
- one compact configuration file with separate `track_a` and `track_b` sections
- one notebook that invokes the scripts and displays progress

No changes are required in existing production experiment scripts. New files call stable project utilities where their contracts are compatible.

## 13. Verification Gates

No full run starts until all gates pass:

1. A depth-two DNDT and a two-tree DNDF agree with a small NumPy reference calculation.
2. Synthetic data loss decreases and probabilities remain finite and in `[0,1]`.
3. Author-array sample counts, class counts, and fold partitions match the released artifacts.
4. Positive-class orientation is verified end to end.
5. Project protocols show zero participant overlap where participant IDs exist.
6. Preprocessing and threshold-selection tests prove training/validation isolation.
7. An intentionally interrupted tiny run resumes to the same result as an uninterrupted run.
8. Final CSV metrics match an independent sklearn recomputation from saved predictions.
9. A one-fold, one-epoch GPU pilot completes before the full queue begins.

Warnings about a true methodological mismatch fail the preflight. Benign library deprecation warnings are captured but do not stop the run.

## 14. Two-Day Execution Order

### Day 1

1. Build the isolated CUDA environment and run unit tests.
2. Run the tiny interruption/resume pilot.
3. Complete Track A DNDT and DNDF reproduction.
4. Run one-seed DNDT/DNDF candidates for breath, cough, and speech.
5. Launch bounded DNDF tuning only where the published configuration fails to win validation.

### Day 2

1. Run three seeds for final DNDF modality branches.
2. Fit and evaluate the primary cough+speech stack.
3. Evaluate time-stratified, early-to-late, and external cough protocols using frozen lineages.
4. Run the one label-shuffle control.
5. Generate confidence intervals, operating points, decision curves, and the final evidence summary.

If runtime becomes constrained, execution priority is:

1. Track A reproduction
2. Track B internal breath/cough/speech DNDT and DNDF
3. primary cough+speech DNDF fusion
4. early-to-late temporal evaluation
5. external cough transfer
6. time-stratified evaluation
7. secondary uniform-mean fusion

Metric post-processing is not dropped because it is inexpensive once predictions exist.

## 15. Completion Criteria

The experiment is complete when:

- Track A records both the paper-style and probability-based metrics.
- Track B records all three modalities and the primary multimodal fusion.
- Required validation protocols use frozen, documented lineages.
- Every reported row has saved participant- or analysis-unit predictions.
- Final rows have uncertainty estimates and operating-point results.
- Resume has been tested, not merely implemented.
- A machine-readable manifest records code, environment, configuration, features, splits, checkpoints, and outputs.
- Results are labelled honestly as improved, comparable, numerically lower, or inconclusive.

The scientific success condition is not a guaranteed higher AUROC. It is a correct answer to whether DNDT/DNDF improves the established pipeline under comparable and strict validation, produced within the available two-day window.
