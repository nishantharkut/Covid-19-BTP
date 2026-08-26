# DNDT/DNDF Final Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an auditable explanation of the ESWA paper-result gap and a complete DNDT/DNDF evaluation inside COVID-RARS without spending the remaining time on redundant experiments.

**Architecture:** Track A remains isolated to the authors' released Coswara arrays and folds. It adds one controlled fresh-model reproduction that changes only fold-to-fold model reuse, while a deterministic overlap audit quantifies global RFECV exposure without another expensive feature-selection run. Track B uses the frozen COVID-RARS top-800 features, validation-only selection, three modality branches, two predefined fusion rules, temporal and external validation, and a shuffle-label control. Evidence generation refuses partial or unauthenticated runs.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pandas, scikit-learn, imbalanced-learn, Optuna, pytest, PowerShell, OpenSSH, NVIDIA CUDA.

---

## Fixed Scientific Questions

1. Does the released-code result rise because the same neural model is trained continuously across nominal CV folds?
2. How much of every outer test fold was already exposed to label-driven global RFECV selection?
3. What performance remains when each fold receives a fresh model and all epoch and threshold decisions use inner validation only?
4. Do DNDT/DNDF improve the established COVID-RARS breath, cough, speech, and multimodal branches?
5. Do the selected branches retain performance under time-stratified, early-to-late, and Coswara-to-COUGHVID evaluation?

No experiment may claim that the paper's full result inflation is attributable to one mechanism unless that mechanism is changed independently. The released-code audit, carry-over ablation, and corrected reference are reported as three separate estimands.

## File Map

**Modify:**

- `src/covid_rars/dndt_dndf_experiment.py` - Track A ablation, Track B authentication, shuffle stage, manifest scope, checkpoint retention.
- `src/covid_rars/dndt_dndf_evidence.py` - strict completeness gate, consistent metric definitions, corrected model-selection table.
- `scripts/80_run_dndt_dndf_track_a.py` - expose the carry-over ablation mode.
- `scripts/81_run_dndt_dndf_track_b.py` - expose shuffle and retention options.
- `scripts/82_make_dndt_dndf_evidence.py` - invoke strict evidence validation.
- `notebooks/08_dndt_dndf_two_day_run.ipynb` - run all required stages with resume.
- `tests/test_dndt_dndf_experiment.py` - controller, isolation, authentication, resume, shuffle, retention tests.
- `tests/test_dndt_dndf_evidence.py` - completeness and evidence-schema tests.
- `tests/test_dndt_dndf_cli.py` - CLI and notebook contracts.

**Create at execution time, outside Git:**

- A Linux-local config under `/home/covid/Desktop/Covid-RARS-runtime/dndt-dndf/`.
- Run artifacts under `/home/covid/Desktop/Covid-RARS-runtime/dndt-dndf/runs/<run-id>`.

## Task 1: Close Track B Integrity Findings

- [ ] Add a failing evidence test proving that a missing, partial, or hash-invalid required manifest blocks publication evidence generation.
- [ ] Add a failing test proving `model_selection.csv` contains exactly breath, cough, and speech from `selected_configurations.json["modalities"]`.
- [ ] Add a failing candidate-receipt test that changes saved validation probabilities while updating only the stored metric and proves authentication rejects it after independent metric recomputation.
- [ ] Add a failing fusion-resume test proving a valid completed unit is loaded without rewriting artifacts and a tampered artifact is rejected.
- [ ] Add a failing manifest test proving cumulative scope is derived from authenticated receipts, while the latest request is recorded separately.
- [ ] Run each test once and confirm the expected failure before editing production code.
- [ ] Implement the smallest fixes, using `complete_metric_bundle` for immediate Track B and fusion metrics so final evidence cannot disagree with execution receipts.
- [ ] Run the focused tests after every fix.

## Task 2: Add the DNDF Shuffle-Retrain Control

- [ ] Add a failing test for a `shuffle` Track B stage.
- [ ] Permute labels once at participant level with fixed seed 42, preserving class counts and one label per participant.
- [ ] Apply the shuffled labels only to source train and validation rows used for training, early stopping, and threshold selection.
- [ ] Keep held-out test labels unchanged and unavailable to the trainer.
- [ ] Save the permutation hash, predictions, complete metrics, and an authenticated completion receipt.
- [ ] Add `shuffle` to script 81 and the notebook.
- [ ] Confirm the smoke fixture completes and produces chance-compatible output without asserting a favorable numeric result.

## Task 3: Bound Checkpoint Storage Without Weakening Resume

- [ ] Add a failing test that a completed Track B unit retains one authenticated best-inference checkpoint and removes superseded recovery generations only after its durable receipt exists.
- [ ] Add a failing interruption test proving incomplete units retain both current and previous recovery generations.
- [ ] Implement post-receipt compaction scoped strictly to that unit's checkpoint directory.
- [ ] Record retained and removed artifact names and hashes in a compaction receipt.
- [ ] Never compact Track A author-behaviour state until its complete model sequence is finished, because later folds depend on the carried model.

## Task 4: Add the Minimal Track A Causal Attribution

- [ ] Add a failing test for mode `fresh_fold_author_protocol`.
- [ ] In this mode, initialize a fresh DNDF for every released fold, train for the paper's fixed 14 epochs on that fold's complete released training split, retain the same globally selected author features, and select the paper-style threshold on the outer test fold.
- [ ] Confirm the only intended difference from `author_behaviour_audit` is model reinitialization and optimizer reset per fold.
- [ ] Add an RFECV exposure table that reports, for each outer fold, the count and percentage of test samples included in the random 80% subset used by the global RFECV fit.
- [ ] Do not call this fold-local feature selection. It is an audit of retained label exposure.
- [ ] Preserve `corrected_reference` as the fresh-model, inner-validation epoch/threshold evaluation.
- [ ] Report probability AUROC and paper-thresholded AUC separately in all modes.

## Task 5: Run Full Local Verification

- [ ] Run all DNDT/DNDF tests with warnings treated as errors:

```powershell
.\.venv-dndt-dndf\Scripts\python.exe -W error -m pytest tests/test_dndt_dndf_models.py tests/test_dndt_dndf_experiment.py tests/test_dndt_dndf_evidence.py tests/test_dndt_dndf_cli.py -q
```

- [ ] Run the legacy metric, split, uncertainty, bootstrap, and clinical-operating-point suites.
- [ ] Run a CUDA smoke for Track A and every Track B stage, including interruption and resume.
- [ ] Independently recompute saved AUROC, AUPRC, confusion metrics, ECE, Brier, and NLL from predictions.
- [ ] Request a fresh independent scientific review and fix all critical or important findings before launch.
- [ ] Commit the verified revision. A dirty tree or revision mismatch blocks scientific execution.

## Task 6: Prepare the SSH Host

- [ ] Verify the T1000 GPU, CUDA-enabled PyTorch, source input hashes, and author commit.
- [ ] Create a dedicated runtime directory outside the existing project checkout.
- [ ] Free only regenerable cache space after resolving and printing the exact path. Do not delete project data, raw audio, feature tables, or previous research results.
- [ ] Require at least 6 GiB free after cache cleanup and enable completed-unit checkpoint compaction.
- [ ] Transfer or pull the exact committed revision and record its Git SHA in the run contract.
- [ ] Run the complete warning-as-error test suite on Linux before starting a scientific unit.

## Task 7: Execute in Parallel Without Sharing State

**SSH GPU, Track B:**

1. candidates for breath, cough, speech;
2. final three-seed DNDF modality branches;
3. primary cough+speech validation-logistic fusion;
4. secondary three-modality uniform fusion;
5. the pre-specified v2 ladder covering existing, early-to-late, external cough,
   and time-stratified protocols;
6. DNDF shuffle retrain.

**Windows GPU, Track A:**

1. complete `fresh_fold_author_protocol` DNDF across ten released folds;
2. complete `corrected_reference` DNDF across ten folds;
3. generate RFECV exposure audit;
4. retain already completed released-behaviour receipts only when their code revision and hashes authenticate against the final runner, otherwise rerun them.

The two hosts must use different run-stage directories. Results are combined only by the evidence generator after artifact hashes and code revisions match.

## Task 8: Final Evidence Gate

- [ ] Require complete Track A, candidates, final, fusion,
  pre-specified-v2 ladder, and shuffle manifests.
- [ ] Generate one comparison table containing paper-reported values, released-code audit, fresh-fold author protocol, corrected reference, Track B internal branches, multimodal fusion, temporal rows, and external cough transfer.
- [ ] Label paper-thresholded AUC, probability AUROC, internal CV, temporal validation, and external transfer explicitly so unlike quantities are never presented as direct equivalents.
- [ ] Generate confidence intervals and deltas only from authenticated saved predictions.
- [ ] Write a short root-cause statement that distinguishes confirmed mechanisms from retained limitations.

## Stop Rules

- Do not continue a run after a code, configuration, feature, split, or receipt hash mismatch.
- Do not use partial manifests in papers, reports, or slides.
- Do not rerun RFECV fold-locally under the current deadline.
- Do not add HST, DNDT/DNDF architecture search beyond the frozen six trials, or another dataset.
- If time expires, retain results in this order: Track A causal comparison, Track B internal three modalities, primary fusion, early-to-late, external cough, time-stratified, secondary fusion.

## Completion Criteria

- The paper-number gap is supported by a released-code audit, a one-factor carry-over ablation, and a corrected reference evaluation.
- Track B has authenticated predictions for all required modalities, fusion rules, strict protocols, and shuffle control.
- Every reported metric is independently recomputable from saved predictions.
- All full tests pass on the exact committed revision used on both hosts.
- Remaining limitations are methodological facts, not unresolved implementation defects.
