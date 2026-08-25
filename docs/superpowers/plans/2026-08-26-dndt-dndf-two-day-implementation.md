# DNDT/DNDF Two-Day Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable DNDT/DNDF reproduction and COVID-RARS integration that produces comparable internal, multimodal, temporal, and external evidence within two days.

**Architecture:** A small PyTorch model module implements the published soft decision tree and forest. A single experiment module owns feature contracts, train-only preprocessing, training, checkpoint/resume, author-fold reproduction, and project protocols. A separate evidence module derives complete metrics and uncertainty only from saved participant-level predictions. Four numbered CLIs and one notebook call these modules without changing existing experiment code.

**Tech Stack:** Python 3.12, PyTorch 2.13 CUDA 13.0 wheels, NumPy, pandas, scikit-learn, imbalanced-learn, Optuna, joblib, pytest, Jupyter.

---

## File Map

**Create:**

- `requirements-dndt-dndf.txt` - non-PyTorch dependencies for the isolated environment.
- `configs/dndt_dndf_two_day.json` - fixed paths, published parameters, bounded search space, seeds, and protocol order.
- `src/covid_rars/dndt_dndf_models.py` - differentiable tree and forest mathematics only.
- `src/covid_rars/dndt_dndf_experiment.py` - data contracts, preprocessing, fitting, resume, Track A, Track B, tuning, and fusion.
- `src/covid_rars/dndt_dndf_evidence.py` - complete metrics, uncertainty, operating points, DCA, and evidence tables.
- `scripts/79_dndt_dndf_preflight.py` - environment, source, data, disk, GPU, and pilot checks.
- `scripts/80_run_dndt_dndf_track_a.py` - author-protocol reproduction.
- `scripts/81_run_dndt_dndf_track_b.py` - project candidates, selection, strict ladder, and fusion.
- `scripts/82_make_dndt_dndf_evidence.py` - deterministic post-processing from predictions.
- `notebooks/08_dndt_dndf_two_day_run.ipynb` - one-click queue and progress display.
- `tests/test_dndt_dndf_models.py` - model equation and gradient tests.
- `tests/test_dndt_dndf_experiment.py` - leakage, checkpoint, Track A, Track B, and fusion tests.
- `tests/test_dndt_dndf_evidence.py` - metric, uncertainty, operating-point, and schema tests.
- `tests/test_dndt_dndf_cli.py` - smoke execution and artifact tests.

**Do not modify:** existing HST files, existing model runners, existing result CSVs, or the source feature tables on `G:`.

## Task 1: Freeze Configuration and Environment Contract

**Files:**

- Create: `requirements-dndt-dndf.txt`
- Create: `configs/dndt_dndf_two_day.json`
- Create: `scripts/79_dndt_dndf_preflight.py`
- Test: `tests/test_dndt_dndf_cli.py`

- [ ] **Step 1: Write the failing configuration/preflight tests**

Add tests that import script 79 and assert:

```python
def test_preflight_rejects_cpu_only_torch_for_cuda_mode(monkeypatch, tmp_path):
    module = load_script("79_dndt_dndf_preflight.py")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        module.validate_runtime(device="cuda", run_root=tmp_path, minimum_free_gib=1.0)


def test_preflight_accepts_exact_project_feature_schema(tmp_path):
    module = load_script("79_dndt_dndf_preflight.py")
    path = write_feature_fixture(tmp_path, feature_count=800)
    audit = module.audit_project_feature_table(path)
    assert audit["feature_count"] == 800
    assert audit["required_columns_present"] is True
```

- [ ] **Step 2: Run the tests and verify the expected failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dndt_dndf_cli.py -q
```

Expected: collection fails because `scripts/79_dndt_dndf_preflight.py` does not exist.

- [ ] **Step 3: Add the isolated dependency contract**

Write:

```text
-r requirements.txt
-e .
imbalanced-learn>=0.14,<0.15
optuna>=4.5,<5
joblib>=1.5,<2
jupyterlab>=4.4,<5
ipykernel>=6.30,<7
pytest>=8.4,<9
```

PyTorch is installed separately so the CUDA wheel source is unambiguous:

```powershell
py -3.12 -m venv .venv-dndt-dndf
.\.venv-dndt-dndf\Scripts\python.exe -m pip install --upgrade pip
.\.venv-dndt-dndf\Scripts\python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv-dndt-dndf\Scripts\python.exe -m pip install -r requirements-dndt-dndf.txt
```

Create a durable author-repository checkout outside the project repository:

```powershell
New-Item -ItemType Directory -Force G:\Covid-19-BTP\external | Out-Null
git clone https://github.com/Rofiquldk1/COVID-19-Detection-from-Cough-Sound.git G:\Covid-19-BTP\external\COVID-19-Detection-from-Cough-Sound
git -C G:\Covid-19-BTP\external\COVID-19-Detection-from-Cough-Sound checkout feb0e63c790c042eaa21e9f3fc83ed64bdc8a24e
```

If that directory already exists, do not reclone it. Verify `git rev-parse HEAD` and a clean tracked worktree instead.

- [ ] **Step 4: Add the compact JSON configuration**

The configuration must contain these concrete values:

```json
{
  "author_repo": "G:/Covid-19-BTP/external/COVID-19-Detection-from-Cough-Sound",
  "author_commit": "feb0e63c790c042eaa21e9f3fc83ed64bdc8a24e",
  "project_features": "G:/Covid-19-BTP/covid_audio_btp/data/processed/features_compare_is10_top800.csv",
  "external_features": "G:/Covid-19-BTP/covid_audio_btp/data/processed/coughvid_features_compare_is10_top800.csv",
  "metadata": "G:/Covid-19-BTP/covid_audio_btp/data/processed/metadata_with_quality.csv",
  "run_root": "G:/Covid-19-BTP/dndt_dndf_runs",
  "device": "cuda",
  "published": {
    "depth": 11,
    "used_features_rate": 0.6,
    "learning_rate": 0.01,
    "batch_size": 16,
    "epochs": 14,
    "dndt_trees": 1,
    "dndf_trees": 25
  },
  "balancing": {
    "track_a": "svm_smote",
    "track_b": "smote"
  },
  "selection": {
    "primary_metric": "auroc",
    "tie_breaker": "auprc",
    "max_trials_per_modality": 6,
    "patience": 3
  },
  "seeds": {
    "candidate": [42],
    "final": [42, 314, 2026]
  },
  "modalities": ["breath", "cough", "speech"],
  "protocols": ["existing", "time_stratified", "early_to_late", "external_cough"]
}
```

- [ ] **Step 5: Implement preflight checks**

The preflight must verify the CUDA runtime, 6 GiB GPU, free disk, author commit, exact `.npy` shapes, ten fold partitions, top-800 schema, matching external feature columns, required metadata fields, and writable run root. It must output JSON and exit nonzero on failure.

- [ ] **Step 6: Run the focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dndt_dndf_cli.py -q
```

Expected: the two preflight tests pass.

- [ ] **Step 7: Commit the contract**

```powershell
git add requirements-dndt-dndf.txt configs/dndt_dndf_two_day.json scripts/79_dndt_dndf_preflight.py tests/test_dndt_dndf_cli.py
git commit -m "Add DNDT DNDF runtime preflight"
```

## Task 2: Implement and Prove the Model Mathematics

**Files:**

- Create: `src/covid_rars/dndt_dndf_models.py`
- Create: `tests/test_dndt_dndf_models.py`

- [ ] **Step 1: Write a failing depth-two tree oracle test**

The test must set feature indices, decision-layer weights, and leaf logits explicitly, then compare the module with a direct NumPy path-probability calculation:

```python
def test_depth_two_tree_matches_numpy_oracle():
    tree = NeuralDecisionTree(
        num_features=3,
        depth=2,
        used_features_rate=1.0,
        num_classes=2,
        seed=7,
        feature_indices=torch.tensor([0, 1, 2]),
    )
    with torch.no_grad():
        tree.decision.weight.copy_(torch.tensor([[9.0, 9.0, 9.0], [0.2, -0.1, 0.4], [0.3, 0.5, -0.2], [-0.4, 0.1, 0.2]]))
        tree.decision.bias.copy_(torch.tensor([9.0, 0.1, -0.2, 0.3]))
        tree.leaf_logits.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]]))
    x = torch.tensor([[0.5, -1.0, 0.25]])
    expected = numpy_tree_probability(x.numpy(), tree)
    np.testing.assert_allclose(tree(x).detach().numpy(), expected, atol=1e-6)
```

- [ ] **Step 2: Run the model test and verify it fails**

Run:

```powershell
.\.venv-dndt-dndf\Scripts\python.exe -m pytest tests/test_dndt_dndf_models.py -q
```

Expected: import failure for `covid_rars.dndt_dndf_models`.

- [ ] **Step 3: Implement `NeuralDecisionTree`**

Use these public signatures:

```python
@dataclass(frozen=True)
class ModelConfig:
    model_name: Literal["dndt", "dndf"]
    num_trees: int
    depth: int
    used_features_rate: float


class NeuralDecisionTree(nn.Module):
    def __init__(self, num_features: int, depth: int, used_features_rate: float,
                 num_classes: int = 2, seed: int = 42,
                 feature_indices: torch.Tensor | None = None) -> None: ...

    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
```

The forward pass must reproduce the author implementation, including its unused routing slot:

1. select the frozen per-tree feature subset;
2. produce `2**depth` sigmoid routing probabilities;
3. ignore routing output zero and traverse outputs `1..2**depth-1` in breadth-first order;
4. softmax `2**depth x num_classes` leaf logits;
5. return the path-weighted class probabilities.

Add a test proving that changing only routing output zero cannot change predictions. This apparently redundant parameter is retained because Track A reproduces the released Keras architecture exactly.

- [ ] **Step 4: Add a failing forest-average test**

```python
def test_forest_probability_is_arithmetic_mean_of_tree_probabilities():
    forest = NeuralDecisionForest(num_trees=2, num_features=4, depth=2,
                                  used_features_rate=0.75, seed=9)
    x = torch.randn(5, 4)
    expected = torch.stack([tree(x) for tree in forest.trees]).mean(dim=0)
    torch.testing.assert_close(forest(x), expected)
```

- [ ] **Step 5: Implement `NeuralDecisionForest` and deterministic feature subsets**

```python
class NeuralDecisionForest(nn.Module):
    def __init__(self, num_trees: int, num_features: int, depth: int,
                 used_features_rate: float, num_classes: int = 2,
                 seed: int = 42) -> None: ...

    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
```

Each tree uses `seed + tree_index` and stores selected indices as a buffer so checkpoints reproduce the same representation.

- [ ] **Step 6: Add the author batch-normalization wrapper**

Implement:

```python
class NeuralDecisionClassifier(nn.Module):
    def __init__(self, *, num_features: int, model_config: ModelConfig, seed: int) -> None: ...
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...
```

It applies `nn.BatchNorm1d(num_features)` before the tree or forest, matching `layers.BatchNormalization()` in the released code. Test that the normalization parameters are present in checkpoints and that evaluation mode uses frozen running statistics.

- [ ] **Step 7: Add finite-probability, gradient, and parameter-count tests**

Assert probabilities sum to one, gradients are finite after negative log likelihood, invalid depth/rate/tree counts raise `ValueError`, and the published DNDF configuration fits within an explicit parameter estimate before allocation.

- [ ] **Step 8: Run tests and commit**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe -m pytest tests/test_dndt_dndf_models.py -q
git add src/covid_rars/dndt_dndf_models.py tests/test_dndt_dndf_models.py
git commit -m "Implement verified neural decision trees and forests"
```

Expected: all model tests pass without warnings.

## Task 3: Add Training, Preprocessing, and Resume

**Files:**

- Create: `src/covid_rars/dndt_dndf_experiment.py`
- Create: `tests/test_dndt_dndf_experiment.py`

- [ ] **Step 1: Write failing train-only preprocessing tests**

```python
def test_preprocessor_statistics_use_training_rows_only():
    train = np.array([[0.0, np.nan], [2.0, 4.0]])
    validation = np.array([[1000.0, 1000.0]])
    fitted = fit_preprocessor(train)
    assert fitted.scaler.mean_[0] == pytest.approx(1.0)
    assert fitted.imputer.statistics_[1] == pytest.approx(4.0)
    assert transform_features(fitted, validation)[0, 0] > 100.0
```

Also assert the configured oversampler sees training rows only and falls back to class-weighted sampling when the minority count is too small for its neighbor requirement. Track A uses `SVMSMOTE` for author-code parity. Track B uses ordinary `SMOTE`, matching the established project training family at lower computational cost.

- [ ] **Step 2: Run and verify failure**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe -m pytest tests/test_dndt_dndf_experiment.py -q
```

Expected: missing experiment module.

- [ ] **Step 3: Implement compact data and configuration types**

Import `ModelConfig` from `covid_rars.dndt_dndf_models` and add these immutable experiment dataclasses:

```python
@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    seed: int
    balance_method: Literal["svm_smote", "smote", "class_weight"]


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    validation_auroc: float
    validation_auprc: float
    threshold: float
    checkpoint_path: Path
    validation_probability: np.ndarray
```

- [ ] **Step 4: Implement deterministic fitting**

`fit_model(...)` must seed Python, NumPy, and Torch; fit imputer/scaler on training only; oversample training only; minimize `torch.nn.functional.nll_loss(torch.log(probabilities.clamp_min(1e-7)), labels)`; select epochs by validation AUROC then AUPRC; and stop after configured patience. Epoch `e` uses a data-loader generator seeded with `seed + e`, making an interrupted resume reproduce the same batch order without depending on an opaque iterator state.

Add a deterministic batch-sampler test for a row count whose remainder is one. The sampler must redistribute the final rows so every training batch has at least two samples, allowing the author-compatible batch-normalization layer to stay in training mode without dropping a row.

Validation thresholds are selected with `best_threshold_by_balanced_accuracy` only after the best epoch is restored.

- [ ] **Step 5: Write a failing interruption/resume equivalence test**

```python
def test_resume_matches_uninterrupted_training(tmp_path):
    uninterrupted = run_tiny_fit(tmp_path / "full", interrupt_after_epoch=None)
    with pytest.raises(PlannedInterruption):
        run_tiny_fit(tmp_path / "resume", interrupt_after_epoch=2)
    resumed = run_tiny_fit(tmp_path / "resume", interrupt_after_epoch=None, resume=True)
    np.testing.assert_allclose(resumed.validation_probability,
                               uninterrupted.validation_probability, atol=1e-7)
```

- [ ] **Step 6: Implement atomic checkpoint/resume**

Checkpoint payloads include model, optimizer, epoch, best state, early-stop state, RNG states, model/train configuration, input-feature hash, split hash, and code revision. Write to `.<name>.tmp`, flush, then `os.replace`.

Reject resume when any fingerprint differs. Keep `latest.pt` and `best.pt`; delete neither until the task completion receipt has been written.

- [ ] **Step 7: Add participant aggregation and prediction schema tests**

Require these columns in every saved prediction row:

```text
run_id, track, protocol, fold, seed, dataset, split, model_name,
modality, participant_id, recording_id, label_binary, probability,
threshold, threshold_source, configuration_sha256, split_sha256,
feature_sha256, checkpoint_sha256
```

Track A uses `analysis_id` and `analysis_unit=author_sample` when participant IDs are unavailable.

- [ ] **Step 8: Run focused tests and commit**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe -m pytest tests/test_dndt_dndf_experiment.py -q
git add src/covid_rars/dndt_dndf_experiment.py tests/test_dndt_dndf_experiment.py
git commit -m "Add resumable DNDT DNDF training core"
```

## Task 4: Implement Track A Reproduction

**Files:**

- Modify: `src/covid_rars/dndt_dndf_experiment.py`
- Create: `scripts/80_run_dndt_dndf_track_a.py`
- Modify: `tests/test_dndt_dndf_experiment.py`
- Modify: `tests/test_dndt_dndf_cli.py`

- [ ] **Step 1: Write failing author-artifact contract tests**

Test the expected Coswara array shape `(1319, 193)`, 1,319 labels, 185 `C`, 1,134 `N`, ten disjoint outer test folds whose union contains every sample once, and explicit `C -> 1`, `N -> 0` conversion. Reproduce the author's single global feature-selection pass using `train_test_split(test_size=0.20, random_state=42)`, `ExtraTreesClassifier(n_estimators=50, random_state=0)`, and `RFECV(step=1, cv=StratifiedKFold(10), scoring="roc_auc", min_features_to_select=1)`. Assert that it selects 33 variables.

- [ ] **Step 2: Implement author artifact loading and hashing**

Use only:

```text
Extracted Features/Coswara/cough_X_features_np.npy
Extracted Features/Coswara/cough_y_features_np.npy
Train-Test Split/coswaradataset/train/{0..9}.csv
Train-Test Split/coswaradataset/test/{0..9}.csv
```

Fail on count, class, overlap, partition, commit, or hash mismatch.

Cache the 33 selected indices and transformed array with their hashes. Both author-behaviour and corrected-reference modes use this same cached author selection. Do not rerun RFECV inside every fold. The Track A report must mark global feature selection and the preprocessed released array as retained limitations.

- [ ] **Step 3: Write failing separation test for the two modes**

Assert corrected mode instantiates a fresh model for every fold and chooses threshold from an inner stratified validation subset. It trains on the inner-training subset, restores the validation-selected epoch, and touches the outer test fold once without a second refit. Assert author-behaviour mode records `threshold_source=test_balanced_accuracy_author_audit` and `model_reinitialized_per_fold=False`.

- [ ] **Step 4: Implement `run_track_a(...)`**

Run DNDT and DNDF for all ten folds in:

- `author_behaviour_audit`
- `corrected_reference`

Save each fold immediately. Compute probability AUROC in `auroc` and the thresholded-label quantity only in `paper_thresholded_auc`.

Author-behaviour thresholding evaluates `np.arange(0.0, 1.0, 0.001)` on the outer test fold, matching the released notebook. Convert its class-one probability to COVID-positive orientation before normal metrics. Do not reproduce the notebook's incorrect variable names for confusion-matrix cells; TP, FP, TN, and FN always follow sklearn's `labels=[0, 1]` layout in COVID-positive orientation.

- [ ] **Step 5: Add CLI smoke test and implementation**

The CLI contract is:

```powershell
python scripts/80_run_dndt_dndf_track_a.py --config configs/dndt_dndf_two_day.json --run-id <id> --resume
```

Test mode adds `--smoke`, limiting execution to one fold, depth two, two trees, and two epochs without changing full-run configuration files.

- [ ] **Step 6: Run tests and commit**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe -m pytest tests/test_dndt_dndf_experiment.py tests/test_dndt_dndf_cli.py -q
git add src/covid_rars/dndt_dndf_experiment.py scripts/80_run_dndt_dndf_track_a.py tests/test_dndt_dndf_experiment.py tests/test_dndt_dndf_cli.py
git commit -m "Add auditable DNDT DNDF paper reproduction"
```

## Task 5: Implement Track B Modalities and Frozen Validation Ladder

**Files:**

- Modify: `src/covid_rars/dndt_dndf_experiment.py`
- Create: `scripts/81_run_dndt_dndf_track_b.py`
- Modify: `tests/test_dndt_dndf_experiment.py`
- Modify: `tests/test_dndt_dndf_cli.py`

- [ ] **Step 1: Write failing split-isolation tests**

Build a synthetic three-modality feature table and assert zero participant overlap among train, validation, and test for existing, time-stratified, and early-to-late protocols. Assert COUGHVID rows never enter fit, tuning, early stopping, scaling, or threshold selection.

- [ ] **Step 2: Implement feature loading and protocol construction**

Use `covid_rars.features.feature_columns`, `build_time_stratified_split_assignments`, `build_temporal_split_assignments`, and `_apply_split_to_features`. Load the 807-column top-800 table once, retain labelled rows, and validate that the external table has the same ordered 800 feature columns.

- [ ] **Step 3: Write a failing bounded-selection test**

```python
def test_tuning_never_exceeds_six_trials_and_never_receives_test_rows(tmp_path):
    result = select_modality_configuration(fixture, modality="cough", max_trials=6,
                                           storage=tmp_path / "study.sqlite3")
    assert result.completed_trials <= 6
    assert set(result.observed_splits) <= {"train", "validation"}
```

- [ ] **Step 4: Implement candidate selection**

For each modality:

1. run published DNDT once;
2. run published DNDF once;
3. rank validation AUROC then AUPRC;
4. if DNDF does not win, run at most six persistent Optuna trials;
5. freeze the best DNDF configuration in `selected_configurations.json` with validation metrics and hashes;
6. record whether DNDT or DNDF is the overall validation winner without replacing the required DNDF lineage.

The allowed search is:

```python
depth = trial.suggest_int("depth", 5, 11)
num_trees = trial.suggest_categorical("num_trees", [5, 10, 15, 25])
used_features_rate = trial.suggest_categorical("used_features_rate", [0.4, 0.6, 0.8])
learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
```

- [ ] **Step 5: Write a failing frozen-lineage test**

Assert that strict protocol jobs receive exactly the selected DNDF modality configuration and feature hash from the existing-split selection receipt. A protocol-specific configuration override must raise `ValueError`.

- [ ] **Step 6: Implement final modality runs**

Run three final seeds for each selected DNDF modality configuration on the existing split. If DNDT is the overall validation winner for a modality, run its published configuration for the same three seeds on the existing split only. Retrain the frozen DNDF configuration on time-stratified and early-to-late protocol training rows. For external cough, train and select only on Coswara train/validation and evaluate the frozen DNDF cough branch once on COUGHVID.

- [ ] **Step 7: Implement CLI stages and immediate receipts**

The CLI supports:

```powershell
python scripts/81_run_dndt_dndf_track_b.py --config configs/dndt_dndf_two_day.json --run-id <id> --stage candidates --resume
python scripts/81_run_dndt_dndf_track_b.py --config configs/dndt_dndf_two_day.json --run-id <id> --stage final --resume
python scripts/81_run_dndt_dndf_track_b.py --config configs/dndt_dndf_two_day.json --run-id <id> --stage ladder --resume
```

Each completed modality/protocol/seed writes predictions and a completion JSON before the next unit starts.

- [ ] **Step 8: Run tests and commit**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe -m pytest tests/test_dndt_dndf_experiment.py tests/test_dndt_dndf_cli.py -q
git add src/covid_rars/dndt_dndf_experiment.py scripts/81_run_dndt_dndf_track_b.py tests/test_dndt_dndf_experiment.py tests/test_dndt_dndf_cli.py
git commit -m "Add bounded DNDT DNDF project validation ladder"
```

## Task 6: Implement Primary and Secondary Fusion

**Files:**

- Modify: `src/covid_rars/dndt_dndf_experiment.py`
- Modify: `tests/test_dndt_dndf_experiment.py`

- [ ] **Step 1: Write failing validation-stack tests**

Assert the cough+speech logistic stack fits only validation participant predictions, freezes coefficients/intercept, and applies them unchanged to test rows. Changing test labels or probabilities must not change learned weights.

- [ ] **Step 2: Implement participant alignment and primary stack**

Aggregate recording probabilities to participant-modality means. Inner-join cough and speech for the primary stack, require identical labels, fit `LogisticRegression(random_state=seed, max_iter=2000)`, record coefficients/intercept, and apply to test.

- [ ] **Step 3: Write and implement the secondary uniform-mean test**

Use `covid_rars.fusion.uniform_fusion` on breath/cough/speech participant predictions. Assert a row with two available modalities averages exactly those two and records `available_modalities`.

- [ ] **Step 4: Add fusion stages to script 81**

```powershell
python scripts/81_run_dndt_dndf_track_b.py --config configs/dndt_dndf_two_day.json --run-id <id> --stage fusion --resume
```

- [ ] **Step 5: Run tests and commit**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe -m pytest tests/test_dndt_dndf_experiment.py -q
git add src/covid_rars/dndt_dndf_experiment.py tests/test_dndt_dndf_experiment.py
git commit -m "Add validation-trained DNDF multimodal fusion"
```

## Task 7: Implement Complete Metrics and Uncertainty

**Files:**

- Create: `src/covid_rars/dndt_dndf_evidence.py`
- Create: `scripts/82_make_dndt_dndf_evidence.py`
- Create: `tests/test_dndt_dndf_evidence.py`
- Modify: `tests/test_dndt_dndf_cli.py`

- [ ] **Step 1: Write failing exact-metric tests**

For a four-row fixture with known confusion matrix, assert exact values for accuracy, balanced accuracy, F1, sensitivity, specificity, PPV, NPV, TP, FP, TN, FN, prevalence, Brier, ECE, MCE, NLL, AUROC, average precision, and AUPRC lift.

Use left-closed/right-open calibration bins with the last bin closed.

- [ ] **Step 2: Implement `complete_metric_bundle(...)`**

```python
def complete_metric_bundle(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    n_bins: int = 10,
) -> dict[str, float | int]: ...
```

Reject nonfinite or out-of-range probabilities instead of silently clipping except for the explicit NLL calculation.

- [ ] **Step 3: Write failing participant-bootstrap tests**

Assert deterministic output for a fixed seed, cluster resampling by participant, and two-sample independent resampling for Coswara-to-COUGHVID deltas. Paired comparison must reject nonidentical participant sets.

- [ ] **Step 4: Implement final-only uncertainty**

Generate 2,000 participant bootstrap replicates for AUROC, AUPRC, Brier, and ECE on final selected rows. Record point estimate, percentile 95% interval, valid replicate count, seed, and paired/unpaired design.

- [ ] **Step 5: Implement operating point and DCA tests**

Select the sensitivity-at-least-0.90 threshold on validation only, then evaluate test/external. DCA must report model, treat-all, and treat-none net benefit over threshold probabilities 0.01 through 0.50 in increments of 0.01.

- [ ] **Step 6: Implement the evidence CLI**

```powershell
python scripts/82_make_dndt_dndf_evidence.py --config configs/dndt_dndf_two_day.json --run-id <id>
```

It reads saved predictions only and writes:

```text
metrics.csv
final_summary.csv
bootstrap_ci.csv
paired_comparisons.csv
external_deltas.csv
fixed_sensitivity_operating_points.csv
decision_curve.csv
model_selection.csv
fusion_weights.csv
run_manifest.json
```

- [ ] **Step 7: Independently recompute final CSV metrics in the CLI test**

Read the emitted predictions and compare AUROC/AP/confusion metrics directly with sklearn to `1e-12` tolerance.

- [ ] **Step 8: Run tests and commit**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe -m pytest tests/test_dndt_dndf_evidence.py tests/test_dndt_dndf_cli.py -q
git add src/covid_rars/dndt_dndf_evidence.py scripts/82_make_dndt_dndf_evidence.py tests/test_dndt_dndf_evidence.py tests/test_dndt_dndf_cli.py
git commit -m "Add complete DNDT DNDF evidence metrics"
```

## Task 8: Build the One-Click Notebook and Progress View

**Files:**

- Create: `notebooks/08_dndt_dndf_two_day_run.ipynb`
- Modify: `tests/test_dndt_dndf_cli.py`

- [ ] **Step 1: Write a failing notebook contract test**

Load notebook JSON and assert it contains no embedded dataset, no hard-coded run ID, calls scripts 79 through 82, always passes `--resume`, stops on nonzero return code, and prints completed/pending task counts from receipts.

- [ ] **Step 2: Build six short notebook cells**

1. resolve project root, config, and a date-based run ID;
2. run preflight;
3. run Track A;
4. run Track B stages in order;
5. generate evidence;
6. display progress and final summary.

Use `subprocess.run(command, cwd=PROJECT_ROOT, check=True)` rather than notebook shell magic so errors cannot be ignored.

- [ ] **Step 3: Run notebook contract and full focused suite**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe -m pytest tests/test_dndt_dndf_models.py tests/test_dndt_dndf_experiment.py tests/test_dndt_dndf_evidence.py tests/test_dndt_dndf_cli.py -q
```

Expected: all focused tests pass.

- [ ] **Step 4: Commit the notebook**

```powershell
git add notebooks/08_dndt_dndf_two_day_run.ipynb tests/test_dndt_dndf_cli.py
git commit -m "Add one-click DNDT DNDF experiment notebook"
```

## Task 9: Run Scientific Acceptance Gates

**Files:**

- Modify only if a failing test reveals a defect in the new DNDT/DNDF files.

- [ ] **Step 1: Run the existing regression suite most relevant to contracts**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe -m pytest tests/test_metrics.py tests/test_split.py tests/test_compare_is10_final_validation.py tests/test_final_uncertainty.py tests/test_delta_bootstrap.py tests/test_clinical_operating_points.py -q
```

Expected: all pass.

- [ ] **Step 2: Run the full new suite with warnings treated as errors**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe -W error -m pytest tests/test_dndt_dndf_models.py tests/test_dndt_dndf_experiment.py tests/test_dndt_dndf_evidence.py tests/test_dndt_dndf_cli.py -q
```

Expected: all pass with no warnings.

- [ ] **Step 3: Run the actual preflight**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe scripts/79_dndt_dndf_preflight.py --config configs/dndt_dndf_two_day.json --device cuda
```

Expected JSON: `status=ready`, CUDA true, 800 project features, matching external features, verified author commit, and ten valid folds.

- [ ] **Step 4: Run the interruption/resume GPU smoke**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe scripts/80_run_dndt_dndf_track_a.py --config configs/dndt_dndf_two_day.json --run-id acceptance-smoke --smoke --planned-interrupt-after-epoch 1
.\.venv-dndt-dndf\Scripts\python.exe scripts/80_run_dndt_dndf_track_a.py --config configs/dndt_dndf_two_day.json --run-id acceptance-smoke --smoke --resume
```

Expected: first command exits with the documented planned-interruption code; second resumes at epoch two and completes.

- [ ] **Step 5: Inspect GPU memory and runtime receipt**

The pilot passes only when peak allocated CUDA memory remains below 5.2 GiB, all probabilities are finite, and the completion receipt contains hashes for configuration, source, features, split, checkpoint, and predictions.

- [ ] **Step 6: Commit any acceptance-only fixes and tag the runnable revision**

```powershell
git status --short
git tag dndt-dndf-run-ready-2026-08-26
```

Do not start a full scientific run from an uncommitted tree.

## Task 10: Execute the Two-Day Queue

**Files:**

- Generated artifacts only under `G:/Covid-19-BTP/dndt_dndf_runs/<run-id>`.

- [ ] **Step 1: Start Track A and record elapsed time**

```powershell
$RUN_ID="dndt-dndf-20260826"
.\.venv-dndt-dndf\Scripts\python.exe scripts/80_run_dndt_dndf_track_a.py --config configs/dndt_dndf_two_day.json --run-id $RUN_ID --resume
```

- [ ] **Step 2: Run candidates and bounded selection**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe scripts/81_run_dndt_dndf_track_b.py --config configs/dndt_dndf_two_day.json --run-id $RUN_ID --stage candidates --resume
```

- [ ] **Step 3: Run final seeds, fusion, and ladder**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe scripts/81_run_dndt_dndf_track_b.py --config configs/dndt_dndf_two_day.json --run-id $RUN_ID --stage final --resume
.\.venv-dndt-dndf\Scripts\python.exe scripts/81_run_dndt_dndf_track_b.py --config configs/dndt_dndf_two_day.json --run-id $RUN_ID --stage fusion --resume
.\.venv-dndt-dndf\Scripts\python.exe scripts/81_run_dndt_dndf_track_b.py --config configs/dndt_dndf_two_day.json --run-id $RUN_ID --stage ladder --resume
```

- [ ] **Step 4: Generate evidence and independently verify it**

```powershell
.\.venv-dndt-dndf\Scripts\python.exe scripts/82_make_dndt_dndf_evidence.py --config configs/dndt_dndf_two_day.json --run-id $RUN_ID
.\.venv-dndt-dndf\Scripts\python.exe -m pytest tests/test_dndt_dndf_evidence.py -q
```

- [ ] **Step 5: Apply the runtime fallback order if the deadline is reached**

Never kill an active task mid-write. Stop before the next queued unit in this order: secondary uniform fusion, time-stratified ladder, external cough, early-to-late ladder. Track A, three project modalities, and primary cough+speech fusion remain the minimum complete package.

## Final Self-Review Checklist

- [ ] Track A and Track B never share samples or preprocessing objects.
- [ ] Positive class is COVID positive in all output files.
- [ ] `auroc` always uses probabilities.
- [ ] `paper_thresholded_auc` is visibly separated.
- [ ] Test and external labels never influence fitting, tuning, early stopping, thresholding, or fusion weights.
- [ ] The same frozen selected configuration is used throughout each Track B validation lineage.
- [ ] Every long unit resumes from an atomic checkpoint.
- [ ] All metrics can be recomputed from saved predictions.
- [ ] Existing project files and `G:` source tables remain unchanged.
- [ ] A full run begins only from the tagged, tested revision.
