# DNDT & DNDF Ubuntu / Linux Full Execution Runbook

**Target Architecture:** Deep Neural Decision Trees (DNDT) & Deep Neural Decision Forests (DNDF)
**Environment:** Ubuntu 20.04 / 22.04 LTS (x86_64, CUDA >= 11.8)
**Repository Working Directory:** `Covid-RARS-main`

---

## 1. System Requirements & Inferred Dependencies

> **Note on Dependencies:** The dependency set below is inferred from the imports in `app/models/deep_trees.py` and `scripts/run_dndt_dndf.py`. If you have specific pinned packages in your root `pyproject.toml` or `requirements.txt`, align the versions accordingly.

### Minimal Dependency Stack
* **Python:** `>= 3.9` (Recommended: `3.10` or `3.11`)
* **PyTorch:** `torch >= 2.0.0` (with CUDA support)
* **Core ML & Data:** `scikit-learn >= 1.2.0`, `pandas >= 1.5.0`, `numpy >= 1.23.0`
* **Optional Pipeline Acceleration:** `scipy >= 1.10.0`, `tqdm >= 4.65.0`

---

## 2. Environment Provisioning

### Step 2.1: Conda / Mamba Setup (Recommended)

```bash
# Create dedicated environment
conda create -n covid-audio python=3.10 -y
conda activate covid-audio

# Install PyTorch with CUDA support (adjust CUDA version to match your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install tabular & evaluation dependencies
pip install scikit-learn pandas numpy scipy
```

### Step 2.2: Standard Virtualenv Alternative

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn pandas numpy scipy
```

### Step 2.3: Environment Health Check

```bash
# Verify torch and CUDA visibility
python3 -c "import torch; print(f'PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 3. Pre-Flight Verification & Smoke Testing

Before initiating multi-fold cross-validation or loading large feature matrices, execute the pre-flight checks:

### Step 3.1: Python Path & Module Syntax Check

```bash
# Ensure repository root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Compile check to ensure no syntax/import errors
python3 -m py_compile app/models/deep_trees.py scripts/run_dndt_dndf.py tests/test_deep_trees.py
```

### Step 3.2: Unit Smoke Test (Shapes & Backprop Gradients)

```bash
python3 tests/test_deep_trees.py
```

Expected terminal output:
```
[SUCCESS] All shape, forward, and backward gradient assertions passed!
```

### Step 3.3: Pipeline Dry-Run (Synthetic End-to-End)

```bash
python3 scripts/run_dndt_dndf.py --dry-run --output_dir data/outputs/metrics
```

Expected terminal output:
```
[DRY-RUN] Executing synthetic pipeline check...
[DNDT] AUROC: ... [..., ...] | AUPRC: ... [..., ...]
[DNDF] AUROC: ... [..., ...] | AUPRC: ... [..., ...]
[DRY-RUN COMPLETE] Saved sample metrics to data/outputs/metrics/dndt_dndf_dryrun_metrics.csv
```

---

## 4. Execution Protocol: Full Acoustic Feature Benchmark

### Step 4.1: Standard Stratified 5-Fold Evaluation

Execute the benchmark over pre-extracted feature sets (openSMILE ComParE / eGeMAPS / SSL embeddings):

```bash
# Headless run with stdout logging
python3 scripts/run_dndt_dndf.py \
    --config configs/dndt_dndf_reliability.json \
    --output_dir data/outputs/metrics \
    2>&1 | tee data/outputs/metrics/dndt_dndf_benchmark.log
```

### Step 4.2: Background / Detached Execution (Long-Running Tasks)

If training alongside full SSL representation extraction, use `nohup` or `tmux`:

```bash
nohup python3 scripts/run_dndt_dndf.py \
    --config configs/dndt_dndf_reliability.json \
    --output_dir data/outputs/metrics \
    > data/outputs/metrics/dndt_dndf_run.log 2>&1 &

# Monitor real-time progress
tail -f data/outputs/metrics/dndt_dndf_run.log
```

---

## 5. Post-Execution Artifact Validation

Once training completes, verify the output artifacts in `data/outputs/metrics/`:

```bash
# Check generated metric files
ls -lh data/outputs/metrics/dndt_dndf*
```

Expected generated records:
* `dndt_dndf_paper_comparable_cv_metrics.csv` — Fold-level and aggregate AUROC / AUPRC.
* `dndt_dndf_bootstrap_ci.csv` — 1,000-round non-parametric bootstrap 95% confidence intervals.
* `dndt_dndf_feature_selection_record.csv` — Retained top-ranked acoustic feature indices per fold.

---

## 6. Downstream Asset & Manuscript Rebuild

Sync the new tree architecture metrics into the manuscript assets and verification suites:

```bash
# Rebuild SVG figures and CSV summary tables
python3 manuscripts/iatmsi_2027/submission_final/scripts/build_assets.py

# Audit all generated tables against submission evidence gates
python3 manuscripts/iatmsi_2027/submission_final/scripts/audit_submission.py
```
