# DNDT / DNDF — Immutable Source and Initialization

*Companion section to the HST freeze doc. Structure mirrors HST's "Immutable
source and initialization" section, adapted for the fact that this component
is not a pinned external submodule.*

## 1. Provenance

**Status: in-repo original implementation. Not vendored, not a git submodule.**

Confirmed by inspection of `.gitmodules` at repo root — the only registered
submodule is `HST` (`https://github.com/icon-lab/HST.git`). No entry exists
for DNDT/DNDF. The implementation lives directly in this repository as
first-party code.

| File | Role |
|---|---|
| `app/models/deep_trees.py` | Model definitions: `DeepNeuralDecisionTree`, `DecisionTree`, `DeepNeuralDecisionForest` |
| `scripts/run_dndt_dndf.py` | Training/eval driver (k-fold CV, bootstrap CI, calibration) |
| `configs/dndt_dndf_reliability.json` | Hyperparameter config for the benchmark run |

All three files carry a filesystem timestamp of 2026-08-20, one week after
the rest of the repository (2026-08-13), indicating this component was added
in a separate, later batch.

**Note on git history:** the working copy this was verified against is a
GitHub ZIP export (`Covid-RARS-main.zip`), not a git clone — there is no
`.git` directory, so no commit-level history is available for this repo,
including for the HST submodule pin itself. Integrity below is therefore
anchored to file-content hashes rather than commit SHAs. If commit-level
provenance is later needed, re-clone with `git clone --recurse-submodules`
and regenerate this section from `git log` / `git rev-parse`.

## 2. Design lineage

Per the module's own docstrings (not inferred, not external claims):

- `DeepNeuralDecisionTree` — **Yang, Morillo & Hospedales, 2018**, "Deep
  Neural Decision Trees" (arXiv:1806.06988). Soft per-feature binning via
  cutpoints, joint routing via tensor outer product, differentiable leaf
  layer.
- `DecisionTree` / `DeepNeuralDecisionForest` — **Kontschieder et al., 2015**,
  "Deep Neural Decision Forests" (ICCV 2015). Oblique soft split via a linear
  decision layer, ensemble of trees averaged at output.

**Correction to earlier working assumption:** an initial hypothesis in this
review linked the DNDT/DNDF setup to Islam, Chowdhury & Kabir (2025),
arXiv:2501.01117, based on surface similarity (DNDT/DNDF + COVID cough
audio). That paper is **not cited anywhere in the code** and the
`k_dndt: 8` config value that seemed to match its reported leaf cap is
actually a `sklearn.feature_selection.SelectKBest` parameter — a
feature-selection width, not a decision-tree leaf count. This attribution is
**not included** in this freeze doc. The two design references above are the
only ones the code itself supports.

## 3. Architecture constraints (as implemented)

- `DeepNeuralDecisionTree` asserts `in_features <= 12` (guards against the
  `(cutpoints+1)^D` leaf-count explosion).
- `DecisionTree` (DNDF base learner) leaf count is `2^depth`, internal node
  count is `2^depth - 1`.
- `DeepNeuralDecisionForest` averages outputs across `num_trees` independent
  `DecisionTree` instances — no boosting, no weighting.

## 4. Hyperparameters (`configs/dndt_dndf_reliability.json`)

| Component | Param | Value |
|---|---|---|
| Feature reduction | method | `select_k_best` (ANOVA F-test) |
| Feature reduction | k (DNDT input) | 8 |
| Feature reduction | k (DNDF input) | 32 |
| DNDT | num_cutpoints | 1 |
| DNDT | temperature | 1.0 |
| DNDT | lr | 0.01 |
| DNDT | epochs | 60 |
| DNDF | num_trees | 12 |
| DNDF | depth | 4 |
| DNDF | lr | 0.003 |
| DNDF | epochs | 80 |
| Validation | n_splits | 5 |
| Validation | bootstrap_rounds | 1000 |
| Validation | leakage_safe | true |

Modalities: cough, speech, breath. Feature sets: `opensmile_compare_is10`,
`beats`, `panns`.

## 5. Integrity — file-content SHA-256

Computed locally via `Get-FileHash -Algorithm SHA256` against the working
copy, verified as 64-character hex digests before inclusion here.

| File | SHA-256 |
|---|---|
| `app/models/deep_trees.py` | `3488EE60178F844C6993FBA601EAE21F822781774F402E1C2865B4704227F462`[^len] |
| `scripts/run_dndt_dndf.py` | `0461E2123B4E311DFE5448A774807742F465AA23C0F9E5ADF50A3A7260CDE4E7`[^len] |
| `configs/dndt_dndf_reliability.json` | `3006885DB096F6A6C3B285F95B7729918970266FE2990DEEC5DE300167F927CA`[^len] |

[^len]: Each verified as exactly 64 hex characters (standard SHA-256 length)
before being recorded here.

**Anchor date:** 2026-08-20.

**Reproducibility check:** anyone verifying this freeze should re-run
`Get-FileHash -Algorithm SHA256` (Windows) or `sha256sum` (Linux/Mac)
against the same three files and confirm an exact match against the table
above. Any mismatch means the files have changed since this freeze and the
doc must be re-anchored, not silently trusted.

## 6. Pretrained checkpoints

**Not applicable.** Both `DeepNeuralDecisionTree` and
`DeepNeuralDecisionForest` initialize all parameters from scratch
(`nn.Parameter(torch.randn(...) * 0.01)` for leaf weights, `linspace`-based
cutpoint init, standard `nn.Linear` init for the DNDF decision layer). No
checkpoint loading path exists in `run_dndt_dndf.py`'s imports or model
constructors. If this changes (e.g. a warm-start checkpoint is introduced),
this section must be updated with the checkpoint's own SHA-256 before the
freeze doc can be considered current.

## 7. Open items before this section can be considered fully locked

- [ ] Confirm no `.git` re-clone is planned that would supersede file-hash
      anchoring with commit-hash anchoring (optional, not blocking).
- [ ] Confirm training script (`run_dndt_dndf.py`) is itself unchanged since
      the hash above was taken — re-run hash check if any edits occur before
      the pilot/acceptance run.
- [ ] Author/owner sign-off on the corrected design-lineage citations (§2)
      replacing the earlier Islam/Chowdhury/Kabir reference.
