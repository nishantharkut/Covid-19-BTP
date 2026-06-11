# BTP A-Grade And Publication Target

Last updated: 2026-06-10

## Priority Order

Primary goal:

```text
Get an A-grade BTP with a working, reproducible, well-defended respiratory-audio research system.
```

Secondary goal:

```text
Build enough rigor that the work can become a tier-2 conference submission or a serious journal manuscript after real results and ablations exist.
```

Do not reverse these priorities. A complex model that fails to run locally is worse than a simpler pipeline with clean data handling, strong evaluation, and honest limitations.

## Non-Negotiable Position

This project must not be presented as a clinical COVID diagnostic tool.

The defensible framing is:

```text
A reliability-aware, calibration-aware, confounding-controlled evaluation framework for crowdsourced respiratory-audio COVID screening under quality variation and cross-dataset cough-domain shift.
```

The unsafe framing is:

```text
A state-of-the-art COVID detector.
```

## What Counts As A-Grade BTP Completion

The BTP is A-grade ready only when these are true:

1. The full Coswara pipeline runs from raw cloned dataset to final tables.
2. The notebook and scripts are reproducible from a clean local environment.
3. Participant-level leakage is explicitly checked and absent.
4. Audio quality is audited before model conclusions are made.
5. Positive and negative labels exist in train, validation, and test splits.
6. Classical audio baselines beat dummy baselines on valid metrics.
7. Calibration metrics are reported, not only accuracy/F1.
8. Fusion is compared against individual modalities and naive fusion.
9. Metadata-only baseline is included to expose confounding risk.
10. The report explains failures and limitations honestly.
11. A small demo or prediction interface can load artifacts after local training.
12. The final BTP presentation has architecture, workflow, metrics, and error-analysis figures.

Minimum artifact set:

```text
reports/tables/coswara_layout_audit.csv
reports/tables/validation_issues.csv
data/interim/coswara_index.csv
data/interim/split_manifest.csv
data/processed/metadata_clean.csv
data/processed/audio_quality.csv
data/processed/features_mfcc.csv
data/outputs/metrics/ml_baseline_metrics.csv
data/outputs/metrics/calibration_metrics.csv
data/outputs/metrics/fusion_metrics.csv
reports/tables/paper_metric_table.csv
```

If these exist and make sense, the BTP is defensible even before advanced SSL/adversarial work.

## What Counts As Publication-Track Evidence

The project becomes publication-track only after the A-grade baseline is stable.

Minimum publication-track evidence:

1. Full Coswara baseline results with confidence intervals.
2. Calibration and abstention analysis.
3. Metadata-only and matched-cohort confounding analysis.
4. Cough-only external validation on COUGHVID.
5. Feature-shift/domain-shift report between Coswara cough and COUGHVID cough.
6. Paired model comparisons showing whether calibration/fusion/quality weighting help.
7. Reproducibility manifest with code version, package versions, artifact hashes, and run settings.

Stronger publication evidence:

1. Compact CNN spectrogram baseline, if the GPU run is stable.
2. Frozen SSL embedding baseline on cough-only audio.
3. Acoustic-domain proxy labels and proxy leakage analysis.
4. SSL plus adversarial/domain-proxy ablation only if compute and baseline evidence justify it.
5. Ablation table comparing classical, CNN, SSL, and adversarial variants under the same split and metric rules.

## Tier-1/Tier-2 Reality Check

No codebase can guarantee tier-1 or tier-2 acceptance before real metrics exist.

For a serious submission, the paper must show at least one of these:

1. A clear methodological contribution that changes the reliability story.
2. A rigorous negative result showing why apparent COVID audio performance collapses under confounding or domain shift.
3. A strong ablation proving that calibration, quality filtering, abstention, or domain-aware modeling materially improves reliability.
4. External validation that is honestly scoped to cough-only COUGHVID.

If the raw accuracy is not high, the project can still be publishable as a reliability audit. If the evaluation is weak or leaky, even high accuracy is not publishable.

## Gemini/PDF Audit Integration

The Gemini chats and PDF critiques are treated as an audit layer, not as direct implementation instructions.

Accepted into the current baseline:

- quality filtering before claims;
- participant-level split;
- no train/test leakage;
- calibration before fusion;
- non-uniform fusion alternatives;
- metadata-only baseline for confounding;
- confounding/matching diagnostics;
- COUGHVID used as cough-only external validation;
- abstention and reliability analysis.

Delayed to advanced ablation:

- Wav2Vec/HuBERT/SSL embeddings;
- acoustic-domain proxy labels;
- Gradient Reversal Layer adversarial training;
- SSL versus MFCC versus CNN ablation matrix.

Rejected as unsafe:

- replacing the baseline with a pure adversarial SSL model before first results;
- claiming COUGHVID supports full cough-breath-speech external validation;
- claiming GRL automatically removes bias;
- making saliency/XAI the core novelty before confounding is controlled.

## Lab Hardware Decision

Known lab machine:

```text
CPU: Intel i7-14700, 24 exposed cores
RAM: 19 GB
GPU: NVIDIA T1000 8 GB
Disk: about 430 GB
```

Use this machine for:

- Coswara baseline;
- quality audit and feature extraction;
- classical ML;
- calibration/fusion/confounding analysis;
- compact CNN with small batch size;
- frozen SSL embedding extraction only if done carefully.

Avoid on this machine unless necessary:

- full SSL fine-tuning;
- large Wav2Vec/HuBERT end-to-end training;
- heavy GRL training with large encoders;
- large multi-dataset transformer training.

For T1000 8 GB, the safe advanced option is:

```text
Precompute frozen cough-only SSL embeddings with batch size 1-4, then train lightweight classifiers on the embeddings.
```

The unsafe advanced option is:

```text
Fine-tune a large SSL model end-to-end with adversarial heads.
```

## Execution Ladder

### Phase 1: A-Grade Survival Baseline

Run only:

```text
Coswara -> index -> metadata -> split -> quality -> MFCC features -> ML baselines -> calibration -> fusion -> paper tables
```

Do not enable:

```text
CNN
COUGHVID feature extraction
SSL
GRL
```

### Phase 2: A-Grade Strong Additions

Enable:

```text
metadata-only baseline
quality-weighted fusion
abstention analysis
bootstrap confidence intervals
paired comparisons
confounding matching
experiment manifest
```

These are more important for marks than a risky large model because they show engineering maturity and research honesty.

### Phase 3: External Validity

Enable COUGHVID only after Coswara succeeds.

Allowed:

```text
Coswara cough-only -> COUGHVID cough-only external validation
feature-shift report
calibration decay report
```

Not allowed:

```text
multimodal external validation on COUGHVID
```

### Phase 4: Publication Upgrade

Only after phases 1-3 produce usable artifacts:

```text
25_acoustic_domain_proxy.py
26_extract_ssl_embeddings.py
27_train_adversarial_ssl.py
28_compare_adversarial_vs_baselines.py
```

On the T1000 GPU, start with frozen embeddings and lightweight classifiers. Add GRL only as a controlled ablation if the embedding baseline runs reliably.

## Presentation Strategy For BTP

Lead with:

1. Problem and dataset difficulty.
2. Why naive COVID audio claims are risky.
3. Leakage-safe pipeline.
4. Quality audit.
5. Baselines and calibration.
6. Fusion and uncertainty.
7. Confounding analysis.
8. External cough-only validation plan or results.
9. Demo/output artifacts.
10. Limitations and future publication extension.

Do not lead with:

```text
We beat SOTA.
```

Lead with:

```text
We built a reproducible research framework that tests whether respiratory-audio COVID screening remains reliable under real-world quality, confounding, and domain-shift stress.
```

## What The User Does

The user runs local compute:

1. Clone project locally.
2. Clone Coswara under `data/raw/coswara`.
3. Run `python extract_data.py` inside the Coswara folder.
4. Run preflight.
5. Run the notebook section by section.
6. Send failed output or generated artifact tables.

## What Codex Does

Codex maintains the code/research system:

1. Audits code and docs.
2. Fixes failures from local logs.
3. Interprets metrics and detects invalid claims.
4. Decides when to enable CNN/COUGHVID/advanced ablations.
5. Produces report-ready tables, figure plans, and manuscript framing.
6. Keeps the project from drifting into risky, un-runnable experiments before the BTP baseline is safe.

## Stop Conditions

Stop and debug before claiming anything if:

- labels are mostly unknown;
- participant leakage appears;
- train/validation/test has missing classes;
- most audio fails quality checks;
- dummy baselines beat real models;
- metadata-only model strongly beats audio;
- calibration worsens severely;
- COUGHVID external performance collapses without explanation.

These are not project failures. They are research findings if handled honestly.

## Final Decision

The codebase should optimize for this order:

```text
1. Runs reliably.
2. Produces valid BTP artifacts.
3. Supports strong professor evaluation.
4. Supports publication-grade analysis.
5. Supports advanced SSL/adversarial ablations.
```

Any change that threatens steps 1-3 should be postponed, even if it sounds more publishable.
