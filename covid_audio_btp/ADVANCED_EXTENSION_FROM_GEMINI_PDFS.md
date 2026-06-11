# Advanced Extension From Gemini/ChatGPT PDF Review

Last updated: 2026-06-10

This file records the advanced ideas from the two uploaded PDFs and the later Gemini conversation. These are not first-run requirements. They are the post-baseline publication upgrade path.

Reviewed source files:

```text
.covid_audio_btp_private/gemini_plan_review/adversarial_debiasing_respiratory_diagnostics.pdf
.covid_audio_btp_private/gemini_plan_review/methodological_audit_debiasing_framework.pdf
.covid_audio_btp_private/gemini_conversation_review/gemini_complete_conversation
```

## Final Decision

Do not replace the current implementation.

The current MFCC/CNN/calibration/fusion/confounding pipeline is the baseline and control condition. The PDF ideas become an optional advanced ablation layer after the local Coswara baseline and cough-only COUGHVID checks produce evidence.

## Why The PDF Plan Is Not Used Directly

The PDFs propose:

- Wav2Vec/HuBERT frozen or partially frozen SSL encoders;
- cross-modal attention over cough, breath, and speech embeddings;
- Gradient Reversal Layer adversarial debiasing;
- acoustic-domain/confounder proxy labels;
- cross-dataset calibration decay analysis;
- uncertainty/rejection instead of standard saliency/XAI.

This is useful but too risky as the first implementation path because:

- COUGHVID is cough-only, so it cannot support full cough-breath-speech external validation.
- GRL can erase disease-relevant acoustic information if the confounder proxy overlaps with true respiratory signal.
- Wav2Vec/HuBERT is not guaranteed to outperform MFCC/acoustic features on Coswara.
- Full SSL/GRL training requires more GPU, more time, and more debugging than the baseline.
- Peer-review value comes from ablation evidence, not from claiming a complex architecture is automatically better.

## New Research Question

The advanced extension should answer:

```text
Can respiratory-audio screening models remain calibrated and useful under acoustic-domain shift, and does adding SSL/adversarial debiasing improve robustness enough to justify its complexity?
```

## What Must Happen Before This Extension

Do not start this extension until these files exist from the local baseline run:

```text
reports/tables/coswara_layout_audit.csv
reports/tables/validation_issues.csv
data/interim/coswara_index.csv
data/processed/metadata_clean.csv
data/processed/audio_quality.csv
data/outputs/metrics/ml_baseline_metrics.csv
data/outputs/metrics/calibration_metrics.csv
data/outputs/metrics/fusion_metrics.csv
```

Prefer also having:

```text
data/interim/coughvid_index.csv
data/processed/coughvid_features_mfcc.csv
data/outputs/metrics/cross_dataset_metrics.csv
```

## Advanced Module Plan

Add these only after baseline evidence exists:

```text
scripts/25_acoustic_domain_proxy.py
scripts/26_extract_ssl_embeddings.py
scripts/27_train_adversarial_ssl.py
scripts/28_compare_adversarial_vs_baselines.py

src/covid_audio_btp/domain_proxy.py
src/covid_audio_btp/ssl_embeddings.py
src/covid_audio_btp/adversarial_ssl.py
src/covid_audio_btp/advanced_ablation.py
```

## Module 25: Acoustic Domain Proxy

Purpose:

Generate objective domain/confounder labels that are comparable across Coswara and COUGHVID.

Do not use raw device/browser text as the main confounder label because Coswara and COUGHVID metadata schemas differ.

Proxy features:

```text
sample_rate_original
codec or extension
duration_sec
silence_ratio
clipping_ratio
snr_proxy
rms_mean
rms_std
spectral_centroid_mean
spectral_flatness_mean
zero_crossing_rate_mean
manual_quality_label when available
dataset source
```

Outputs:

```text
data/processed/acoustic_domain_proxy.csv
reports/tables/acoustic_domain_proxy_summary.csv
reports/figures/acoustic_domain_proxy_clusters.png
```

Minimum tests:

- proxy labels are deterministic for a fixed random seed;
- proxy labels contain no clinical label values;
- every usable recording receives a proxy label or explicit `unknown`;
- Coswara and COUGHVID can both be represented in the same proxy schema.

## Module 26: SSL Embeddings

Purpose:

Extract frozen Wav2Vec/HuBERT-style embeddings as an ablation against MFCC/acoustic features.

Important constraints:

- Run only after baseline works.
- Use fixed duration or pooling to avoid padding blowup.
- Start with cough-only embeddings first.
- Treat layer choice as an ablation, not a magic constant.

Initial settings:

```text
sample_rate = 16000
max_duration_sec = 4.0
pooling = mean over valid frames
modalities_first = cough only
layer_idx candidates = 4, 6, 8, 12 only if compute permits
```

Outputs:

```text
data/processed/ssl_embeddings_cough.parquet
data/outputs/metrics/ssl_embedding_baseline_metrics.csv
```

Minimum tests:

- waveform shorter than 4 seconds is padded/masked safely;
- waveform longer than 4 seconds is chunked or cropped deterministically;
- embedding dimensions are stable;
- missing modalities do not crash the dataset loader.

## Module 27: Adversarial SSL

Purpose:

Train a clinical classifier with an adversarial confounder/domain head using Gradient Reversal Layer.

This is an ablation, not the main project.

Architecture:

```text
SSL embedding encoder or frozen embedding input
clinical head -> positive/negative prediction
confounder head -> acoustic proxy/domain prediction
GRL between representation and confounder head
```

Safety rules:

- GRL lambda must ramp gradually.
- Compare against the same SSL model without GRL.
- Track whether clinical performance collapses.
- Track whether confounder predictability actually decreases.
- Do not claim debiasing if clinical signal disappears.

Outputs:

```text
data/outputs/metrics/adversarial_ssl_metrics.csv
data/outputs/metrics/adversarial_ssl_calibration.csv
data/outputs/metrics/adversarial_ssl_proxy_leakage.csv
```

Minimum tests:

- GRL flips gradient sign in a controlled tensor test;
- lambda schedule starts low and increases;
- training loop logs clinical loss, confounder loss, ECE, and AUROC;
- evaluation can run without COUGHVID.

## Module 28: Compare Against Baselines

Purpose:

Produce the ablation matrix that decides whether advanced complexity is justified.

Required comparisons:

```text
MFCC/acoustic classical baseline
compact CNN if available
SSL frozen embedding baseline
SSL + acoustic proxy GRL
quality-weighted calibrated fusion
cough-only Coswara -> COUGHVID external test
```

Metrics:

```text
AUROC
AUPRC
sensitivity
specificity
F1
Brier score
ECE
NLL when available
coverage/abstention metrics
paired bootstrap confidence intervals
generalization drop from in-domain to external cough-only validation
```

Outputs:

```text
reports/tables/advanced_ablation_matrix.csv
reports/tables/advanced_compute_cost_table.csv
reports/figures/advanced_calibration_decay.png
reports/figures/domain_proxy_leakage_before_after_grl.png
```

## Correct COUGHVID Handling

COUGHVID is cough-only.

Allowed:

```text
Coswara cough-only model -> COUGHVID cough external validation
SSL cough-only embeddings on Coswara and COUGHVID
domain proxy analysis across Coswara cough and COUGHVID cough
calibration decay across Coswara cough and COUGHVID cough
```

Not allowed:

```text
Full cough-breath-speech external validation on COUGHVID
Claiming multimodal cross-dataset validation using COUGHVID
```

## XAI Decision

Do not make Grad-CAM/SHAP/saliency the core novelty.

Reason:

In crowdsourced respiratory audio, post-hoc XAI often explains the model's shortcut, such as noise floor, codec, microphone, or background artifact. For this project, reliability evidence is stronger:

```text
calibration
uncertainty
abstention
domain-shift metrics
confounding checks
paired ablations
```

XAI can be a small optional appendix only after confounding is controlled.

## Compute Gate

Do not start SSL/GRL unless the local machine has:

```text
GPU: 12-16 GB VRAM minimum
RAM: 32 GB preferred
Disk: 80-150 GB free
```

If no such GPU is available, keep the project as:

```text
baseline reliability pipeline + cough-only external validation + acoustic-domain proxy analysis
```

This is still a defensible project.

For the known lab machine with NVIDIA T1000 8 GB and 19 GB RAM:

Allowed after the baseline succeeds:

```text
frozen cough-only SSL embedding extraction
batch size 1-4
short fixed/chunked clips
lightweight classifiers on saved embeddings
```

Not recommended on that machine:

```text
end-to-end Wav2Vec/HuBERT fine-tuning
large SSL encoder plus GRL training
multi-dataset transformer training
```

This keeps the publication upgrade realistic without damaging the A-grade BTP path.

## Manuscript Framing

Do not claim:

```text
We built a perfect COVID detector.
SSL/GRL is guaranteed to remove bias.
```

Claim:

```text
We perform an ablation-driven reliability audit of respiratory-audio screening under quality variation, confounding, calibration uncertainty, and cough-only cross-dataset shift.
```

If SSL/GRL improves results:

```text
Adversarial SSL improved cross-domain calibration/generalization under cough-only external validation.
```

If SSL/GRL does not improve results:

```text
The added complexity did not justify itself, and calibrated lightweight baselines were more dependable under the available data constraints.
```

Both outcomes are scientifically usable.

