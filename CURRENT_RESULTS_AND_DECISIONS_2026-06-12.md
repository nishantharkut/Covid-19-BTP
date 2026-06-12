# COVID Audio BTP: Current Results, Decisions, and Status

Date: 2026-06-12  
Repository checkpoint inspected: `f3f5038 add latest results - new models added`  
Project: `Covid-19-BTP / covid_audio_btp`  
Status: BTP-grade technical pipeline completed; publication framing should be reliability/domain-shift focused.

---

## 1. Executive Summary

The project has moved from a simple COVID cough-classification idea into a reliability-aware respiratory-audio screening audit.

The current results support this conclusion:

> A calibrated multimodal Coswara pipeline performs strongly under internal participant-level evaluation, but its learned audio signal does not transfer reliably to COUGHVID. This shows that crowdsourced COVID respiratory-audio models are highly vulnerable to dataset shift, metadata confounding, recording protocol mismatch, and label noise.

This is the correct research framing. The project should not be presented as a deployment-ready COVID diagnostic model.

---

## 2. What Has Been Completed

### 2.1 Coswara Pipeline

Completed:

- Coswara dataset cloned and extracted successfully.
- Dataset layout audited.
- Coswara metadata cleaned.
- Participant-level train/validation/test split created.
- Audio quality audit completed.
- Feature extraction completed.
- Classical ML baselines trained.
- Branch calibration completed.
- Late fusion completed.
- Quality-weighted fusion completed.
- Abstention analysis completed.
- Bootstrap confidence intervals generated.
- Metadata-only baseline generated.
- Confounding/matching checks generated.
- Paper metric tables generated.
- Experiment manifest generated.
- Artifact validation passed with only expected unknown-label warning.

Important correction:

- The validation leakage issue was fixed.
- Fusion weights now use `data/outputs/metrics/ml_validation_metrics.csv`.
- `ml_validation_metrics.csv` contains `metric_split = validation`.
- Old leaked fusion results must not be cited.

### 2.2 COUGHVID Pipeline

Completed:

- COUGHVID v3 zip extracted.
- COUGHVID index built.
- COUGHVID MFCC/acoustic features extracted.
- COUGHVID smoke test passed.
- COUGHVID full labeled feature set built.
- Coswara-to-COUGHVID external validation completed.
- Feature-shift report generated.
- COUGHVID internal baseline completed.

### 2.3 Rescue / Stronger Model Grid

Completed:

- Added external model grid script:
  - `scripts/25_run_external_model_grid.py`
- Added COUGHVID internal baseline script:
  - `scripts/26_run_coughvid_internal_baseline.py`
- Added shared rescue logic:
  - `src/covid_audio_btp/rescue_experiments.py`
- Added tests:
  - `tests/test_rescue_experiments.py`
- Added optional dependency support for:
  - XGBoost
  - LightGBM
  - CatBoost
  - OpenSMILE
  - SHAP
- Added paper-table support for new result files.
- Fixed paper table so `feature_strategy` and `calibration_method` appear in external-grid rows.

Verification completed:

- Focused rescue tests passed.
- Full test suite passed.
- Synthetic end-to-end smoke tests passed.
- Final pushed repo contains rescue code, tests, result bundle, and notebook.

---

## 3. Current Final Results

### 3.1 Coswara Internal Baselines

| Model | Modality | AUROC | AUPRC | Balanced Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Cough | 0.796 | 0.672 | 0.739 | 0.648 |
| Random Forest | Cough | 0.804 | 0.717 | 0.657 | 0.493 |
| Logistic Regression | Breath | 0.762 | 0.576 | 0.690 | 0.595 |
| Random Forest | Breath | 0.797 | 0.675 | 0.678 | 0.543 |
| Logistic Regression | Speech | 0.748 | 0.602 | 0.674 | 0.578 |
| Random Forest | Speech | 0.756 | 0.617 | 0.643 | 0.473 |
| Compact CNN | Cough | 0.750 | 0.559 | 0.669 | 0.578 |

Interpretation:

- Cough is the strongest single classical branch.
- Breath is close to cough.
- Speech is useful but weaker.
- The compact CNN is not the champion, but it provides a deep-learning baseline.

### 3.2 Coswara Fusion Results

| Fusion Method | AUROC | AUPRC | Balanced Accuracy | F1 |
|---|---:|---:|---:|---:|
| Uniform mean | 0.881 | 0.841 | 0.823 | 0.760 |
| Validation-weighted AUPRC fusion | 0.882 | 0.844 | 0.813 | 0.749 |
| Quality-weighted AUPRC fusion | 0.879 | 0.832 | 0.804 | 0.731 |

Interpretation:

- Multimodal fusion improves over any single branch internally.
- The best internal result is validation-weighted fusion: AUROC about 0.882 and AUPRC about 0.844.
- Quality weighting does not improve over validation-weighted fusion, but remains competitive.

### 3.3 Metadata Baseline

| Model | Input | AUROC | AUPRC | Balanced Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Metadata | 0.936 | 0.922 | 0.891 | 0.854 |

Interpretation:

- Metadata predicts the label better than audio.
- This is a major confounding warning.
- The project should explicitly state that audio-only COVID screening can be inflated by metadata/symptom/demographic shortcuts.

### 3.4 COUGHVID Internal Baseline

COUGHVID labeled feature set:

- Total labeled rows: 8331
- Negative: 8046
- Positive: 285
- Positive prevalence: about 3.42%

| Model | AUROC | AUPRC | Balanced Accuracy | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.714 | 0.138 | 0.617 | 0.122 |
| Random Forest | 0.767 | 0.120 | 0.715 | 0.166 |
| XGBoost | 0.754 | 0.138 | 0.686 | 0.133 |
| LightGBM | 0.781 | 0.178 | 0.698 | 0.180 |
| CatBoost | 0.752 | 0.134 | 0.679 | 0.115 |

Best COUGHVID internal model:

- LightGBM
- AUROC: 0.781
- AUPRC: 0.178

Interpretation:

- COUGHVID is learnable internally, but it is hard and highly imbalanced.
- AUPRC must be interpreted relative to the prevalence baseline of about 0.034.
- LightGBM improves substantially over prevalence, but F1 remains low because positives are rare.

### 3.5 Coswara-to-COUGHVID External Transfer

Best external model-grid result:

| Model | Feature Strategy | AUROC | AUPRC | Balanced Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | top_stable_50 | 0.535 | 0.042 | 0.532 | 0.072 |

Other high rows:

| Model | Feature Strategy | AUROC | AUPRC |
|---|---|---:|---:|
| CatBoost | all | 0.526 | 0.039 |
| Random Forest | all | 0.524 | 0.039 |
| LightGBM | all | 0.518 | 0.039 |
| XGBoost | all | 0.508 | 0.037 |

Interpretation:

- Coswara-trained models fail to transfer meaningfully to COUGHVID.
- Stronger classifiers did not solve the transfer problem.
- Feature pruning gave a small AUROC improvement for logistic regression, but not enough for deployment-grade generalization.
- This is not a model-capacity-only problem. It is likely dataset shift, label mismatch, recording protocol mismatch, and confounding.

### 3.6 Feature Shift

Feature shift summary:

| Metric | Value |
|---|---:|
| Number of compared features | 253 |
| High-shift features | 21 |
| Maximum absolute standardized mean difference | 2.646 |
| SMD threshold | 0.5 |

Interpretation:

- There is measurable covariate shift between Coswara and COUGHVID feature distributions.
- This supports the observed external transfer collapse.

---

## 4. Key Decisions Made

### Decision 1: Use Reliability Framing, Not Accuracy-Chasing

Original weak framing rejected:

> CNN detects COVID from cough with high accuracy.

Final framing accepted:

> Leakage-controlled and confidence-calibrated multimodal respiratory-audio screening under cross-dataset shift.

Reason:

- COVID-only detection is overclaimed in many papers.
- External validation shows the model does not generalize strongly.
- Reliability, calibration, confounding, and domain shift are more defensible.

### Decision 2: Keep Coswara As Primary Dataset

Reason:

- Publicly available.
- Includes cough, breath, vowel, and speech/counting recordings.
- Supports same-participant multimodal analysis.

### Decision 3: Use COUGHVID As External Cough Validation

Reason:

- Public external dataset.
- Different acquisition protocol.
- Useful to test whether Coswara-learned cough signal transfers.

Important limitation:

- COUGHVID labels are noisy and highly imbalanced.
- It should be framed as robustness/external stress testing, not perfect clinical ground truth.

### Decision 4: Fix Validation Leakage Before Reporting

Reason:

- Fusion weights must not use test metrics.
- Validation-only metrics are required for defensible model selection.

Current status:

- Fixed.
- `ml_validation_metrics.csv` is used for fusion weights.

### Decision 5: Add Metadata Baseline

Reason:

- To test whether non-audio variables predict labels.
- To avoid overclaiming audio biomarkers.

Finding:

- Metadata AUROC is 0.936, higher than audio fusion.

Conclusion:

- Confounding is a central result, not a side note.

### Decision 6: Add Stronger External Model Grid

Models added:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Feature strategies:

- all features
- drop high-shift features
- top stable 50
- top stable 80
- top stable 120

Finding:

- Stronger models did not solve external transfer.

Conclusion:

- The external failure is likely due to domain/label/protocol shift, not just weak classifiers.

### Decision 7: Keep Modern Embeddings As Future Work

Current feature family:

- MFCC
- delta MFCC
- delta-delta MFCC
- RMS
- zero-crossing rate
- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral flatness
- duration/event fields

Decision:

- This is acceptable for BTP and audit-oriented classical baselines.
- For publication strengthening, add OpenSMILE and/or pretrained audio embeddings.

Recommended future additions:

1. OpenSMILE eGeMAPS / ComParE.
2. PANN/CNN14 embeddings.
3. wav2vec2 / HuBERT / Whisper embeddings.
4. Domain-adversarial learning / GRL if targeting stronger publication.

---

## 5. Comparison With Original Plan Papers

### B1. JMIR 2025 Drift-Adaptive Cough-Audio Framework

Original plan role:

- Main base paper.
- Cough-only drift framework.
- Reported development AUC around 0.691 on COVID-19 Sounds and 0.668 on Coswara.
- Postdevelopment dropped to around 0.607 and 0.597.

Our comparison:

- Our internal Coswara fusion AUROC is 0.882, higher than their static baseline.
- We did not implement temporal unsupervised domain adaptation or active learning.
- Our COUGHVID external AUROC of 0.535 strongly supports the same reliability problem: models degrade under shift.

Position:

- We extend the reliability idea from temporal drift to cross-dataset shift and multimodal fusion.

### B2. ESWA 2026 Deep Neural Decision Tree / Forest

Original plan role:

- Cross-dataset robustness reference.
- Reported AUC values around 0.92 to 0.99, combined setting around 0.97.

Our comparison:

- Our external transfer AUROC is far lower at 0.535.
- We do not beat this paper on headline accuracy.
- Our pipeline is simpler but more transparent, calibrated, and explicitly shows transfer failure.

Position:

- We should not claim SOTA.
- We can say our work provides a stricter audit-style result and exposes failure under blind Coswara-to-COUGHVID transfer.

### B3. IEEE-BHI 2024 Confidence-Calibrated Screening

Original plan role:

- Calibration base paper.
- Reported ENCL-DNN AUROC 0.834 on Coswara and 0.854 on Cambridge.
- Reported ECE reductions.

Our comparison:

- Our Coswara fusion AUROC 0.882 is higher than their reported Coswara AUROC.
- We use branch calibration and report ECE/Brier/NLL.
- We add multimodal fusion and COUGHVID external testing.

Position:

- This is one of our strongest comparisons.
- Our work aligns well with calibration-focused screening.

### B4. Hierarchical Spectrogram Transformer

Original plan role:

- Advanced deep architecture reference.
- Reported over 83% AUC.

Our comparison:

- Our internal Coswara fusion AUROC 0.882 is competitive internally.
- We did not implement transformer architecture.
- Our novelty is evaluation discipline, not architecture.

Position:

- Cite as advanced architecture future work.
- Do not claim architectural novelty.

### B5. SympCoughNet 2025

Original plan role:

- Symptom-assisted recent paper.
- Reported accuracy 89.30%, AUROC 94.74%, PR 91.62%.

Our comparison:

- Our audio-only fusion is lower.
- Our metadata-only baseline AUROC 0.936 shows why symptom/metadata-assisted models can achieve high metrics.
- We use symptoms/metadata as confounding audit, not as central input claim.

Position:

- Our work is more conservative.
- We can use this paper to argue that symptom information is powerful but must be handled carefully.

### B6. PLOS ONE 2025 Speech/Vowel Paper

Original plan role:

- Supports speech/vowel experiments.
- Reported speech/vowel accuracy around 75-76%.

Our comparison:

- Our speech branch AUROC is 0.756.
- Speech is useful, but cough and breath are stronger in our setup.
- Fusion improves over speech alone.

Position:

- Confirms speech is a valid modality but not sufficient alone.

### B7. Nature Machine Intelligence 2024 Caution Paper

Original plan role:

- Critical caution paper.
- Warns that audio models may not outperform symptom checkers and may learn confounders.

Our comparison:

- Our metadata baseline AUROC 0.936 is direct evidence of this issue.
- Audio fusion AUROC 0.882 is lower than metadata-only prediction.
- Cross-dataset transfer collapses.

Position:

- This paper strongly supports our final framing.
- Our project should explicitly cite this as motivation for confounding and external validation.

### B8. Coswara Dataset Paper

Original plan role:

- Primary dataset.
- Supports cough, breath, vowel, and speech/counting categories.

Our comparison:

- We used the intended multimodal structure.
- We produced modality-specific and fusion comparisons.
- We included quality and participant-level split controls.

Position:

- Strong alignment with dataset design.

### B9. COVID-19 Sounds NeurIPS 2021

Original plan role:

- Large multimodal dataset reference.
- Not used due access constraints.

Our comparison:

- No direct metric comparison.
- Use as field context and future external validation target.

### B10. COUGHVID Scientific Data 2021

Original plan role:

- External cough validation dataset.
- Labels noisy, quality tools available.

Our comparison:

- We completed the planned external validation.
- COUGHVID internal AUROC reached 0.781.
- Coswara-to-COUGHVID external AUROC reached only 0.535.

Position:

- This is one of the strongest completed extensions from the original plan.

---

## 6. Novelty Statement

The final novelty should be stated as:

> Unlike accuracy-focused COVID cough classifiers, this project implements a leakage-controlled, calibrated, multimodal respiratory-audio pipeline and demonstrates that strong internal Coswara performance collapses under COUGHVID external validation. Metadata-only and feature-shift analyses show that confounding and dataset shift are central reliability risks in crowdsourced respiratory-audio screening.

Shorter version:

> A reliability-first multimodal respiratory-audio audit showing that internal COVID-audio performance does not imply external robustness.

---

## 7. What We Should Not Claim

Do not claim:

- This is a clinical diagnostic tool.
- The model detects COVID reliably in real-world deployment.
- The project beats SOTA.
- MFCC/boosted trees solve COVID cough detection.
- External validation succeeded.

---

## 8. What We Can Safely Claim

Safe claims:

- The pipeline is leakage-controlled at participant level.
- Calibration and validation-only fusion were implemented.
- Coswara internal multimodal fusion achieved strong AUROC/AUPRC.
- Metadata-only prediction outperformed audio, showing confounding risk.
- COUGHVID was moderately learnable internally.
- Coswara-to-COUGHVID external transfer was near chance.
- Feature-shift analysis found measurable distribution shift.
- The final contribution is reliability evaluation, not deployment-grade diagnosis.

---

## 9. Current BTP Completion Status

BTP status: complete enough for a strong submission.

Completed BTP deliverables:

- Working reproducible pipeline.
- Dataset processing.
- Classical ML baselines.
- CNN baseline.
- Fusion.
- Calibration.
- Quality audit.
- Abstention.
- Confounding audit.
- External validation.
- Feature-shift analysis.
- Tests.
- Final result bundle.
- Notebook workflow.

Remaining BTP/report work:

- Write final report.
- Build result figures/tables cleanly.
- Prepare slides.
- Decide whether to include or exclude large notebook outputs in final submission.
- Add clear disclaimer: research prototype only, not clinical diagnosis.

---

## 10. Publication-Grade Next Steps

If continuing beyond BTP, the most useful technical extensions are:

1. Add OpenSMILE eGeMAPS / ComParE features.
2. Add pretrained audio embeddings such as PANN/CNN14.
3. Add wav2vec2 / HuBERT / Whisper embeddings if compute allows.
4. Rerun the same three evaluations:
   - Coswara internal.
   - COUGHVID internal.
   - Coswara-to-COUGHVID external.
5. Compare whether modern representations improve external robustness.
6. If external transfer still fails, frame the paper as strong evidence of domain shift.
7. If external transfer improves, report the improved representation as an extension.

Recommended publication framing:

> Leakage-Controlled Evaluation of Multimodal Respiratory Audio Screening Under Cross-Dataset Shift

or

> An Engineering Audit of Confounding and Domain Shift in Crowdsourced Respiratory-Audio COVID Screening

---

## 11. Final Current Verdict

The project is not a SOTA COVID detector.

It is a strong reliability-aware BTP and a plausible tier-2 applied publication foundation if written honestly.

The strongest scientific result is not the internal AUROC. The strongest result is the contrast:

- Coswara internal fusion: AUROC about 0.882.
- COUGHVID internal: AUROC about 0.781.
- Coswara-to-COUGHVID external: AUROC about 0.535.

This contrast demonstrates that internal performance, even when calibrated and multimodal, does not guarantee external respiratory-audio robustness.
