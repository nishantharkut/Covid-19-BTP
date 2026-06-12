# COVID Audio BTP Final Results Report

## Executive Position

The implementation supports a defensible BTP and potential robustness-oriented publication story. It is not a clinically deployable diagnostic model. It does not support claiming clinical deployment. The strongest finding is that internal audio performance can look promising while external transfer, confounding, and calibration analyses reveal major reliability limits.

## Pipeline Architecture

The pipeline is organized as an evidence-first audio ML workflow:

1. Build cleaned Coswara metadata, quality labels, participant-level splits, and audio features.
2. Train internal audio baselines and calibrated fusion models on leakage-controlled splits.
3. Compare handcrafted and learned representations: MFCC, OpenSMILE eGeMAPSv02, BEATs, and PANNs CNN14.
4. Evaluate source-trained models on COUGHVID as external transfer validation.
5. Run COUGHVID internal baselines to separate representation capacity from dataset shift.
6. Audit metadata confounding, then evaluate quality-weighted audio after inverse-propensity weighting.
7. Report clinical operating points and calibration-under-shift to avoid relying only on AUROC.

This architecture supports a robustness and external-validation study. It does not support a clinical-deployment claim.

## Decision Log

- MFCC was retained as the traditional handcrafted baseline because it anchors the project against the original pipeline.
- OpenSMILE eGeMAPSv02 was added as a stronger handcrafted clinical-acoustic baseline.
- BEATs was selected as the main general-audio learned embedding model because it directly tests whether modern transformer audio representations help.
- PANNs CNN14 was added as the CNN audio-embedding comparison, so representation architecture is not reduced to a single transformer result.
- wav2vec2 was treated as a lower-priority speech-biased option because the current scientific question is cough/audio representation robustness, not speech recognition transfer.
- External validation was prioritized over leaderboard tuning because Coswara-to-COUGHVID transfer is the central generalization test.
- Metadata confounding, IPW control, operating points, and calibration shift were added because medical-audio claims require reliability and bias checks, not only accuracy metrics.

## Quantitative Results

- Best external transfer: auroc=0.553.
- Best COUGHVID internal baseline: auroc=0.781.
- Strongest metadata-only confounding audit: auroc=0.964.
- IPW-controlled audio: auroc=0.780.

### External Transfer
| claim_id | comparison | primary_metric | primary_value | secondary_metrics | n_samples |
| --- | --- | --- | --- | --- | --- |
| external_transfer_mfcc_best | MFCC / logistic_regression / top_stable_50 evaluated on COUGHVID external labels | auroc | 0.535 | auprc=0.042; balanced_accuracy=0.532; sensitivity=0.495; specificity=0.569; f1=0.072; brier=0.115; ece=0.285; nll=0.411 | 8331 |
| external_transfer_opensmile_egemaps_best | OpenSMILE eGeMAPSv02 / logistic_regression / drop_high_shift evaluated on COUGHVID external labels | auroc | 0.552 | auprc=0.039; balanced_accuracy=0.517; sensitivity=0.211; specificity=0.823; f1=0.068; brier=0.080; ece=0.180; nll=0.303 | 8331 |
| external_transfer_beats_best | BEATs / logistic_regression / drop_high_shift evaluated on COUGHVID external labels | auroc | 0.553 | auprc=0.039; balanced_accuracy=0.515; sensitivity=0.186; specificity=0.845; f1=0.067; brier=0.091; ece=0.201; nll=0.331 | 8331 |
| external_transfer_panns_best | PANNs CNN14 / logistic_regression / drop_high_shift evaluated on COUGHVID external labels | auroc | 0.502 | auprc=0.035; balanced_accuracy=0.493; sensitivity=0.214; specificity=0.771; f1=0.056; brier=0.115; ece=0.271; nll=0.404 | 8331 |

### Internal COUGHVID Baselines
| claim_id | comparison | primary_metric | primary_value | secondary_metrics | n_samples |
| --- | --- | --- | --- | --- | --- |
| coughvid_internal_mfcc_best | MFCC / lightgbm trained and tested within COUGHVID split | auroc | 0.781 | auprc=0.178; balanced_accuracy=0.698; sensitivity=0.561; specificity=0.834; f1=0.180; brier=0.033; ece=0.000; nll=0.147 | 1667 |
| coughvid_internal_opensmile_egemaps_best | OpenSMILE eGeMAPSv02 / logistic_regression trained and tested within COUGHVID split | auroc | 0.763 | auprc=0.120; balanced_accuracy=0.689; sensitivity=0.561; specificity=0.817; f1=0.167; brier=0.032; ece=0.001; nll=0.136 | 1667 |
| coughvid_internal_beats_best | BEATs / lightgbm trained and tested within COUGHVID split | auroc | 0.756 | auprc=0.143; balanced_accuracy=0.663; sensitivity=0.632; specificity=0.695; f1=0.123; brier=0.033; ece=0.000; nll=0.148 | 1667 |
| coughvid_internal_panns_best | PANNs CNN14 / random_forest trained and tested within COUGHVID split | auroc | 0.652 | auprc=0.074; balanced_accuracy=0.602; sensitivity=0.561; specificity=0.642; f1=0.096; brier=0.033; ece=0.000; nll=0.149 | 1667 |

### Confounding And Controlled Audio
| claim_id | primary_metric | primary_value | secondary_metrics | n_samples | evidence_direction |
| --- | --- | --- | --- | --- | --- |
| metadata_confounding_full_safe_metadata | auroc | 0.964 | auprc=0.928; balanced_accuracy=0.890; sensitivity=0.858; specificity=0.922; f1=0.849; brier=0.077; ece=0.064; nll=0.257 | 2862 | cautionary |
| metadata_confounding_symptoms_only | auroc | 0.932 | auprc=0.898; balanced_accuracy=0.912; sensitivity=0.893; specificity=0.930; f1=0.876; brier=0.075; ece=0.096; nll=0.281 | 2862 | cautionary |
| metadata_confounding_demographic_protocol_only | auroc | 0.914 | auprc=0.737; balanced_accuracy=0.827; sensitivity=0.864; specificity=0.790; f1=0.751; brier=0.122; ece=0.117; nll=0.423 | 2862 | cautionary |
| confounding_controlled_audio_ipw | auroc | 0.780 | auprc=0.537; balanced_accuracy=0.721; sensitivity=0.679; specificity=0.763; f1=0.450; brier=0.147; ece=0.210; nll=0.478; effective_sample_size=130.439 | 318 | qualified_supportive |

### Clinical Operating Points
| claim_id | comparison | primary_metric | primary_value | secondary_metrics | n_samples |
| --- | --- | --- | --- | --- | --- |
| clinical_fusion_specificity_0_800 | specificity>=0.800 | sensitivity | 0.806 | threshold=0.337; sensitivity=0.806; specificity=0.814; precision=0.675; npv=0.897; f1=0.735; balanced_accuracy=0.810 | 318 |
| clinical_fusion_specificity_0_900 | specificity>=0.900 | sensitivity | 0.699 | threshold=0.356; sensitivity=0.699; specificity=0.907; precision=0.783; npv=0.863; f1=0.738; balanced_accuracy=0.803 | 318 |
| clinical_fusion_specificity_0_950 | specificity>=0.950 | sensitivity | 0.612 | threshold=0.374; sensitivity=0.612; specificity=0.958; precision=0.875; npv=0.837; f1=0.720; balanced_accuracy=0.785 | 318 |
| clinical_fusion_sensitivity_0_900 | sensitivity>=0.900 | specificity | 0.591 | threshold=0.313; sensitivity=0.903; specificity=0.591; precision=0.514; npv=0.927; f1=0.655; balanced_accuracy=0.747 | 318 |

### Calibration Under Shift
| claim_id | comparison | primary_metric | primary_value | secondary_metrics | n_samples |
| --- | --- | --- | --- | --- | --- |
| calibration_external_transfer_worst | external_model_grid_predictions | ece | 0.286 | observed_prevalence=0.034; mean_probability=0.321; calibration_gap=0.286; mce=0.464; brier=0.120; nll=0.419 | 8331 |

## Quantitative Evidence Matrix
| claim_id | evidence_type | primary_metric | primary_value | n_samples | evidence_direction |
| --- | --- | --- | --- | --- | --- |
| internal_quality_weighted_fusion | internal_audio | auroc | 0.879 | 318 | supportive |
| external_transfer_mfcc_best | external_transfer | auroc | 0.535 | 8331 | cautionary |
| external_transfer_opensmile_egemaps_best | external_transfer | auroc | 0.552 | 8331 | cautionary |
| external_transfer_beats_best | external_transfer | auroc | 0.553 | 8331 | cautionary |
| external_transfer_panns_best | external_transfer | auroc | 0.502 | 8331 | cautionary |
| coughvid_internal_mfcc_best | internal_baseline | auroc | 0.781 | 1667 | context |
| coughvid_internal_opensmile_egemaps_best | internal_baseline | auroc | 0.763 | 1667 | context |
| coughvid_internal_beats_best | internal_baseline | auroc | 0.756 | 1667 | context |
| coughvid_internal_panns_best | internal_baseline | auroc | 0.652 | 1667 | context |
| metadata_confounding_full_safe_metadata | metadata_confounding | auroc | 0.964 | 2862 | cautionary |
| metadata_confounding_symptoms_only | metadata_confounding | auroc | 0.932 | 2862 | cautionary |
| metadata_confounding_demographic_protocol_only | metadata_confounding | auroc | 0.914 | 2862 | cautionary |
| confounding_controlled_audio_ipw | confounding_controlled_audio | auroc | 0.780 | 318 | qualified_supportive |
| clinical_fusion_specificity_0_800 | clinical_operating_point | sensitivity | 0.806 | 318 | operational_context |
| clinical_fusion_specificity_0_900 | clinical_operating_point | sensitivity | 0.699 | 318 | operational_context |
| clinical_fusion_specificity_0_950 | clinical_operating_point | sensitivity | 0.612 | 318 | operational_context |
| clinical_fusion_sensitivity_0_900 | clinical_operating_point | specificity | 0.591 | 318 | operational_context |
| calibration_quality_weighted_fusion | calibration | ece | 0.150 | 318 | mixed |
| calibration_external_transfer_worst | calibration_under_shift | ece | 0.286 | 8331 | cautionary |

## Interpretation

The current evidence does not show a deployable COVID screening model. The stronger and more defensible interpretation is that internal audio models can learn label-associated signal, but external transfer and calibration degrade substantially across datasets. The contrast between internal COUGHVID performance and Coswara-to-COUGHVID transfer supports domain shift as a central finding.

The metadata audits are especially important: symptoms, demographics, recording protocol, and related metadata predict the label strongly. Therefore, any audio-only claim must be framed as association under measured controls rather than as a causal COVID biomarker.

## Novelty

The novelty is not a new neural architecture. The contribution is a controlled, evidence-driven evaluation layer around COVID audio classification:

- Direct comparison of traditional handcrafted, stronger handcrafted, transformer audio, and CNN audio representations.
- Explicit internal versus external validation contrast on Coswara and COUGHVID.
- Metadata-only confounding audit showing how non-audio variables can explain labels.
- Inverse-propensity weighted controlled audio evaluation.
- Clinical operating-point reporting instead of AUROC-only reporting.
- Calibration-under-shift analysis showing that external probabilities should not be interpreted as calibrated risk.

## Related-Paper Comparison

This is a conservative comparison against the exact source papers captured in the original project documents. It is not a claim that our metrics exceed prior work; it explains how our reliability checks differ from headline-accuracy studies.

| paper_id | title | source_year | role | datasets | method | reported_results | main_limitation | how_ours_compares | source_doc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | A Comprehensive Drift-Adaptive Framework for Sustaining Model Performance in COVID-19 Detection From Dynamic Cough Audio Data | JMIR, 2025 | Main base paper | COVID-19 Sounds; Coswara | Cough audio, mel spectrograms, VGGish features, chronological splits, MMD drift detection, CUSUM alerts, UDA and active learning | Development AUROC 69.1% on COVID-19 Sounds and 66.8% on Coswara; postdevelopment AUROC dropped to 60.7% and 59.7%; UDA improved balanced accuracy by roughly 10%-20% | Focuses on temporal drift; limited interpretation of causes, cross-dataset drift, and interdemographic variability; cough-only | We add Coswara-to-COUGHVID cross-dataset transfer, metadata confounding, operating points, and calibration-under-shift, but do not implement full UDA/active learning | references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md |
| P2 | Coswara: A respiratory sounds and symptoms dataset for remote screening of SARS-CoV-2 infection | Scientific Data, 2023 | Primary dataset paper | Coswara: 2635 individuals, 23,700 recordings, about 65 hours, 9 sound categories | Crowdsourced cough, breath, vowel, and counting recordings plus symptoms and metadata | Dataset paper; not a model leaderboard in our comparison table | Crowdsourced recordings and metadata-driven label associations require careful quality and confounding analysis | We use Coswara as the core source dataset with participant-level split, quality checks, metadata audit, and representation comparison | references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/verified_source_registry.md |
| P3 | COVID-19 Sounds: A Large-Scale Audio Dataset for Digital Respiratory Screening | NeurIPS Datasets and Benchmarks, 2021 | Large multimodal dataset reference | 53,449 audio samples, 552+ hours, 36,116 participants, breathing/cough/voice | Large-scale respiratory audio dataset with participant metadata | Dataset reference; raw access requires request/data agreement | Not instantly available for this implementation; cannot be required for reproducibility | We cite it for field context but rely on public Coswara plus COUGHVID for executable validation | references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md |
| P4 | The COUGHVID crowdsourcing dataset | Scientific Data, 2021 | External cough validation dataset | More than 25,000 crowdsourced cough recordings; more than 2800 expert-labeled recordings | Crowdsourced cough collection with metadata, cough detection, and quality tools | Dataset paper; expert agreement was limited for some labels | Noisy labels and crowdsourced quality make it an external robustness target, not perfect ground truth | We use COUGHVID only as external validation/internal baseline context and report weak transfer honestly | references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/verified_source_registry.md |
| P5 | Audio-based AI classifiers show no evidence of improved COVID-19 screening over simple symptoms checkers | Nature Machine Intelligence, 2024 | Required confounding critique | UK COVID-19 Vocal Audio Dataset with PCR-referenced cough, exhalation, and speech | Audio classifiers compared against symptom/checker baselines under realistic evaluation | Main conclusion: audio models may not outperform simple symptom checkers under realistic evaluation | Shows that apparent audio performance can reflect confounding signals, symptoms, and sampling differences | We add metadata confounding audits, symptom/demographic/protocol-only models, IPW-controlled audio evaluation, and conservative claims | references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md |
| P6 | Confidence-Calibrated Clinical Decision Support System for Reliable Respiratory Disease Screening | IEEE-BHI/OpenReview, 2024 | Calibration base paper | Coswara; Cambridge COVID-19 Sounds | MFCC features, DNN, ensemble-based confidence calibration, LIME-style interpretability | ENCL-DNN AUROC 0.834 on Coswara and 0.854 on Cambridge; ECE reduced by 50.0% on Coswara and 28.74% on Cambridge | Mostly cough-focused and not a full external shift/confounding pipeline | We include calibration metrics, Brier/NLL/ECE, calibration-under-shift, and avoid interpreting external probabilities as calibrated risks | references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/verified_source_registry.md |
| P7 | COVID-19 Detection From Respiratory Sounds With Hierarchical Spectrogram Transformers | IEEE JBHI, 2023/2024 | Advanced architecture reference | Crowdsourced respiratory sound datasets with cough and breathing sounds | Hierarchical Spectrogram Transformer with local-to-global attention | Reports over 83% AUC for COVID-19 detection from respiratory sounds | Compute-heavy transformer architecture; not necessary for BTech reproducibility | We compare modern learned embeddings through BEATs/PANNs and keep the main contribution on reliability rather than architecture novelty | references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md |
| P8 | SympCoughNet: symptom assisted audio-based COVID-19 detection | Frontiers in Digital Health, 2025 | Symptom-assisted recent paper | UK COVID-19 Vocal Audio Dataset; 72,999 participants and 25,766 PCR-positive cases reported in project notes | Log-mel spectrograms, CNN backbone, symptom-encoded channel attention, augmentation, VAD/noise preprocessing | Accuracy 89.30%, AUROC 94.74%, PR 91.62% | Symptom-assisted performance can be dominated by symptom metadata rather than audio-specific signal | We use symptoms as confounding/audit metadata rather than central diagnostic input | references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md |
| P9 | Speech-based respiratory diagnostics: A study on COVID-19 detection with machine learning | PLOS ONE, 2025 | Speech/vowel modality reference | Coswara vowel sounds /a/, /e/, and /o/ | ITU-T P.56 normalization, OpenSMILE 1582-dimensional features, RF/SVM/Decision Tree/ANN, feature selection | Random Forest with ANOVA-selected features; accuracy around 76.47% for vowel /a/ and 75.54% for /a/+/o/ | Speech-only performance is moderate and task-specific | We prioritize cough plus representation robustness and keep speech-biased wav2vec2 as lower priority for the current cough transfer story | references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md |
| P10 | Robust COVID-19 detection from cough sounds using deep neural decision tree and forest: A comprehensive cross-datasets evaluation | Expert Systems with Applications, 2026 | Latest cross-dataset robustness reference | Cambridge COVID-19 Sounds, Coswara, COUGHVID, Virufy, NoCoCoDa | Deep Neural Decision Tree/Forest, RFECV, Bayesian hyperparameter optimization, SMOTE, threshold moving | AUC around 0.92 to 0.99 across individual settings; combined DNDF reports accuracy/AUC around 0.97 | Cough-only and method-heavy; high reported metrics may depend on dataset construction and tuning | Our external transfer is far weaker but more conservative; we emphasize calibration/confounding and do not claim comparable SOTA accuracy | references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md |
| P11 | Audio-Based Screening of Respiratory Diseases | MDPI, 2026 | Segmentation framework reference | Respiratory disease audio screening datasets | Event-focused respiratory audio preprocessing and segmentation pipeline | Used for methodology guidance rather than direct COVID metric comparison in current docs | Broader respiratory screening, not directly reproduced here | We include quality filtering and note event segmentation as a future extension beyond current global feature extraction | references/source_plans/literature_matrix_v2_18_papers.md |
| P12 | Sensors 2022 systematic review on audio-based COVID/respiratory screening | Sensors, 2022 | Methodology breadth review | Review across COVID and respiratory audio studies | Systematic review of preprocessing, models, metrics, and study designs | Review source; no single directly comparable metric | Review-level evidence cannot validate our model directly | We follow the review's multi-metric and preprocessing-documentation guidance | references/source_plans/literature_matrix_v2_18_papers.md |
| P13 | COVID-19 detection in cough, breath and speech using deep transfer learning and bottleneck features | Computers in Biology and Medicine, 2022 | Older multimodal baseline | Coswara, ComParE, Sarcos and related respiratory audio data | Deep transfer learning and bottleneck features across cough, breath, and speech | AUC about 0.98 for cough, 0.94 for breath, and 0.92 for speech | Older high metrics require careful comparison because validation/control assumptions may differ | Our results are lower, but we add external COUGHVID transfer, confounding checks, calibration, and operating points | references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md |
| P14 | Pay Attention to the Speech | Alexandria Engineering Journal, 2022 | Speech/fusion precedent | Coswara multimodal audio | Handcrafted audio features, class imbalance handling, attention/fusion over speech and respiratory sounds | Used as speech/fusion precedent in project docs; exact metric extraction deferred to full paper table | Fusion can degrade if not calibrated and leakage-safe | We compare fusion against best single modality and add calibration/quality/confounding evidence | references/source_plans/literature_matrix_v2_18_papers.md; references/verified_source_registry.md |
| P15 | Audio texture analysis of COVID-19 cough, breath, and speech sounds | Biomedical Signal Processing and Control / Frontiers, 2022 | Interpretable feature-engineering baseline | Cambridge COVID-19 Sounds subset: 1141 cough, 392 breath, 893 speech samples | Audio texture and handcrafted spectrogram features | 5-class accuracy around 71.7% cough, 72.2% breath, and speech binary accuracy around 79.7% | Dataset-specific feature-engineering results; exact binary comparisons need full table extraction | We include handcrafted MFCC/OpenSMILE baselines and learned embeddings, then test external robustness | references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md |
| P16 | Audio feature ranking for sound-based COVID-19 patient detection | arXiv, 2021 | Feature ranking reference | Cambridge and Coswara in project notes | Feature ranking and selection for classical sound-based COVID detection | Supports feature-selection improvements; exact table metrics not captured in current docs | Feature-ranking evidence does not by itself solve external transfer or confounding | We use feature strategies and shift-based feature selection, then report weak external generalization honestly | references/source_plans/literature_matrix_v2_18_papers.md; references/verified_source_registry.md |
| P17 | Investigating Feature Selection And Explainability For COVID-19 Diagnostics From Cough Sounds | Interspeech, 2021 | Feature selection and explainability support | DiCOVA challenge context in project notes | High-dimensional acoustic features, log-mel CNN, probability-score fusion, explainability/feature selection | Reported DiCOVA challenge improvements; exact metrics not captured in current docs | Explainability is inspection, not proof of causal COVID biomarkers | We use feature selection and evidence tables, but avoid causal biomarker claims | references/source_plans/literature_matrix_v2_18_papers.md; references/verified_source_registry.md |
| P18 | COVID-19 cough classification using machine learning and global smartphone recordings | Computers in Biology and Medicine, 2021 | Real-world recording variability reference | Coswara and Sarcos/South Africa smartphone cough recordings | Machine learning/deep residual architecture on global smartphone cough recordings | Highest AUC around 0.98 from residual architecture in paper highlights | Real-world smartphone recordings introduce device/noise/domain variability | We directly show that cross-dataset robustness is difficult and keep device/noise/domain-shift limitations explicit | references/source_plans/BTP_COVID_AUDIO_E2E_DETAILED_PLAN_FROM_DRIVE.md; references/source_plans/literature_matrix_v2_18_papers.md |

## Publication Readiness

This is BTP-ready if presented as a rigorous negative/robustness study. It is not ready as a clinical diagnostic or high-accuracy detector paper.

Defensible claim: COVID audio models can perform well internally, but measured metadata confounding and cross-dataset shift substantially weaken external validity; robust reporting requires external validation, confounding audits, operating points, and calibration analysis.

Non-defensible claim: this system is ready for clinical diagnosis, screening deployment, or individual health decision-making.

## Limitations

- External transfer remains weak, so deployment claims are not supported.
- IPW control addresses measured confounders only; unmeasured confounding may remain.
- COUGHVID and Coswara labels, collection protocols, and class prevalences differ.
- Related-paper quantitative comparison still needs a structured table based on the exact papers from the original document.
- The current system is a research prototype, not a medical device.

## Remaining Work

1. Add a structured related-paper comparison table from the original project document.
2. Convert this report into the final BTP manuscript sections.
3. Keep the central claim conservative: robustness analysis, not clinical deployment.
4. Archive final artifacts and Git tags after the final report and related-paper table are frozen.
