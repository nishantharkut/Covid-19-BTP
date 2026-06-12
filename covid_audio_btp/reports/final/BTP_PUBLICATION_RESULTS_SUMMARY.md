# COVID Audio BTP Results Summary

- Best external transfer: auroc=0.553; this remains weak and cautionary.
- IPW-controlled audio: auroc=0.780; audio signal persists but is reduced after measured confounder control.
- Strongest metadata confounding result: auroc=0.964; non-audio variables strongly predict labels.
- Worst external calibration shift: ece=0.286; external probabilities are not reliable calibrated risks.

Conclusion: the project is defensible as a robustness and external-validation BTP study, not as a clinically deployable diagnostic model.
