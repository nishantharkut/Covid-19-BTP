# Representation Results: OpenSMILE + BEATs

Date: 2026-06-12
Scope: Cough-only representation comparison after corrected Coswara and COUGHVID setup.

Included:
- OpenSMILE eGeMAPSv02 Coswara cough features
- OpenSMILE eGeMAPSv02 COUGHVID cough features
- BEATs official Coswara cough embeddings
- BEATs official COUGHVID cough embeddings
- External Coswara-to-COUGHVID metrics
- COUGHVID internal metrics
- Feature-shift reports
- Refreshed paper metric tables
- Experiment manifest

Key external results:
- MFCC best external: AUROC 0.535, AUPRC 0.042
- OpenSMILE best external: AUROC 0.552, AUPRC about 0.041
- BEATs best external: AUROC 0.553, AUPRC 0.039

Key COUGHVID internal results:
- MFCC best internal: AUROC 0.781, AUPRC 0.178
- OpenSMILE best internal: AUROC 0.763, AUPRC 0.128
- BEATs best internal: AUROC 0.756, AUPRC 0.148

Interpretation:
OpenSMILE and BEATs slightly improve external AUROC but do not solve AUPRC or internal performance. Current evidence points to dataset/domain/label shift rather than a simple MFCC limitation.
