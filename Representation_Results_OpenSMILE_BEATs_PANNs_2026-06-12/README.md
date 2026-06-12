  # Representation Results - OpenSMILE, BEATs, PANNs

  Date: 2026-06-12
  Dataset scope: Coswara cough to COUGHVID cough plus COUGHVID internal baselines.

  Included representations:
  - OpenSMILE eGeMAPSv02
  - BEATs official embeddings
  - PANNs CNN14 embeddings

  Important interpretation:
  - External Coswara-to-COUGHVID transfer remains weak across all representations.
  - BEATs and OpenSMILE slightly improve external AUROC over MFCC, but AUPRC remains near COUGHVID prevalence.
  - PANNs is weaker in this setup.
  - COUGHVID internal baselines are stronger than external transfer, supporting dataset/domain shift as the main finding.
