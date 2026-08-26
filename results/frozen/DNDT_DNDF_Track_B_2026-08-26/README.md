# DNDT/DNDF Track B Evidence

This folder contains the compact, directly reviewable outputs from the completed
DNDF integration experiment. The run used frozen COVID-RARS features,
participant-separated evaluation, three final seeds, validation-only model and
threshold selection, multimodal fusion, temporal protocols, COUGHVID external
transfer, and a participant-level shuffle-label control.

The complete non-checkpoint run is stored in:

`artifacts/bundles/dndt_dndf_track_b_evidence_20260826.tar.gz`

SHA-256:

`68477930b8967b7522f9fd40e273ebbb092dacc6bcdc63eeca06eba0110d3565`

The archive contains all CSV predictions and metrics, selected configurations,
split records, manifests, receipts, feature lineage, and checkpoint manifests.
PyTorch `.pt` checkpoint and recovery binaries are excluded because they total
approximately 37.7 GiB and are reproducible from the committed code and recorded
configuration. Their manifests and hashes remain in the archive for provenance.

The CSV and JSON files beside this README are the compact headline evidence used
for report, manuscript, and presentation updates. They do not replace the full
archive when prediction-level recomputation is required.
