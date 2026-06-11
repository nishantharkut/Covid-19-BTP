from __future__ import annotations

METADATA_COLUMNS = [
    "participant_id",
    "recording_id",
    "dataset",
    "modality",
    "submodality",
    "audio_path",
    "label_raw",
    "label_binary",
    "label_group",
    "recording_date",
    "age",
    "gender",
    "country",
    "symptoms_json",
    "comorbidities_json",
    "duration_sec",
    "sample_rate_original",
    "quality_flag",
    "manual_quality_score",
    "manual_quality_label",
    "split",
]

INDEX_COLUMNS = [
    "participant_id",
    "recording_id",
    "dataset",
    "modality",
    "submodality",
    "audio_path",
    "label_raw",
    "label_binary",
    "manual_quality_score",
    "manual_quality_label",
]

SPLIT_COLUMNS = [
    "participant_id",
    "dataset",
    "split",
    "label_binary",
    "n_recordings",
    "modalities_available",
    "age_bucket",
    "gender",
    "split_stratify_group",
    "split_seed",
]

QUALITY_COLUMNS = [
    "recording_id",
    "audio_path",
    "duration_sec",
    "sample_rate_original",
    "rms_mean",
    "rms_std",
    "zero_crossing_rate_mean",
    "silence_ratio",
    "clipping_ratio",
    "spectral_centroid_mean",
    "spectral_flatness_mean",
    "snr_proxy",
    "event_start_sec",
    "event_end_sec",
    "event_duration_sec",
    "active_audio_ratio",
    "quality_flag",
    "quality_reasons",
]

MODALITY_COLUMNS = [
    "participant_id",
    "has_cough",
    "has_breath",
    "has_speech",
    "n_cough",
    "n_breath",
    "n_speech",
    "complete_case",
    "available_modalities",
]

VALID_MODALITIES = {"cough", "breath", "speech", "unknown"}
VALID_BINARY_LABELS = {"positive", "negative", "unknown"}
VALID_SPLITS = {"train", "validation", "test", "external", "unused"}

