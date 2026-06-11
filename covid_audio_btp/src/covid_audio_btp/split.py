from __future__ import annotations

import math

import pandas as pd
from sklearn.model_selection import train_test_split

from covid_audio_btp.config import SPLIT_CONFIG, SplitConfig
from covid_audio_btp.data_index import build_modality_availability
from covid_audio_btp.schemas import SPLIT_COLUMNS


def assert_no_participant_leakage(df: pd.DataFrame) -> None:
    if "participant_id" not in df.columns or "split" not in df.columns:
        raise ValueError("DataFrame must contain participant_id and split columns")
    usable = df[df["split"].isin(["train", "validation", "test"])]
    counts = usable.groupby("participant_id")["split"].nunique()
    leaked = counts[counts > 1]
    if not leaked.empty:
        raise ValueError(f"participant leakage detected: {list(leaked.index[:10])}")


def _can_stratify(labels: pd.Series) -> bool:
    counts = labels.value_counts()
    return len(counts) >= 2 and counts.min() >= 2


def _age_bucket(value: object) -> str:
    try:
        age = float(value)
    except Exception:
        return "unknown"
    if age < 30:
        return "<30"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    return "60+"


def _gender_group(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"m", "male"}:
        return "male"
    if text in {"f", "female"}:
        return "female"
    if text in {"other", "others", "nonbinary", "non-binary"}:
        return "other"
    return "unknown"


def _first_nonempty(values: pd.Series, default: str = "unknown") -> object:
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned.ne("")]
    return cleaned.iloc[0] if not cleaned.empty else default


def _participant_table(metadata: pd.DataFrame) -> pd.DataFrame:
    supervised = metadata[metadata["label_binary"].isin(["positive", "negative"])].copy()
    if supervised.empty:
        raise ValueError("No supervised positive/negative rows found")

    def join_modalities(values: pd.Series) -> str:
        return ",".join(sorted(set(v for v in values.dropna().astype(str) if v and v != "unknown")))

    aggregations = {
        "label_binary": ("label_binary", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
        "n_recordings": ("recording_id", "nunique"),
        "modalities_available": ("modality", join_modalities),
    }
    if "age" in supervised.columns:
        aggregations["age"] = ("age", _first_nonempty)
    if "gender" in supervised.columns:
        aggregations["gender"] = ("gender", _first_nonempty)

    table = supervised.groupby(["participant_id", "dataset"], dropna=False).agg(**aggregations).reset_index()
    if "age" not in table.columns:
        table["age"] = "unknown"
    if "gender" not in table.columns:
        table["gender"] = "unknown"
    table["age_bucket"] = table["age"].map(_age_bucket)
    table["gender"] = table["gender"].map(_gender_group)
    table["label_age_gender"] = (
        table["label_binary"].astype(str) + "|" + table["age_bucket"].astype(str) + "|" + table["gender"].astype(str)
    )
    table["label_age"] = table["label_binary"].astype(str) + "|" + table["age_bucket"].astype(str)
    return table


def _best_stratify_series(participants: pd.DataFrame) -> tuple[pd.Series | None, str]:
    for column in ("label_age_gender", "label_age", "label_binary"):
        if column not in participants.columns:
            continue
        labels = participants[column].astype(str)
        if _can_stratify(labels):
            return labels, column
    return None, "none"


def create_participant_splits(
    metadata: pd.DataFrame,
    config: SplitConfig = SPLIT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not math.isclose(
        config.train_size + config.validation_size + config.test_size,
        1.0,
        rel_tol=1e-6,
    ):
        raise ValueError("Split sizes must sum to 1.0")

    participants = _participant_table(metadata)
    stratify, stratify_group = _best_stratify_series(participants)

    train_df, temp_df = train_test_split(
        participants,
        train_size=config.train_size,
        random_state=config.seed,
        stratify=stratify,
    )

    temp_ratio = config.validation_size + config.test_size
    validation_relative = config.validation_size / temp_ratio
    temp_stratify, temp_stratify_group = _best_stratify_series(temp_df)
    validation_df, test_df = train_test_split(
        temp_df,
        train_size=validation_relative,
        random_state=config.seed,
        stratify=temp_stratify,
    )

    split_manifest = pd.concat(
        [
            train_df.assign(split="train"),
            validation_df.assign(split="validation"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    split_manifest["split_seed"] = config.seed
    split_manifest["split_stratify_group"] = stratify_group + ";temp=" + temp_stratify_group
    split_manifest = split_manifest[SPLIT_COLUMNS]

    metadata_with_split = metadata.copy()
    mapping = split_manifest.set_index("participant_id")["split"].to_dict()
    metadata_with_split["split"] = metadata_with_split["participant_id"].map(mapping).fillna("unused")
    assert_no_participant_leakage(metadata_with_split)
    return split_manifest, metadata_with_split


def split_audit(metadata: pd.DataFrame) -> pd.DataFrame:
    return (
        metadata.groupby(["split", "label_binary", "modality"], dropna=False)
        .agg(n_recordings=("recording_id", "nunique"), n_participants=("participant_id", "nunique"))
        .reset_index()
        .sort_values(["split", "label_binary", "modality"])
    )

