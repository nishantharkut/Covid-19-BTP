import pandas as pd
import pytest

from covid_audio_btp.split import assert_no_participant_leakage, build_modality_availability


def test_no_participant_leakage_passes():
    df = pd.DataFrame(
        {
            "participant_id": ["p1", "p2", "p3"],
            "split": ["train", "validation", "test"],
        }
    )

    assert_no_participant_leakage(df)


def test_no_participant_leakage_fails():
    df = pd.DataFrame(
        {
            "participant_id": ["p1", "p1"],
            "split": ["train", "test"],
        }
    )

    with pytest.raises(ValueError, match="participant leakage"):
        assert_no_participant_leakage(df)


def test_modality_availability_marks_complete_cases():
    df = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p1", "p2"],
            "modality": ["cough", "breath", "speech", "cough"],
        }
    )

    availability = build_modality_availability(df)
    row_p1 = availability.set_index("participant_id").loc["p1"]
    row_p2 = availability.set_index("participant_id").loc["p2"]

    assert bool(row_p1["complete_case"]) is True
    assert bool(row_p2["complete_case"]) is False
    assert row_p1["available_modalities"] == "breath,cough,speech"



def test_participant_splits_include_demographic_stratification_columns():
    from covid_audio_btp.split import create_participant_splits

    rows = []
    for i in range(40):
        rows.append(
            {
                "participant_id": f"p{i}",
                "recording_id": f"r{i}",
                "dataset": "toy",
                "modality": "cough",
                "label_binary": "positive" if i % 2 else "negative",
                "age": 25 if i % 4 < 2 else 65,
                "gender": "male" if i % 4 in {0, 1} else "female",
            }
        )
    metadata = pd.DataFrame(rows)

    manifest, metadata_with_split = create_participant_splits(metadata)

    assert {"age_bucket", "gender", "split_stratify_group"}.issubset(manifest.columns)
    assert manifest["split_stratify_group"].str.contains("label").all()
    assert set(metadata_with_split["split"]).issubset({"train", "validation", "test"})
    assert_no_participant_leakage(metadata_with_split)
