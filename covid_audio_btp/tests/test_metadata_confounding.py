import pandas as pd


def _metadata_rows() -> pd.DataFrame:
    rows = []
    labels = ["negative", "positive"] * 18
    splits = ["train"] * 24 + ["validation"] * 6 + ["test"] * 6
    for i, (label, split) in enumerate(zip(labels, splits)):
        positive = label == "positive"
        rows.append(
            {
                "recording_id": f"rec_{i}",
                "participant_id": f"p_{i}",
                "dataset": "coswara",
                "modality": "cough",
                "label_raw": "positive_mild" if positive else "healthy",
                "label_binary": label,
                "label_group": label,
                "split": split,
                "age": 50 if positive else 25,
                "gender": "male" if i % 2 else "female",
                "country": "India" if i % 3 else "United States",
                "recording_date": f"2020-04-{(i % 28) + 1:02d}",
                "symptoms_json": '{"cough": true, "fever": true}' if positive else '{"cough": false, "fever": false}',
                "comorbidities_json": '{"asthma": true}' if positive else '{"asthma": false}',
                "duration_sec": 7.5 if positive else 4.0,
                "sample_rate_original": 48000 if i % 2 else 44100,
                "quality_flag": "ok",
            }
        )
    return pd.DataFrame(rows)


def test_feature_builder_excludes_direct_label_columns_and_tags_groups():
    from covid_audio_btp.metadata_confounding import build_audit_feature_frame, feature_group_for_column

    features, groups = build_audit_feature_frame(_metadata_rows(), feature_set="full_safe_metadata")

    assert "label_binary" not in features.columns
    assert "label_raw" not in features.columns
    assert "label_group" not in features.columns
    assert "symptoms_json_cough" in features.columns
    assert "comorbidities_json_asthma" in features.columns
    assert "recording_year" in features.columns
    assert "recording_month" in features.columns
    assert groups["symptoms_json_cough"] == "symptom_or_label_proxy"
    assert feature_group_for_column("duration_sec") == "recording_protocol"


def test_metadata_confounding_audit_outputs_metrics_importance_and_group_summary():
    from covid_audio_btp.metadata_confounding import run_metadata_confounding_audit

    result = run_metadata_confounding_audit(_metadata_rows(), random_state=0)

    assert set(result.metrics["audit_model"]) == {
        "symptoms_only",
        "demographic_protocol_only",
        "full_safe_metadata",
    }
    assert result.metrics["auroc"].notna().all()
    assert {"feature", "feature_group", "coefficient", "importance_abs", "audit_model"}.issubset(
        result.feature_importance.columns
    )
    assert not result.feature_importance["feature"].isin(["label_raw", "label_group", "label_binary"]).any()
    assert {"audit_model", "feature_group", "importance_abs_sum", "top_feature"}.issubset(
        result.group_summary.columns
    )
    assert "symptom_or_label_proxy" in set(result.group_summary["feature_group"])
