from pathlib import Path

import numpy as np
import pandas as pd


def test_coughvid_adapter_normalizes_common_metadata(tmp_path):
    from covid_audio_btp.external_datasets import build_coughvid_index

    audio_dir = tmp_path / "public_dataset"
    audio_dir.mkdir()
    (audio_dir / "abc.webm").write_bytes(b"fake")
    (audio_dir / "def.webm").write_bytes(b"fake")
    pd.DataFrame(
        {
            "uuid": ["abc", "def"],
            "status": ["COVID-19", "healthy"],
            "cough_detected": [0.92, 0.81],
            "SNR": [12.5, 8.0],
            "age": [31, 52],
            "gender": ["male", "female"],
            "respiratory_condition": [False, True],
        }
    ).to_csv(tmp_path / "metadata_compiled.csv", index=False)

    index = build_coughvid_index(tmp_path)

    assert list(index["participant_id"]) == ["abc", "def"]
    assert list(index["label_binary"]) == ["positive", "negative"]
    assert set(index["modality"]) == {"cough"}
    assert index.loc[0, "dataset"] == "coughvid"
    assert Path(index.loc[0, "audio_path"]).name == "abc.webm"
    assert index.loc[0, "manual_quality_score"] == 0.92
    assert index.loc[0, "snr_proxy"] == 12.5


def test_coughvid_adapter_supports_real_sidecar_json_layout(tmp_path):
    import json

    from covid_audio_btp.external_datasets import build_coughvid_index

    public = tmp_path / "public_dataset"
    public.mkdir()
    uuid = "00039425-7f3a-42aa-ac13-834aaa2b6b92"
    (public / f"{uuid}.webm").write_bytes(b"fake")
    (public / f"{uuid}.json").write_text(
        json.dumps(
            {
                "datetime": "2020-04-13T21:30:59.801831+00:00",
                "cough_detected": "0.9609",
                "latitude": "31.3",
                "longitude": "34.8",
                "age": "15",
                "gender": "male",
                "respiratory_condition": "False",
                "fever_muscle_pain": "False",
                "status": "healthy",
            }
        ),
        encoding="utf-8",
    )

    index = build_coughvid_index(tmp_path)

    assert len(index) == 1
    assert index.loc[0, "participant_id"] == uuid
    assert index.loc[0, "label_binary"] == "negative"
    assert index.loc[0, "recording_date"].startswith("2020-04-13")
    assert index.loc[0, "manual_quality_label"] == "ok"


def test_quality_weighted_fusion_downweights_bad_modalities():
    from covid_audio_btp.fusion import quality_weighted_fusion

    predictions = pd.DataFrame(
        {
            "recording_id": ["r1", "r2"],
            "participant_id": ["p1", "p1"],
            "modality": ["cough", "breath"],
            "label_binary": ["positive", "positive"],
            "split": ["test", "test"],
            "probability": [0.80, 0.20],
        }
    )
    quality = pd.DataFrame(
        {
            "recording_id": ["r1", "r2"],
            "quality_flag": ["ok", "corrupt"],
        }
    )
    validation_metrics = pd.DataFrame(
        {
            "modality": ["cough", "breath"],
            "auprc": [0.90, 0.60],
        }
    )

    fused = quality_weighted_fusion(predictions, quality, validation_metrics)

    assert fused.loc[0, "fusion_method"] == "quality_weighted_auprc"
    assert fused.loc[0, "available_modalities"] == "breath,cough"
    assert fused.loc[0, "probability"] > 0.70


def test_bootstrap_metric_ci_returns_reproducible_interval():
    from covid_audio_btp.statistics import bootstrap_metric_ci

    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.7])

    ci = bootstrap_metric_ci(y_true, y_prob, metric="auroc", n_bootstraps=100, random_state=7)

    assert ci["metric"] == "auroc"
    assert ci["n_bootstraps"] == 100
    assert 0.0 <= ci["ci_low"] <= ci["mean"] <= ci["ci_high"] <= 1.0


def test_abstention_marks_uncertain_and_low_quality_samples():
    from covid_audio_btp.abstention import apply_abstention, coverage_curve

    predictions = pd.DataFrame(
        {
            "recording_id": ["r1", "r2", "r3", "r4"],
            "label_binary": ["negative", "positive", "positive", "negative"],
            "probability": [0.10, 0.52, 0.91, 0.20],
            "quality_flag": ["ok", "ok", "corrupt", "ok"],
        }
    )

    out = apply_abstention(predictions, uncertainty_low=0.4, uncertainty_high=0.6)
    curve = coverage_curve(out, probability_column="probability")

    assert list(out["abstention_reason"]) == ["accepted", "uncertain_probability", "low_quality", "accepted"]
    assert out["accepted"].sum() == 2
    assert not curve.empty
    assert {"coverage", "auroc", "auprc"}.issubset(curve.columns)


def test_metadata_baseline_trains_and_predicts_on_metadata_only():
    from covid_audio_btp.metadata_baseline import train_metadata_baseline

    metadata = pd.DataFrame(
        {
            "recording_id": [f"r{i}" for i in range(12)],
            "participant_id": [f"p{i}" for i in range(12)],
            "dataset": ["toy"] * 12,
            "split": ["train"] * 6 + ["validation"] * 3 + ["test"] * 3,
            "label_binary": ["negative", "positive"] * 6,
            "age": [20, 60, 22, 62, 24, 64, 26, 66, 28, 68, 30, 70],
            "gender": ["female", "male"] * 6,
            "symptoms_json": ["{}", "{\"cough\": true}"] * 6,
            "quality_flag": ["ok"] * 12,
        }
    )

    result = train_metadata_baseline(metadata, feature_columns=["age", "gender", "symptoms_json", "quality_flag"])

    assert result.model_name == "metadata_logistic_regression"
    assert set(result.test_predictions["split"]) == {"test"}
    assert result.test_predictions["probability"].between(0, 1).all()
    assert "auprc" in result.metrics


def test_harmonize_feature_columns_aligns_train_and_external_tables():
    from covid_audio_btp.cross_dataset import harmonize_feature_columns

    train = pd.DataFrame({"recording_id": ["a"], "mfcc_1": [1.0], "mfcc_2": [2.0]})
    external = pd.DataFrame({"recording_id": ["b"], "mfcc_2": [3.0], "mfcc_3": [4.0]})

    train_x, external_x, cols = harmonize_feature_columns(train, external, id_columns=["recording_id"])

    assert cols == ["mfcc_1", "mfcc_2", "mfcc_3"]
    assert train_x.loc[0, "mfcc_3"] == 0.0
    assert external_x.loc[0, "mfcc_1"] == 0.0


def test_coughvid_adapter_supports_official_zip_sidecar_layout(tmp_path):
    import json
    import zipfile

    from covid_audio_btp.external_datasets import build_coughvid_index

    uuid = "00039425-7f3a-42aa-ac13-834aaa2b6b92"
    zip_path = tmp_path / "public_dataset.zip"
    with zipfile.ZipFile(zip_path, mode="w") as zf:
        zf.writestr(
            f"public_dataset/{uuid}.json",
            json.dumps(
                {
                    "datetime": "2020-04-13T21:30:59.801831+00:00",
                    "cough_detected": "0.9609",
                    "latitude": "31.3",
                    "longitude": "34.8",
                    "age": "15",
                    "gender": "male",
                    "respiratory_condition": "False",
                    "fever_muscle_pain": "False",
                    "status": "healthy",
                }
            ),
        )
        zf.writestr(f"public_dataset/{uuid}.webm", b"fake audio bytes")

    index = build_coughvid_index(zip_path, min_cough_detected=0.8)

    assert len(index) == 1
    assert index.loc[0, "participant_id"] == uuid
    assert index.loc[0, "audio_path"] == f"{zip_path.as_posix()}::public_dataset/{uuid}.webm"
    assert index.loc[0, "label_binary"] == "negative"
    assert index.loc[0, "manual_quality_label"] == "ok"


def test_local_audio_path_materializes_zip_member_temporarily(tmp_path):
    import zipfile

    from covid_audio_btp.audio_io import local_audio_path, split_archive_member_path

    zip_path = tmp_path / "public_dataset.zip"
    with zipfile.ZipFile(zip_path, mode="w") as zf:
        zf.writestr("public_dataset/sample.webm", b"fake audio bytes")

    archive_member = split_archive_member_path(f"{zip_path.as_posix()}::public_dataset/sample.webm")
    assert archive_member == (zip_path, "public_dataset/sample.webm")

    with local_audio_path(f"{zip_path.as_posix()}::public_dataset/sample.webm") as local_path:
        assert local_path.exists()
        assert local_path.suffix == ".webm"
        assert local_path.read_bytes() == b"fake audio bytes"
    assert not local_path.exists()



def test_coughvid_adapter_supports_v3_metadata_inside_zip(tmp_path):
    import zipfile

    from covid_audio_btp.external_datasets import build_coughvid_index

    uuid = "ghi"
    zip_path = tmp_path / "public_dataset_v3.zip"
    metadata = pd.DataFrame(
        {
            "uuid": [uuid],
            "status_SSL": ["COVID-19"],
            "datetime": ["2021-08-26T12:00:00+00:00"],
            "cough_detected": [0.95],
            "SNR": [11.2],
            "latitude": [46.5],
            "longitude": [6.6],
            "age": [44],
            "gender": ["female"],
            "respiratory_condition": [False],
            "fever_muscle_pain": [True],
        }
    )
    with zipfile.ZipFile(zip_path, mode="w") as zf:
        zf.writestr("metadata_compiled.csv", metadata.to_csv(index=False))
        zf.writestr(f"public_dataset/{uuid}.ogg", b"fake audio bytes")

    index = build_coughvid_index(zip_path, min_cough_detected=0.8)

    assert len(index) == 1
    assert index.loc[0, "participant_id"] == uuid
    assert index.loc[0, "label_raw"] == "COVID-19"
    assert index.loc[0, "label_binary"] == "positive"
    assert index.loc[0, "audio_path"] == f"{zip_path.as_posix()}::public_dataset/{uuid}.ogg"
    assert index.loc[0, "manual_quality_label"] == "ok"

def test_coughvid_adapter_supports_v3_status_ssl_metadata(tmp_path):
    from covid_audio_btp.external_datasets import build_coughvid_index

    audio_dir = tmp_path / "public_dataset"
    audio_dir.mkdir()
    (audio_dir / "ghi.ogg").write_bytes(b"fake")
    pd.DataFrame(
        {
            "uuid": ["ghi"],
            "status_SSL": ["COVID-19"],
            "datetime": ["2021-08-26T12:00:00+00:00"],
            "cough_detected": [0.95],
            "SNR": [11.2],
            "latitude": [46.5],
            "longitude": [6.6],
            "age": [44],
            "gender": ["female"],
            "respiratory_condition": [False],
            "fever_muscle_pain": [True],
        }
    ).to_csv(tmp_path / "metadata_compiled.csv", index=False)

    index = build_coughvid_index(tmp_path)

    assert len(index) == 1
    assert index.loc[0, "label_raw"] == "COVID-19"
    assert index.loc[0, "label_binary"] == "positive"
    assert index.loc[0, "recording_date"].startswith("2021-08-26")
    assert index.loc[0, "latitude"] == 46.5
    assert bool(index.loc[0, "fever_muscle_pain"]) is True


def test_prepare_external_feature_metadata_marks_external_split_and_filters_quality():
    from covid_audio_btp.external_features import prepare_external_feature_metadata

    index = pd.DataFrame(
        {
            "recording_id": ["r1", "r2", "r3"],
            "participant_id": ["p1", "p2", "p3"],
            "dataset": ["coughvid"] * 3,
            "modality": ["cough"] * 3,
            "audio_path": ["a.webm", "b.webm", "c.webm"],
            "label_binary": ["positive", "negative", "unknown"],
            "manual_quality_label": ["ok", "bad", "ok"],
        }
    )

    prepared = prepare_external_feature_metadata(index, quality_ok_only=True)

    assert list(prepared["recording_id"]) == ["r1"]
    assert prepared.loc[0, "split"] == "external"
    assert prepared.loc[0, "quality_flag"] == "ok"


def test_build_paper_metric_table_formats_point_estimates_and_ci():
    from covid_audio_btp.reporting import build_paper_metric_table

    metrics = pd.DataFrame(
        {
            "model_name": ["logistic_regression"],
            "modality": ["cough"],
            "auroc": [0.8123],
            "auprc": [0.7012],
            "brier": [0.1833],
            "ece": [0.0444],
        }
    )
    ci = pd.DataFrame(
        {
            "model_name": ["logistic_regression", "logistic_regression"],
            "modality": ["cough", "cough"],
            "metric": ["auroc", "auprc"],
            "point": [0.8123, 0.7012],
            "ci_low": [0.70, 0.61],
            "ci_high": [0.90, 0.80],
        }
    )

    table = build_paper_metric_table(metrics, ci_table=ci, group_columns=["model_name", "modality"])

    assert table.loc[0, "model_name"] == "logistic_regression"
    assert table.loc[0, "auroc"] == "0.812 [0.700, 0.900]"
    assert table.loc[0, "auprc"] == "0.701 [0.610, 0.800]"
    assert table.loc[0, "brier"] == "0.183"

