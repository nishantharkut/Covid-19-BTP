from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from covid_audio_btp.data_index import AUDIO_EXTENSIONS, stable_recording_id
from covid_audio_btp.labels import normalize_label


def normalize_coughvid_label(value: object) -> str:
    """Map common COUGHVID status values into the project label vocabulary."""
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if not text or text in {"nan", "none", "unknown"}:
        return "unknown"
    if "covid" in text or "positive" in text:
        return "positive"
    if text in {"healthy", "negative", "control", "no covid"}:
        return "negative"
    if text in {"symptomatic", "other", "respiratory condition"}:
        return "unknown"
    return normalize_label(text)


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {str(col).lower().strip(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    for col in columns:
        lowered = str(col).lower()
        if any(candidate.lower() in lowered for candidate in candidates):
            return col
    return None


def find_coughvid_metadata(raw_dir: Path) -> Path:
    raw_dir = Path(raw_dir)
    preferred = ["metadata_compiled.csv", "metadata.csv"]
    for name in preferred:
        path = raw_dir / name
        if path.exists():
            return path
    candidates = sorted(p for p in raw_dir.rglob("*.csv") if "metadata" in p.name.lower())
    if not candidates:
        raise FileNotFoundError(f"No COUGHVID metadata CSV found under {raw_dir}")
    return candidates[0]


def find_coughvid_audio_path(raw_dir: Path, uuid: str) -> Path:
    raw_dir = Path(raw_dir)
    for suffix in AUDIO_EXTENSIONS:
        for base in (raw_dir / "public_dataset", raw_dir):
            path = base / f"{uuid}{suffix}"
            if path.exists():
                return path
    matches = sorted(p for p in raw_dir.rglob(f"{uuid}.*") if p.suffix.lower() in AUDIO_EXTENSIONS)
    if matches:
        return matches[0]
    return raw_dir / "public_dataset" / f"{uuid}.webm"


def _quality_label_from_cough_detected(value: object) -> str:
    try:
        score = float(value)
    except Exception:
        return "unknown"
    if score >= 0.80:
        return "ok"
    if score >= 0.50:
        return "uncertain"
    return "bad"


def _record_from_sidecar(uuid: str, meta: dict[str, object], audio_path: str) -> dict[str, object]:
    cough_score = meta.get("cough_detected", "")
    label_raw = meta.get("status", "unknown")
    symptoms = {
        "fever_muscle_pain": meta.get("fever_muscle_pain", ""),
        "respiratory_condition": meta.get("respiratory_condition", ""),
    }
    return {
        "participant_id": uuid,
        "recording_id": f"coughvid_{uuid}",
        "dataset": "coughvid",
        "modality": "cough",
        "submodality": "cough",
        "audio_path": audio_path,
        "label_raw": label_raw,
        "label_binary": normalize_coughvid_label(label_raw),
        "recording_date": meta.get("datetime", ""),
        "age": meta.get("age", ""),
        "gender": meta.get("gender", ""),
        "country": "",
        "latitude": meta.get("latitude", ""),
        "longitude": meta.get("longitude", ""),
        "respiratory_condition": meta.get("respiratory_condition", ""),
        "fever_muscle_pain": meta.get("fever_muscle_pain", ""),
        "symptoms_json": json.dumps(symptoms),
        "comorbidities_json": json.dumps({"respiratory_condition": meta.get("respiratory_condition", "")}),
        "manual_quality_score": cough_score,
        "manual_quality_label": _quality_label_from_cough_detected(cough_score),
        "cough_detected": cough_score,
        "snr_proxy": meta.get("SNR", meta.get("snr", "")),
    }


def _sidecar_records_from_directory(raw_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for json_path in sorted(raw_dir.rglob("*.json")):
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        uuid = json_path.stem
        audio_path = find_coughvid_audio_path(raw_dir, uuid)
        rows.append(_record_from_sidecar(uuid, meta, audio_path.as_posix()))
    return rows


def _sidecar_records_from_zip(zip_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        json_names = sorted(name for name in names if name.startswith("public_dataset/") and name.endswith(".json"))
        for json_name in json_names:
            try:
                meta = json.loads(zf.read(json_name).decode("utf-8"))
            except Exception:
                continue
            uuid = Path(json_name).stem
            audio_member = ""
            for suffix in AUDIO_EXTENSIONS:
                candidate = f"public_dataset/{uuid}{suffix}"
                if candidate in names:
                    audio_member = candidate
                    break
            audio_path = f"{zip_path.as_posix()}::{audio_member}" if audio_member else f"{zip_path.as_posix()}::public_dataset/{uuid}.webm"
            rows.append(_record_from_sidecar(uuid, meta, audio_path))
    return rows


def _find_metadata_member_in_zip(names: set[str]) -> str | None:
    preferred = {"metadata_compiled.csv", "metadata.csv"}
    for name in sorted(names):
        if Path(name).name in preferred:
            return name
    for name in sorted(names):
        lowered = Path(name).name.lower()
        if lowered.endswith(".csv") and "metadata" in lowered:
            return name
    return None


def _find_audio_member_in_zip(names: set[str], uuid: str) -> str | None:
    for suffix in AUDIO_EXTENSIONS:
        for candidate in (f"public_dataset/{uuid}{suffix}", f"{uuid}{suffix}"):
            if candidate in names:
                return candidate
    for name in sorted(names):
        path = Path(name)
        if path.stem == uuid and path.suffix.lower() in AUDIO_EXTENSIONS:
            return name
    return None


def _records_from_metadata_zip(zip_path: Path, require_audio: bool, min_cough_detected: float | None) -> list[dict[str, object]] | None:
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        metadata_member = _find_metadata_member_in_zip(names)
        if metadata_member is None:
            return None
        with zf.open(metadata_member) as fh:
            meta = pd.read_csv(fh)

    uuid_col = _pick_column(list(meta.columns), ["uuid", "id", "recording_id"])
    status_col = _pick_column(list(meta.columns), ["status_SSL", "status ssl", "status", "covid_status", "label"])
    cough_col = _pick_column(list(meta.columns), ["cough_detected", "cough probability", "cough_prob"])
    snr_col = _pick_column(list(meta.columns), ["snr", "SNR"])
    age_col = _pick_column(list(meta.columns), ["age"])
    gender_col = _pick_column(list(meta.columns), ["gender", "sex"])
    country_col = _pick_column(list(meta.columns), ["country", "location"])
    date_col = _pick_column(list(meta.columns), ["datetime", "recording_date", "date", "timestamp"])
    latitude_col = _pick_column(list(meta.columns), ["latitude", "lat"])
    longitude_col = _pick_column(list(meta.columns), ["longitude", "lon", "lng"])
    respiratory_col = _pick_column(list(meta.columns), ["respiratory_condition", "respiratory condition"])
    fever_col = _pick_column(list(meta.columns), ["fever_muscle_pain", "fever muscle pain"])
    if uuid_col is None:
        raise ValueError("COUGHVID metadata must contain uuid/id/recording_id")

    rows: list[dict[str, object]] = []
    for _, row in meta.iterrows():
        uuid = str(row[uuid_col]).strip()
        if not uuid or uuid.lower() == "nan":
            continue
        cough_score = row[cough_col] if cough_col else ""
        if min_cough_detected is not None:
            try:
                if float(cough_score) < min_cough_detected:
                    continue
            except Exception:
                continue
        audio_member = _find_audio_member_in_zip(names, uuid)
        if require_audio and audio_member is None:
            continue
        audio_path = f"{zip_path.as_posix()}::{audio_member}" if audio_member else f"{zip_path.as_posix()}::public_dataset/{uuid}.webm"
        label_raw = row[status_col] if status_col else "unknown"
        symptoms = {
            "respiratory_condition": row[respiratory_col] if respiratory_col else "",
            "fever_muscle_pain": row[fever_col] if fever_col else "",
        }
        rows.append(
            {
                "participant_id": uuid,
                "recording_id": f"coughvid_{uuid}",
                "dataset": "coughvid",
                "modality": "cough",
                "submodality": "cough",
                "audio_path": audio_path,
                "label_raw": label_raw,
                "label_binary": normalize_coughvid_label(label_raw),
                "recording_date": row[date_col] if date_col else "",
                "age": row[age_col] if age_col else "",
                "gender": row[gender_col] if gender_col else "",
                "country": row[country_col] if country_col else "",
                "latitude": row[latitude_col] if latitude_col else "",
                "longitude": row[longitude_col] if longitude_col else "",
                "respiratory_condition": row[respiratory_col] if respiratory_col else "",
                "fever_muscle_pain": row[fever_col] if fever_col else "",
                "symptoms_json": json.dumps(symptoms),
                "comorbidities_json": json.dumps(symptoms),
                "manual_quality_score": cough_score,
                "manual_quality_label": _quality_label_from_cough_detected(cough_score),
                "cough_detected": cough_score,
                "snr_proxy": row[snr_col] if snr_col else "",
            }
        )
    return rows


def _records_from_csv(raw_dir: Path, metadata_path: Path, require_audio: bool, min_cough_detected: float | None) -> list[dict[str, object]]:
    meta = pd.read_csv(metadata_path)
    uuid_col = _pick_column(list(meta.columns), ["uuid", "id", "recording_id"])
    status_col = _pick_column(list(meta.columns), ["status_SSL", "status ssl", "status", "covid_status", "label"])
    cough_col = _pick_column(list(meta.columns), ["cough_detected", "cough probability", "cough_prob"])
    snr_col = _pick_column(list(meta.columns), ["snr", "SNR"])
    age_col = _pick_column(list(meta.columns), ["age"])
    gender_col = _pick_column(list(meta.columns), ["gender", "sex"])
    country_col = _pick_column(list(meta.columns), ["country", "location"])
    date_col = _pick_column(list(meta.columns), ["datetime", "recording_date", "date", "timestamp"])
    latitude_col = _pick_column(list(meta.columns), ["latitude", "lat"])
    longitude_col = _pick_column(list(meta.columns), ["longitude", "lon", "lng"])
    respiratory_col = _pick_column(list(meta.columns), ["respiratory_condition", "respiratory condition"])
    fever_col = _pick_column(list(meta.columns), ["fever_muscle_pain", "fever muscle pain"])
    if uuid_col is None:
        raise ValueError("COUGHVID metadata must contain uuid/id/recording_id")
    rows: list[dict[str, object]] = []
    for _, row in meta.iterrows():
        uuid = str(row[uuid_col]).strip()
        if not uuid or uuid.lower() == "nan":
            continue
        cough_score = row[cough_col] if cough_col else ""
        if min_cough_detected is not None:
            try:
                if float(cough_score) < min_cough_detected:
                    continue
            except Exception:
                continue
        audio_path = find_coughvid_audio_path(raw_dir, uuid)
        if require_audio and not audio_path.exists():
            continue
        label_raw = row[status_col] if status_col else "unknown"
        symptoms = {
            "respiratory_condition": row[respiratory_col] if respiratory_col else "",
            "fever_muscle_pain": row[fever_col] if fever_col else "",
        }
        rows.append(
            {
                "participant_id": uuid,
                "recording_id": stable_recording_id(audio_path, raw_dir) if audio_path.exists() else f"coughvid_{uuid}",
                "dataset": "coughvid",
                "modality": "cough",
                "submodality": "cough",
                "audio_path": audio_path.as_posix(),
                "label_raw": label_raw,
                "label_binary": normalize_coughvid_label(label_raw),
                "recording_date": row[date_col] if date_col else "",
                "age": row[age_col] if age_col else "",
                "gender": row[gender_col] if gender_col else "",
                "country": row[country_col] if country_col else "",
                "latitude": row[latitude_col] if latitude_col else "",
                "longitude": row[longitude_col] if longitude_col else "",
                "respiratory_condition": row[respiratory_col] if respiratory_col else "",
                "fever_muscle_pain": row[fever_col] if fever_col else "",
                "symptoms_json": json.dumps(symptoms),
                "comorbidities_json": json.dumps(symptoms),
                "manual_quality_score": cough_score,
                "manual_quality_label": _quality_label_from_cough_detected(cough_score),
                "cough_detected": cough_score,
                "snr_proxy": row[snr_col] if snr_col else "",
            }
        )
    return rows


def build_coughvid_index(
    raw_dir: str | Path,
    metadata_path: str | Path | None = None,
    require_audio: bool = False,
    min_cough_detected: float | None = None,
) -> pd.DataFrame:
    """Build a project-compatible cough-only index for COUGHVID.

    Supports both common layouts:
    - extracted `public_dataset/<uuid>.json` + audio sidecars;
    - optional metadata CSV files such as `metadata_compiled.csv`;
    - direct inspection of the official `public_dataset.zip` archive.
    """
    raw_dir = Path(raw_dir)
    if raw_dir.is_file() and raw_dir.suffix.lower() == ".zip" and metadata_path is None:
        rows = _records_from_metadata_zip(raw_dir, require_audio, min_cough_detected)
        if rows is None:
            rows = _sidecar_records_from_zip(raw_dir)
    else:
        if metadata_path is not None:
            rows = _records_from_csv(raw_dir, Path(metadata_path), require_audio, min_cough_detected)
        else:
            try:
                rows = _records_from_csv(raw_dir, find_coughvid_metadata(raw_dir), require_audio, min_cough_detected)
            except FileNotFoundError:
                rows = _sidecar_records_from_directory(raw_dir)
    if min_cough_detected is not None and rows:
        filtered = []
        for row in rows:
            try:
                if float(row.get("cough_detected", "")) >= min_cough_detected:
                    filtered.append(row)
            except Exception:
                continue
        rows = filtered
    if require_audio and rows and not (raw_dir.is_file() and raw_dir.suffix.lower() == ".zip"):
        rows = [row for row in rows if Path(str(row["audio_path"])).exists()]
    return pd.DataFrame(rows)
