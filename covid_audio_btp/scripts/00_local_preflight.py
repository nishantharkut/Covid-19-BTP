#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import json
import py_compile
import sys
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".webm", ".m4a", ".aac"}

REQUIRED_IMPORTS = {
    "covid_audio_btp": "covid_audio_btp",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "joblib": "joblib",
    "tqdm": "tqdm",
}

OPTIONAL_IMPORTS = {
    "xgboost": "xgboost",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "pytest": "pytest",
    "jupyterlab": "jupyterlab",
    "ipykernel": "ipykernel",
    "streamlit": "streamlit",
}


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "covid_audio_btp").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find project root. Run this from the extracted covid_audio_btp folder "
        "or one of its subfolders."
    )


def compile_python_files(root: Path) -> list[str]:
    failures: list[str] = []
    for base in [root / "src", root / "scripts", root / "tests"]:
        if not base.exists():
            continue
        for file_path in sorted(base.rglob("*.py")):
            try:
                py_compile.compile(str(file_path), doraise=True)
            except Exception as exc:  # pragma: no cover - diagnostic path
                failures.append(f"{file_path.relative_to(root)}: {exc}")
    return failures


def compile_notebooks(root: Path) -> list[str]:
    failures: list[str] = []
    notebooks_dir = root / "notebooks"
    if not notebooks_dir.exists():
        return ["notebooks directory is missing"]
    for notebook_path in sorted(notebooks_dir.glob("*.ipynb")):
        try:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{notebook_path.relative_to(root)}: invalid JSON: {exc}")
            continue
        for idx, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            try:
                compile(source, f"{notebook_path}:cell{idx}", "exec")
            except SyntaxError as exc:
                failures.append(
                    f"{notebook_path.relative_to(root)} cell {idx}: {exc.msg} at line {exc.lineno}"
                )
    return failures


def check_imports(root: Path) -> tuple[list[str], list[str]]:
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for package_name, import_name in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            missing_required.append(f"{package_name} ({import_name}): {exc}")
    for package_name, import_name in OPTIONAL_IMPORTS.items():
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            missing_optional.append(f"{package_name} ({import_name}): {exc}")
    return missing_required, missing_optional


def check_coswara(coswara_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not coswara_dir.exists():
        errors.append(f"Coswara directory does not exist: {coswara_dir}")
        return errors, warnings
    if not coswara_dir.is_dir():
        errors.append(f"Coswara path is not a directory: {coswara_dir}")
        return errors, warnings

    audio_files = [p for p in coswara_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    csv_files = [p for p in coswara_dir.rglob("*.csv") if p.is_file()]
    print(f"Coswara audio files discovered: {len(audio_files)}")
    print(f"Coswara CSV files discovered: {len(csv_files)}")
    if len(audio_files) == 0:
        errors.append(
            "No audio files were found under Coswara. If this is the official Coswara repository, "
            "run `python extract_data.py` inside data/raw/coswara before running the notebook."
        )
    if len(csv_files) == 0:
        warnings.append("No CSV metadata files were found under Coswara; labels may become unknown.")
    return errors, warnings


def print_block(title: str, rows: list[str]) -> None:
    if not rows:
        return
    print(f"\n{title}")
    for row in rows:
        print(f"- {row}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast local preflight for the COVID audio BTP notebooks.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--coswara-dir", type=Path, default=None)
    parser.add_argument("--skip-imports", action="store_true")
    parser.add_argument("--skip-coswara", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve() if args.project_root else find_project_root(Path.cwd())
    coswara_dir = args.coswara_dir.resolve() if args.coswara_dir else root / "data" / "raw" / "coswara"

    print(f"Project root: {root}")
    print(f"Python: {sys.executable}")
    print(f"Coswara path: {coswara_dir}")

    errors: list[str] = []
    warnings: list[str] = []

    notebook_failures = compile_notebooks(root)
    python_failures = compile_python_files(root)
    errors.extend(notebook_failures)
    errors.extend(python_failures)
    if not notebook_failures:
        print("Notebook syntax: OK")
    if not python_failures:
        print("Python syntax: OK")

    if not args.skip_imports:
        missing_required, missing_optional = check_imports(root)
        errors.extend(missing_required)
        warnings.extend(missing_optional)
        if not missing_required:
            print("Required imports: OK")

    if not args.skip_coswara:
        coswara_errors, coswara_warnings = check_coswara(coswara_dir)
        errors.extend(coswara_errors)
        warnings.extend(coswara_warnings)

    print_block("WARNINGS", warnings)
    print_block("ERRORS", errors)

    if errors:
        print(
            "\nPreflight failed. Fix these before running the full notebook. "
            "This check is intentionally strict to avoid wasting hours on a broken setup."
        )
        return 1

    print("\nPreflight passed. It is safe to start the notebook pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
