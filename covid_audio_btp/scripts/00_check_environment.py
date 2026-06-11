#!/usr/bin/env python
from __future__ import annotations

import importlib
import sys


REQUIRED = [
    "covid_audio_btp",
    "numpy",
    "pandas",
    "scipy",
    "librosa",
    "soundfile",
    "sklearn",
    "matplotlib",
    "seaborn",
    "joblib",
    "tqdm",
]

OPTIONAL = [
    "xgboost",
    "torch",
    "torchaudio",
    "streamlit",
    "pytest",
    "jupyterlab",
    "ipykernel",
]


def main() -> None:
    print(f"Python: {sys.version}")
    missing = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"OK required {name}: {version}")
        except Exception as exc:
            missing.append((name, str(exc)))
    for name in OPTIONAL:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"OK optional {name}: {version}")
        except Exception as exc:
            print(f"WARN optional {name} unavailable: {exc}")
    if missing:
        print("Missing required packages:")
        for name, error in missing:
            print(f"- {name}: {error}")
        raise SystemExit(1)
    print("Environment check passed")


if __name__ == "__main__":
    main()
