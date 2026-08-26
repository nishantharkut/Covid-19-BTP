#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from covid_rars.dndt_dndf_evidence import generate_evidence


class CliArgumentError(ValueError):
    """Raised when command-line syntax is invalid."""


class MachineReadableArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliArgumentError(message)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = MachineReadableArgumentParser(
        description="Generate DNDT/DNDF evidence tables from saved predictions."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-bootstraps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if args.n_bootstraps < 1:
            raise CliArgumentError("--n-bootstraps must be positive")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("configuration root must be a JSON object")
        run_root = Path(str(config["run_root"]))
        manifest = generate_evidence(
            run_root / args.run_id,
            n_bootstraps=args.n_bootstraps,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "run_id": manifest["run_id"],
                    "evidence_dir": str((run_root / args.run_id / "evidence").resolve()),
                    "outputs": manifest["outputs"],
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
