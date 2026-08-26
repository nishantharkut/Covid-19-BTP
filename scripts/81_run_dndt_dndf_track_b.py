#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence, TypeVar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from covid_rars.dndt_dndf_experiment import run_track_b


T = TypeVar("T", str, int)
ALLOWED_MODALITIES = ("breath", "cough", "speech")
ALLOWED_PROTOCOLS = (
    "early_to_late",
    "existing",
    "external_cough",
    "time_stratified",
)
ALLOWED_SEEDS = (42, 314, 2026)


class CliArgumentError(ValueError):
    """Raised when CLI syntax or a CLI-only value is invalid."""


class MachineReadableArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliArgumentError(message)


def _parse_comma_subset(
    value: str,
    *,
    name: str,
    converter: Callable[[str], T],
    allowed: Sequence[T],
) -> tuple[T, ...]:
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{name} must be a nonempty comma list")
    try:
        parsed = tuple(converter(token) for token in tokens)
    except ValueError as exc:
        raise ValueError(f"{name} contains an invalid value") from exc
    if tuple(sorted(set(parsed))) != parsed:
        raise ValueError(f"{name} must be sorted and unique")
    invalid = [item for item in parsed if item not in allowed]
    if invalid:
        raise ValueError(f"{name} must contain only allowed values; found {invalid}")
    return parsed


def parse_modalities(value: str) -> tuple[str, ...]:
    return _parse_comma_subset(
        value,
        name="modalities",
        converter=str,
        allowed=ALLOWED_MODALITIES,
    )


def parse_protocols(value: str) -> tuple[str, ...]:
    return _parse_comma_subset(
        value,
        name="protocols",
        converter=str,
        allowed=ALLOWED_PROTOCOLS,
    )


def parse_seeds(value: str) -> tuple[int, ...]:
    return _parse_comma_subset(
        value,
        name="seeds",
        converter=int,
        allowed=ALLOWED_SEEDS,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = MachineReadableArgumentParser(
        description="Run bounded, auditable DNDT/DNDF Track B stages."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "candidates",
            "final",
            "prespecified_ladder_v2",
            "fusion",
            "shuffle",
        ),
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--modalities")
    parser.add_argument("--protocols")
    parser.add_argument("--seeds")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args(argv)


def _git_output(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(
            f"git {' '.join(arguments)} failed for {repository}: {detail}"
        )
    return process.stdout.strip()


def _code_revision(repository: Path = PROJECT_ROOT) -> str:
    tracked_status = _git_output(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if tracked_status:
        raise RuntimeError(
            "source tree is dirty; commit, remove, or ignore source changes before "
            "scientific execution"
        )
    revision = _git_output(repository, "rev-parse", "HEAD")
    if not revision:
        raise RuntimeError("cannot resolve a nonempty code revision")
    return revision


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., dict[str, object]] = run_track_b,
    revision_resolver: Callable[[], str] | None = None,
) -> int:
    try:
        args = _parse_args(argv)
        if args.protocols is not None and args.stage != "prespecified_ladder_v2":
            raise CliArgumentError(
                "--protocols is only valid for --stage prespecified_ladder_v2"
            )
        if args.seeds is not None and args.stage != "final":
            raise CliArgumentError("--seeds is only valid for --stage final")
        try:
            modalities = (
                parse_modalities(args.modalities) if args.modalities is not None else None
            )
            protocols = (
                parse_protocols(args.protocols) if args.protocols is not None else None
            )
            seeds = parse_seeds(args.seeds) if args.seeds is not None else None
        except ValueError as exc:
            raise CliArgumentError(str(exc)) from exc
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("configuration root must be a JSON object")
        revision = (revision_resolver or _code_revision)()
        run_id = f"{args.run_id}-smoke" if args.smoke else args.run_id
        result = runner(
            config,
            run_id=run_id,
            stage=args.stage,
            resume=args.resume,
            smoke=args.smoke,
            modalities=modalities,
            protocols=protocols,
            seeds=seeds,
            code_revision=revision,
            device=args.device or config.get("device", "cuda"),
        )
        progress = {key: value for key, value in result.items() if key != "results"}
        print(json.dumps(progress, sort_keys=True, allow_nan=False))
        return 0 if result.get("status") in {"complete", "partial"} else 1
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
