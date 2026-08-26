#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from covid_rars.dndt_dndf_experiment import run_track_a


class CliArgumentError(ValueError):
    """Raised when CLI syntax or a CLI-only value is invalid."""


class MachineReadableArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliArgumentError(message)


def parse_fold_batch(value: str | None) -> tuple[int, ...]:
    if value is None:
        return tuple(range(10))
    token = value.strip()
    if not token:
        raise ValueError("fold batch cannot be empty")
    if "," in token:
        try:
            folds = tuple(int(item.strip()) for item in token.split(","))
        except ValueError as exc:
            raise ValueError("fold batch comma list must contain integers") from exc
    elif "-" in token:
        parts = token.split("-")
        if len(parts) != 2:
            raise ValueError("fold batch range must be START-END")
        try:
            start, end = (int(part.strip()) for part in parts)
        except ValueError as exc:
            raise ValueError("fold batch range must contain integers") from exc
        if start > end:
            raise ValueError("fold batch range start cannot exceed end")
        folds = tuple(range(start, end + 1))
    else:
        try:
            folds = (int(token),)
        except ValueError as exc:
            raise ValueError("fold batch must be an integer, range, or comma list") from exc
    if not folds or tuple(sorted(set(folds))) != folds:
        raise ValueError("fold batch must be sorted, unique, and nonempty")
    if folds[0] < 0 or folds[-1] > 9:
        raise ValueError("fold batch must be within 0..9")
    return folds


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = MachineReadableArgumentParser(
        description="Run the auditable DNDT/DNDF author-artifact reproduction."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--fold-batch",
        help="Inclusive range such as 0-2, a comma list such as 0,3,5, or one fold.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "author_behaviour_audit",
            "fresh_fold_author_protocol",
            "corrected_reference",
        ),
        default=(
            "author_behaviour_audit",
            "fresh_fold_author_protocol",
            "corrected_reference",
        ),
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        choices=("dndt", "dndf"),
        default=("dndf",),
    )
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
        "--untracked-files=no",
    )
    if tracked_status:
        raise RuntimeError(
            "tracked source tree is dirty; commit or restore tracked changes before "
            "scientific execution"
        )
    revision = _git_output(repository, "rev-parse", "HEAD")
    if not revision:
        raise RuntimeError("cannot resolve a nonempty code revision")
    return revision


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., dict[str, object]] = run_track_a,
    revision_resolver: Callable[[], str] | None = None,
) -> int:
    try:
        args = _parse_args(argv)
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("configuration root must be a JSON object")
        try:
            folds = parse_fold_batch(args.fold_batch)
        except ValueError as exc:
            raise CliArgumentError(str(exc)) from exc
        run_id = f"{args.run_id}-smoke" if args.smoke else args.run_id
        if args.smoke:
            folds = (folds[0],)
        revision = (revision_resolver or _code_revision)()
        result = runner(
            config,
            run_id=run_id,
            resume=args.resume,
            smoke=args.smoke,
            fold_batch=folds,
            modes=tuple(args.modes),
            model_names=tuple(args.model_names),
            code_revision=revision,
            device=args.device or config.get("device", "cuda"),
        )
        progress = {
            key: value
            for key, value in result.items()
            if key not in {"metrics", "predictions"}
        }
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
