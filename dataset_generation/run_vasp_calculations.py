#!/usr/bin/env python3
"""Run all candidate VASP calculations sequentially with automatic resume."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_MPI_LAUNCHER = Path("/usr/bin/mpirun")
REQUIRED_INPUTS = ("INCAR", "KPOINTS", "POSCAR", "POTCAR")
FATAL_MARKERS = (
    "I REFUSE TO CONTINUE",
    "VERY BAD NEWS",
    "ERROR FEXCP",
    "ZBRENT: fatal error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_launcher = (
        str(SYSTEM_MPI_LAUNCHER) if SYSTEM_MPI_LAUNCHER.is_file() else "mpirun"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "mlp_dataset_candidates",
        help="Candidate dataset directory.",
    )
    parser.add_argument(
        "--vasp",
        default="vasp_std",
        help="VASP executable name or path (default: vasp_std).",
    )
    parser.add_argument(
        "--launcher",
        default=default_launcher,
        help=(
            "MPI launcher name or path "
            f"(default: {default_launcher}; use a launcher compatible with VASP)."
        ),
    )
    parser.add_argument(
        "--np",
        type=int,
        default=2,
        help="Number of MPI ranks for each calculation (default: 2).",
    )
    parser.add_argument(
        "--start-at",
        metavar="CONFIG_ID",
        help="Ignore configurations before this ID; completed ones are still skipped.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to the next configuration when VASP fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without launching VASP.",
    )
    args = parser.parse_args()
    if args.np < 1:
        parser.error("--np must be at least 1")
    return args


def calculation_is_complete(config_dir: Path) -> bool:
    """Use the same completion criteria as collect_vasp_results.py."""
    outcar = config_dir / "OUTCAR"
    vasprun = config_dir / "vasprun.xml"
    if not outcar.is_file() or not vasprun.is_file():
        return False
    text = outcar.read_text(errors="replace")
    if "General timing and accounting informations for this job:" not in text:
        return False
    if "aborting loop because EDIFF is reached" not in text:
        return False
    return not any(marker in text for marker in FATAL_MARKERS)


def resolve_program(program: str, description: str) -> str:
    path = shutil.which(program)
    if path is None:
        raise FileNotFoundError(f"{description} not found: {program}")
    return path


def validate_inputs(config_dir: Path) -> None:
    missing = [name for name in REQUIRED_INPUTS if not (config_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{config_dir.name}: missing required input(s): {', '.join(missing)}"
        )


def select_configurations(input_dir: Path, start_at: str | None) -> list[Path]:
    configs = sorted(path.parent for path in input_dir.glob("*/*/metadata.json"))
    if not configs:
        raise FileNotFoundError(f"No configuration metadata found under {input_dir}")
    if start_at is None:
        return configs
    matching = [index for index, path in enumerate(configs) if path.name == start_at]
    if not matching:
        raise ValueError(f"Unknown --start-at configuration: {start_at}")
    return configs[matching[0] :]


def run_one(config_dir: Path, command: list[str]) -> int:
    log_path = config_dir / "vasp.run.log"
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    with log_path.open("a") as log:
        log.write(
            f"\n===== {datetime.now().astimezone().isoformat()} START "
            f"{' '.join(command)} =====\n"
        )
        log.flush()
        result = subprocess.run(
            command,
            cwd=config_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
        log.write(
            f"===== {datetime.now().astimezone().isoformat()} EXIT "
            f"{result.returncode} =====\n"
        )
    return result.returncode


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    configs = select_configurations(input_dir, args.start_at)

    if args.dry_run:
        launcher = args.launcher
        vasp = args.vasp
    else:
        launcher = resolve_program(args.launcher, "MPI launcher")
        vasp = resolve_program(args.vasp, "VASP executable")
    command = [launcher, "-np", str(args.np), vasp]

    completed = sum(calculation_is_complete(path) for path in configs)
    pending = len(configs) - completed
    print(
        f"Found {len(configs)} configuration(s): {completed} complete, "
        f"{pending} pending."
    )

    run_number = 0
    failures = 0
    for config_dir in configs:
        if calculation_is_complete(config_dir):
            print(f"[SKIP] {config_dir.name} (already complete)")
            continue

        validate_inputs(config_dir)
        run_number += 1
        print(f"[RUN {run_number}/{pending}] {config_dir.name}", flush=True)
        if args.dry_run:
            print(f"  cwd: {config_dir}")
            print(f"  command: {' '.join(command)}")
            continue

        try:
            returncode = run_one(config_dir, command)
        except KeyboardInterrupt:
            print(
                f"\n[STOP] Interrupted in {config_dir.name}. Re-run the same "
                "command to resume here.",
                file=sys.stderr,
            )
            return 130

        if returncode != 0 or not calculation_is_complete(config_dir):
            failures += 1
            print(
                f"[FAIL] {config_dir.name}; see {config_dir / 'vasp.run.log'}",
                file=sys.stderr,
            )
            if not args.keep_going:
                print("Re-run the same command after fixing the error to resume.")
                return 1
        else:
            print(f"[DONE] {config_dir.name}")

    if args.dry_run:
        print(f"Dry run complete: {pending} calculation(s) would be launched.")
        return 0
    if failures:
        print(f"Finished with {failures} failed calculation(s).", file=sys.stderr)
        return 1
    print("All selected VASP calculations are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
