#!/usr/bin/env python3
"""Collect completed VASP single-point calculations into labeled extxyz."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "mlp_dataset_candidates",
        help="Candidate dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output extxyz (default: INPUT/labeled.extxyz).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write available results even when some configurations are missing.",
    )
    return parser.parse_args()


def electronic_convergence_from_outcar(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "OUTCAR missing"
    text = path.read_text(errors="replace")
    if "General timing and accounting informations for this job:" not in text:
        return False, "OUTCAR is incomplete"
    if "aborting loop because EDIFF is reached" not in text:
        return False, "EDIFF convergence marker missing"
    fatal_markers = (
        "I REFUSE TO CONTINUE",
        "VERY BAD NEWS",
        "ERROR FEXCP",
        "ZBRENT: fatal error",
    )
    for marker in fatal_markers:
        if marker in text:
            return False, f"fatal marker found: {marker}"
    return True, "ok"


def read_result(config_dir: Path):
    vasprun = config_dir / "vasprun.xml"
    outcar = config_dir / "OUTCAR"
    converged, reason = electronic_convergence_from_outcar(outcar)
    if not converged:
        raise RuntimeError(reason)
    if not vasprun.is_file():
        raise RuntimeError("vasprun.xml missing")

    atoms = read(vasprun, index=-1)
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
    if forces.shape != (len(atoms), 3):
        raise RuntimeError(f"unexpected force shape {forces.shape}")
    if stress.shape != (3, 3):
        raise RuntimeError(f"unexpected stress shape {stress.shape}")
    if not (
        np.isfinite(energy)
        and np.isfinite(forces).all()
        and np.isfinite(stress).all()
    ):
        raise RuntimeError("energy, forces, or stress contains non-finite values")

    metadata_path = config_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    atoms.info.update(
        {
            "config_id": metadata["config_id"],
            "phase": metadata["phase"],
            "mode": metadata["mode"],
            "seed": metadata["seed"],
            "source": metadata["source"],
        }
    )
    atoms.calc = SinglePointCalculator(
        atoms, energy=energy, forces=forces, stress=stress
    )
    return atoms, energy, forces, stress


def main() -> None:
    args = parse_args()
    output = args.output or args.input / "labeled.extxyz"
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose another --output.")

    metadata_files = sorted(args.input.glob("*/*/metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No configuration metadata found under {args.input}.")

    frames = []
    rows = []
    failures = []
    for metadata_path in metadata_files:
        config_dir = metadata_path.parent
        try:
            atoms, energy, forces, stress = read_result(config_dir)
        except Exception as exc:
            failures.append(
                {
                    "config_id": config_dir.name,
                    "directory": str(config_dir),
                    "reason": str(exc),
                }
            )
            continue

        force_norms = np.linalg.norm(forces, axis=1)
        frames.append(atoms)
        rows.append(
            {
                "config_id": atoms.info["config_id"],
                "phase": atoms.info["phase"],
                "mode": atoms.info["mode"],
                "energy_eV": energy,
                "energy_per_atom_eV": energy / len(atoms),
                "max_force_eV_per_A": float(force_norms.max()),
                "rms_force_eV_per_A": float(np.sqrt(np.mean(force_norms**2))),
                "max_abs_stress_eV_per_A3": float(np.abs(stress).max()),
                "directory": str(config_dir.relative_to(args.input)),
            }
        )

    failure_report = args.input / "collection_failures.csv"
    with failure_report.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("config_id", "directory", "reason")
        )
        writer.writeheader()
        writer.writerows(failures)

    if failures and not args.allow_partial:
        raise RuntimeError(
            f"{len(failures)} of {len(metadata_files)} calculations are missing or "
            f"invalid. See {failure_report}. Use --allow-partial only if intended."
        )
    if not frames:
        raise RuntimeError("No valid completed calculations were found.")

    write(output, frames, format="extxyz")
    summary_path = output.with_suffix(".csv")
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Collected {len(frames)} labeled structures into {output}")
    print(f"Summary: {summary_path}")
    if failures:
        print(f"Skipped {len(failures)} invalid or missing calculations.")


if __name__ == "__main__":
    main()
