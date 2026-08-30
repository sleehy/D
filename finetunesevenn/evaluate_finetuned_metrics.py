#!/usr/bin/env python3
"""Report energy, force, and stress MAE/RMSE for a SevenNet checkpoint.

Energy errors are evaluated per atom in meV/atom, force errors over all
Cartesian components in eV/Å, and stress errors over the six Voigt components
in GPa.  Every labelled frame in the supplied extxyz files is included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read
from sevenn.calculator import SevenNetCalculator


PROJECT_DIR = Path(__file__).resolve().parent
EV_PER_A3_TO_GPA = 160.21766208


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR / "checkpoint_fine_tuned_al_round1.pth",
        help="Checkpoint to evaluate.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_DIR / "dataset_al_round1",
        help="A labelled .extxyz file or a directory containing them.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "final_model_metrics.json",
        help="JSON file for the aggregate metrics.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device used by SevenNet.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N frames (for a smoke test).",
    )
    return parser.parse_args()


def metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = np.asarray(prediction, dtype=float).reshape(-1) - np.asarray(reference, dtype=float).reshape(-1)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def input_files(path: Path) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.rglob("*.extxyz"))
    if not files:
        raise FileNotFoundError(f"No .extxyz files found at {path}")
    return files


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")

    frames = [atoms for file in input_files(args.input) for atoms in read(file, index=":")]
    if args.limit is not None:
        frames = frames[:args.limit]
    if not frames:
        raise RuntimeError("No frames selected for evaluation")

    calculator = SevenNetCalculator(args.checkpoint, device=args.device)
    energy_reference, energy_prediction = [], []
    force_reference, force_prediction = [], []
    stress_reference, stress_prediction = [], []

    for index, atoms in enumerate(frames, start=1):
        # Read labels before replacing ASE's extxyz SinglePointCalculator.
        energy_reference.append(float(atoms.get_potential_energy()) / len(atoms))
        force_reference.append(np.asarray(atoms.get_forces(), dtype=float).reshape(-1))
        stress_reference.append(np.asarray(atoms.get_stress(voigt=True), dtype=float).reshape(-1))

        atoms.calc = calculator
        energy_prediction.append(float(atoms.get_potential_energy()) / len(atoms))
        force_prediction.append(np.asarray(atoms.get_forces(), dtype=float).reshape(-1))
        stress_prediction.append(np.asarray(atoms.get_stress(voigt=True), dtype=float).reshape(-1))
        print(f"[{index}/{len(frames)}] evaluated", flush=True)

    result = {
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "n_structures": len(frames),
        "energy_per_atom_meV": metrics(1000 * np.array(energy_reference), 1000 * np.array(energy_prediction)),
        "force_component_eV_per_A": metrics(np.concatenate(force_reference), np.concatenate(force_prediction)),
        "stress_voigt_GPa": metrics(
            EV_PER_A3_TO_GPA * np.concatenate(stress_reference),
            EV_PER_A3_TO_GPA * np.concatenate(stress_prediction),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"Energy / atom: MAE={result['energy_per_atom_meV']['mae']:.6g} meV/atom, "
          f"RMSE={result['energy_per_atom_meV']['rmse']:.6g} meV/atom")
    print(f"Force: MAE={result['force_component_eV_per_A']['mae']:.6g} eV/Å, "
          f"RMSE={result['force_component_eV_per_A']['rmse']:.6g} eV/Å")
    print(f"Stress: MAE={result['stress_voigt_GPa']['mae']:.6g} GPa, "
          f"RMSE={result['stress_voigt_GPa']['rmse']:.6g} GPa")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
