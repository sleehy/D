#!/usr/bin/env python3
"""Draw separate MLP equation-of-state curves for O- and t-FAPbI3.

Each phase starts from ``<phase>/CONTCAR``.  Following the SevenNet fine-tuning
tutorial, the structure and cell are first relaxed with the supplied MLP using
hydrostatic strain.  The relaxed cell is then isotropically scaled, with
fractional atomic coordinates held fixed, and evaluated by MLP single points.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.filters import UnitCellFilter
from ase.io import read, write
from ase.optimize import LBFGS
from sevenn.calculator import SevenNetCalculator


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR.parent
PHASES = ("O-FAPI3", "t-FAPI3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "checkpoint_fine_tuned.pth")
    parser.add_argument("--input-root", type=Path, default=ROOT, help="Root containing O-FAPI3/ and t-FAPI3/.")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "eos_results")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--min-linear-strain", type=float, default=-0.05, help="Minimum isotropic linear strain.")
    parser.add_argument("--max-linear-strain", type=float, default=0.05, help="Maximum isotropic linear strain.")
    parser.add_argument("--num-points", type=int, default=11, help="Number of equally spaced EOS points.")
    parser.add_argument("--fmax", type=float, default=0.02, help="MLP relaxation force threshold in eV/A.")
    parser.add_argument("--steps", type=int, default=1000, help="Maximum LBFGS relaxation steps.")
    parser.add_argument("--skip-relax", action="store_true", help="Scale CONTCAR directly rather than MLP-relaxing it first.")
    return parser.parse_args()


def relax_hydrostatic(atoms, calculator: SevenNetCalculator, fmax: float, steps: int, logfile: Path):
    """Relax ionic coordinates and a hydrostatic cell using the MLP."""
    relaxed = deepcopy(atoms)
    relaxed.calc = calculator
    cell_filter = UnitCellFilter(relaxed, hydrostatic_strain=True)
    optimizer = LBFGS(cell_filter, logfile=str(logfile))
    optimizer.run(fmax=fmax, steps=steps)
    return relaxed


def evaluate_scaled_structure(base, linear_strain: float, calculator: SevenNetCalculator):
    """Return a frozen one-shot MLP result after isotropic cell scaling."""
    atoms = base.copy()
    scale = 1.0 + linear_strain
    atoms.set_cell(base.cell * scale, scale_atoms=True)
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
    return atoms, energy


def write_curve_csv(rows: list[dict[str, float]], path: Path) -> None:
    fields = ("linear_strain", "volume_A3", "volume_per_atom_A3", "energy_eV", "energy_per_atom_eV", "relative_energy_meV_per_atom")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(rows: list[dict[str, float]], phase: str, path: Path) -> None:
    cache_dir = path.parent / ".matplotlib"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    volumes = np.array([row["volume_per_atom_A3"] for row in rows])
    energies = np.array([row["relative_energy_meV_per_atom"] for row in rows])
    minimum = int(np.argmin(energies))
    figure, axis = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    axis.plot(volumes, energies, color="tab:blue", linewidth=1.5)
    axis.scatter(volumes, energies, color="tab:blue", s=30, zorder=2, label="SevenNet fine-tuned")
    axis.scatter([volumes[minimum]], [energies[minimum]], color="tab:red", marker="*", s=120, zorder=3, label="sampled minimum")
    axis.set(
        title=f"{phase} EOS (MLP)",
        xlabel=r"Volume per atom ($\AA^3$/atom)",
        ylabel="Relative energy (meV/atom)",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def evaluate_phase(phase: str, args: argparse.Namespace, calculator: SevenNetCalculator) -> dict[str, float | str | bool]:
    source = args.input_root / phase / "CONTCAR"
    if not source.is_file():
        raise FileNotFoundError(f"Reference structure not found: {source}")
    original = read(source)
    logfile = args.output / f"{phase}_relax.log"
    if args.skip_relax:
        base = original.copy()
    else:
        print(f"Relaxing {phase} from {source} ...", flush=True)
        base = relax_hydrostatic(original, calculator, args.fmax, args.steps, logfile)
    base.calc = None

    strains = np.linspace(args.min_linear_strain, args.max_linear_strain, args.num_points)
    evaluated = []
    raw_rows = []
    for strain in strains:
        atoms, energy = evaluate_scaled_structure(base, float(strain), calculator)
        evaluated.append(atoms)
        raw_rows.append({
            "linear_strain": float(strain),
            "volume_A3": float(atoms.get_volume()),
            "volume_per_atom_A3": float(atoms.get_volume() / len(atoms)),
            "energy_eV": energy,
            "energy_per_atom_eV": energy / len(atoms),
        })
    energy_min = min(row["energy_per_atom_eV"] for row in raw_rows)
    rows = [{**row, "relative_energy_meV_per_atom": 1000 * (row["energy_per_atom_eV"] - energy_min)} for row in raw_rows]
    write_curve_csv(rows, args.output / f"eos_{phase}.csv")
    write(args.output / f"eos_{phase}.extxyz", evaluated, format="extxyz")
    plot_curve(rows, phase, args.output / f"eos_curve_{phase}.png")
    minimum = min(rows, key=lambda row: row["energy_per_atom_eV"])
    return {
        "phase": phase,
        "source": str(source),
        "mlp_relaxed_before_scaling": not args.skip_relax,
        "natoms": len(base),
        "initial_volume_A3": float(original.get_volume()),
        "base_volume_A3": float(base.get_volume()),
        "sampled_equilibrium_volume_per_atom_A3": minimum["volume_per_atom_A3"],
        "sampled_equilibrium_energy_per_atom_eV": minimum["energy_per_atom_eV"],
    }


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.num_points < 3:
        raise ValueError("--num-points must be at least 3")
    if args.min_linear_strain >= args.max_linear_strain:
        raise ValueError("--min-linear-strain must be smaller than --max-linear-strain")
    args.output.mkdir(parents=True, exist_ok=True)
    calculator = SevenNetCalculator(args.checkpoint, device=args.device)
    summaries = [evaluate_phase(phase, args, calculator) for phase in PHASES]
    with (args.output / "eos_summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)
        handle.write("\n")
    for summary in summaries:
        print(
            f"{summary['phase']}: sampled minimum at "
            f"V={summary['sampled_equilibrium_volume_per_atom_A3']:.5f} A^3/atom, "
            f"E={summary['sampled_equilibrium_energy_per_atom_eV']:.8f} eV/atom"
        )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
