#!/usr/bin/env python3
"""Evaluate a SevenNet checkpoint on the independent FAPbI3 parity set.

The script uses the final VASP frame in each candidate directory as the DFT
reference, then writes a combined energy/force/stress parity plot and
machine-readable per-frame and aggregate metrics.  It also writes the
structures furthest from the y=x line, split into tetragonal/orthorhombic and
energy/force CSV files.  For those structures, it also compares Pb-I, H-Pb,
and H-I distances with the phase CONTCAR (index-matched Pb-I; minimum H-Pb
and H-I contacts).
The reference and model stress values both follow ASE's stress convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.io import read
from sevenn.calculator import SevenNetCalculator


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR.parent
EV_PER_A3_TO_GPA = 160.21766208
ENERGY_OUTLIER_FIELDS = (
    "config_id", "phase", "crystal_system", "family", "severity", "natoms",
    "reference_eV_per_atom", "prediction_eV_per_atom", "signed_residual_meV_per_atom",
    "absolute_residual_meV_per_atom", "perpendicular_distance_meV_per_atom",
)
FORCE_OUTLIER_FIELDS = (
    "config_id", "phase", "crystal_system", "family", "severity", "natoms",
    "atom_index_zero_based", "cartesian_component", "reference_force_eV_per_A",
    "prediction_force_eV_per_A", "signed_residual_eV_per_A", "absolute_residual_eV_per_A",
    "perpendicular_distance_eV_per_A", "structure_force_mae_eV_per_A",
    "structure_force_rmse_eV_per_A",
)
PAIR_DISTANCE_FIELDS = (
    "config_id", "phase", "crystal_system", "family", "outlier_selection", "pair_type",
    "distance_method", "contcar_atom_i_zero_based", "contcar_symbol_i",
    "contcar_atom_j_zero_based", "contcar_symbol_j", "outlier_atom_i_zero_based",
    "outlier_symbol_i", "outlier_atom_j_zero_based", "outlier_symbol_j",
    "contcar_distance_A", "outlier_distance_A", "signed_distance_change_A",
    "absolute_distance_change_A",
)


@dataclass
class FrameResult:
    config_id: str
    phase: str
    family: str
    severity: float
    natoms: int
    energy_ref_eV_per_atom: float
    energy_pred_eV_per_atom: float
    force_ref_eV_per_A: np.ndarray
    force_pred_eV_per_A: np.ndarray
    stress_ref_GPa: np.ndarray
    stress_pred_GPa: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR / "checkpoint_fine_tuned.pth",
        help="Fine-tuned SevenNet checkpoint.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "parity_dataset_candidates",
        help="Directory containing parity candidate subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "parity_results",
        help="Directory for plots and CSV output.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device used by SevenNet (default: auto).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write results from valid VASP calculations even if some fail validation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N structures; intended for quick smoke tests.",
    )
    parser.add_argument(
        "--outlier-top-n",
        type=int,
        default=10,
        help=(
            "Number of highest-residual structures to write per crystal-system/property "
            "when that property's threshold is not given (default: 10)."
        ),
    )
    parser.add_argument(
        "--energy-outlier-threshold-meV-per-atom",
        type=float,
        default=None,
        help=(
            "Write every structure whose absolute energy residual from y=x is at least "
            "this value. Overrides --outlier-top-n for energy."
        ),
    )
    parser.add_argument(
        "--force-outlier-threshold-eV-per-A",
        type=float,
        default=None,
        help=(
            "Write every structure whose largest absolute force-component residual from "
            "y=x is at least this value. Overrides --outlier-top-n for force."
        ),
    )
    parser.add_argument(
        "--pair-distance-cutoff-A",
        type=float,
        default=5.0,
        help=(
            "Only compare index-matched Pb-I pairs no farther apart than this in the "
            "phase CONTCAR (default: 5.0 A). H-Pb and H-I always use their minimum "
            "distance in each structure."
        ),
    )
    return parser.parse_args()


def electronic_convergence_from_outcar(path: Path) -> tuple[bool, str]:
    """Use the same VASP completeness and EDIFF checks as label collection."""
    if not path.is_file():
        return False, "OUTCAR missing"
    text = path.read_text(errors="replace")
    if "General timing and accounting informations for this job:" not in text:
        return False, "OUTCAR is incomplete"
    if "aborting loop because EDIFF is reached" not in text:
        return False, "EDIFF convergence marker missing"
    for marker in ("I REFUSE TO CONTINUE", "VERY BAD NEWS", "ERROR FEXCP", "ZBRENT: fatal error"):
        if marker in text:
            return False, f"fatal marker found: {marker}"
    return True, "ok"


def compute_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    residual = prediction - reference
    squared_error = np.sum(residual**2)
    total_variance = np.sum((reference - reference.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - squared_error / total_variance) if total_variance > 0 else float("nan"),
    }


def read_reference(config_dir: Path):
    converged, reason = electronic_convergence_from_outcar(config_dir / "OUTCAR")
    if not converged:
        raise RuntimeError(reason)
    vasprun = config_dir / "vasprun.xml"
    if not vasprun.is_file():
        raise RuntimeError("vasprun.xml missing")
    atoms = read(vasprun, index=-1)
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    # Six independent components avoid counting off-diagonal tensor entries twice.
    stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    if not (np.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(stress).all()):
        raise RuntimeError("energy, forces, or stress contains non-finite values")
    return atoms, energy, forces, stress


def evaluate_frame(metadata_path: Path, calculator: SevenNetCalculator) -> FrameResult:
    metadata = json.loads(metadata_path.read_text())
    atoms, energy_ref, forces_ref, stress_ref = read_reference(metadata_path.parent)
    # `atoms` comes from vasprun.xml, so the MLP sees exactly the frame whose DFT
    # labels are used as the reference.  Overwrite the VASP calculator with MLP.
    atoms.calc = calculator
    energy_pred = float(atoms.get_potential_energy())
    forces_pred = np.asarray(atoms.get_forces(), dtype=float)
    stress_pred = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    if forces_pred.shape != forces_ref.shape or stress_pred.shape != stress_ref.shape:
        raise RuntimeError("MLP output shape differs from DFT reference shape")
    if not (np.isfinite(energy_pred) and np.isfinite(forces_pred).all() and np.isfinite(stress_pred).all()):
        raise RuntimeError("MLP energy, forces, or stress contains non-finite values")
    return FrameResult(
        config_id=str(metadata["config_id"]),
        phase=str(metadata["phase"]),
        family=str(metadata["distortion_family"]),
        severity=float(metadata["distortion_severity"]),
        natoms=len(atoms),
        energy_ref_eV_per_atom=energy_ref / len(atoms),
        energy_pred_eV_per_atom=energy_pred / len(atoms),
        force_ref_eV_per_A=forces_ref,
        force_pred_eV_per_A=forces_pred,
        stress_ref_GPa=stress_ref * EV_PER_A3_TO_GPA,
        stress_pred_GPa=stress_pred * EV_PER_A3_TO_GPA,
    )


def write_frame_csv(results: list[FrameResult], path: Path) -> None:
    fields = (
        "config_id", "phase", "family", "severity", "natoms",
        "energy_ref_eV_per_atom", "energy_pred_eV_per_atom", "energy_error_meV_per_atom",
        "force_mae_eV_per_A", "force_rmse_eV_per_A",
        "stress_mae_GPa", "stress_rmse_GPa",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            force = compute_metrics(item.force_ref_eV_per_A, item.force_pred_eV_per_A)
            stress = compute_metrics(item.stress_ref_GPa, item.stress_pred_GPa)
            writer.writerow({
                "config_id": item.config_id,
                "phase": item.phase,
                "family": item.family,
                "severity": item.severity,
                "natoms": item.natoms,
                "energy_ref_eV_per_atom": item.energy_ref_eV_per_atom,
                "energy_pred_eV_per_atom": item.energy_pred_eV_per_atom,
                "energy_error_meV_per_atom": 1000 * abs(item.energy_pred_eV_per_atom - item.energy_ref_eV_per_atom),
                "force_mae_eV_per_A": force["mae"],
                "force_rmse_eV_per_A": force["rmse"],
                "stress_mae_GPa": stress["mae"],
                "stress_rmse_GPa": stress["rmse"],
            })


def metric_rows(results: list[FrameResult]) -> list[dict[str, object]]:
    groups: dict[str, list[FrameResult]] = {"all": results}
    for phase in sorted({item.phase for item in results}):
        groups[f"phase:{phase}"] = [item for item in results if item.phase == phase]
    for family in sorted({item.family for item in results}):
        groups[f"family:{family}"] = [item for item in results if item.family == family]

    rows = []
    for scope, items in groups.items():
        targets = {
            "energy_per_atom": (
                np.array([item.energy_ref_eV_per_atom for item in items]),
                np.array([item.energy_pred_eV_per_atom for item in items]),
                "eV/atom",
            ),
            "force_component": (
                np.concatenate([item.force_ref_eV_per_A.reshape(-1) for item in items]),
                np.concatenate([item.force_pred_eV_per_A.reshape(-1) for item in items]),
                "eV/A",
            ),
            "stress_voigt": (
                np.concatenate([item.stress_ref_GPa for item in items]),
                np.concatenate([item.stress_pred_GPa for item in items]),
                "GPa",
            ),
        }
        for target, (reference, prediction, unit) in targets.items():
            rows.append({
                "scope": scope,
                "target": target,
                "unit": unit,
                "n_structures": len(items),
                "n_values": reference.size,
                **compute_metrics(reference, prediction),
            })
    return rows


def write_metrics_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = ("scope", "target", "unit", "n_structures", "n_values", "mae", "rmse", "r2")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def crystal_system(phase: str) -> str:
    """Map the parity-set phase names to the requested crystal-system labels."""
    phase_map = {"t-FAPI3": "tetragonal", "O-FAPI3": "orthorhombic"}
    try:
        return phase_map[phase]
    except KeyError as exc:
        raise ValueError(
            f"Cannot assign crystal system for phase {phase!r}; update crystal_system()."
        ) from exc


def energy_outlier_row(item: FrameResult) -> dict[str, object]:
    residual_meV_per_atom = 1000 * (item.energy_pred_eV_per_atom - item.energy_ref_eV_per_atom)
    return {
        "config_id": item.config_id,
        "phase": item.phase,
        "crystal_system": crystal_system(item.phase),
        "family": item.family,
        "severity": item.severity,
        "natoms": item.natoms,
        "reference_eV_per_atom": item.energy_ref_eV_per_atom,
        "prediction_eV_per_atom": item.energy_pred_eV_per_atom,
        "signed_residual_meV_per_atom": residual_meV_per_atom,
        # For y=x, perpendicular distance is |prediction-reference| / sqrt(2).
        "absolute_residual_meV_per_atom": abs(residual_meV_per_atom),
        "perpendicular_distance_meV_per_atom": abs(residual_meV_per_atom) / np.sqrt(2),
    }


def force_outlier_row(item: FrameResult) -> dict[str, object]:
    """Summarise the force-parity point furthest from y=x for one structure."""
    residual = item.force_pred_eV_per_A - item.force_ref_eV_per_A
    absolute_residual = np.abs(residual)
    atom_index, component_index = np.unravel_index(np.argmax(absolute_residual), residual.shape)
    component = ("x", "y", "z")[component_index]
    max_residual = float(absolute_residual[atom_index, component_index])
    return {
        "config_id": item.config_id,
        "phase": item.phase,
        "crystal_system": crystal_system(item.phase),
        "family": item.family,
        "severity": item.severity,
        "natoms": item.natoms,
        "atom_index_zero_based": int(atom_index),
        "cartesian_component": component,
        "reference_force_eV_per_A": float(item.force_ref_eV_per_A[atom_index, component_index]),
        "prediction_force_eV_per_A": float(item.force_pred_eV_per_A[atom_index, component_index]),
        "signed_residual_eV_per_A": float(residual[atom_index, component_index]),
        "absolute_residual_eV_per_A": max_residual,
        "perpendicular_distance_eV_per_A": max_residual / np.sqrt(2),
        "structure_force_mae_eV_per_A": float(np.mean(absolute_residual)),
        "structure_force_rmse_eV_per_A": float(np.sqrt(np.mean(residual**2))),
    }


def select_outliers(
    rows: list[dict[str, object]], *, metric: str, threshold: float | None, top_n: int
) -> list[dict[str, object]]:
    """Select all points above a user threshold, or the N largest residuals."""
    ranked = sorted(rows, key=lambda row: float(row[metric]), reverse=True)
    if threshold is not None:
        # Keep values that mathematically equal the threshold despite binary
        # floating-point round-off (for example 0.01 eV/atom -> 10 meV/atom).
        tolerance = max(1.0e-12, abs(threshold) * 1.0e-12)
        return [row for row in ranked if float(row[metric]) >= threshold - tolerance]
    return ranked[:top_n]


def write_outlier_csvs(
    results: list[FrameResult], output_dir: Path, *, top_n: int,
    energy_threshold_meV_per_atom: float | None, force_threshold_eV_per_A: float | None,
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    """Write tetragonal/orthorhombic energy/force outlier tables and a summary."""
    if top_n < 1:
        raise ValueError("--outlier-top-n must be at least 1")
    if energy_threshold_meV_per_atom is not None and energy_threshold_meV_per_atom < 0:
        raise ValueError("--energy-outlier-threshold-meV-per-atom must be non-negative")
    if force_threshold_eV_per_A is not None and force_threshold_eV_per_A < 0:
        raise ValueError("--force-outlier-threshold-eV-per-A must be non-negative")

    specifications = (
        (
            "energy", energy_outlier_row, "absolute_residual_meV_per_atom",
            energy_threshold_meV_per_atom, ENERGY_OUTLIER_FIELDS,
        ),
        (
            "force", force_outlier_row, "absolute_residual_eV_per_A",
            force_threshold_eV_per_A, FORCE_OUTLIER_FIELDS,
        ),
    )
    summary: dict[str, dict[str, object]] = {}
    selections: dict[str, list[dict[str, object]]] = {}
    for system in ("tetragonal", "orthorhombic"):
        phase_results = [item for item in results if crystal_system(item.phase) == system]
        for property_name, row_builder, metric, threshold, fieldnames in specifications:
            selected = select_outliers(
                [row_builder(item) for item in phase_results],
                metric=metric,
                threshold=threshold,
                top_n=top_n,
            )
            filename = f"outliers_{system}_{property_name}.csv"
            path = output_dir / filename
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(selected)
            label = f"{system}_{property_name}"
            selections[label] = selected
            summary[label] = {
                "file": filename,
                "n_selected": len(selected),
                "ranking_metric": metric,
                "threshold": threshold,
                "top_n_when_no_threshold": top_n,
            }
    return summary, selections


def pair_distance_rows(
    selections: dict[str, list[dict[str, object]]], input_dir: Path, reference_root: Path,
    crystal_system_name: str, cutoff_A: float,
) -> list[dict[str, object]]:
    """Compare Pb-I by index, and H-Pb/H-I by each structure's closest pair."""
    if cutoff_A <= 0:
        raise ValueError("--pair-distance-cutoff-A must be positive")

    selected_configs: dict[str, dict[str, object]] = {}
    for property_name in ("energy", "force"):
        for row in selections[f"{crystal_system_name}_{property_name}"]:
            config_id = str(row["config_id"])
            details = selected_configs.setdefault(
                config_id, {"selected_for": set(), "family": str(row["family"])},
            )
            if details["family"] != str(row["family"]):
                raise ValueError(f"Inconsistent distortion family for {config_id}")
            details["selected_for"].add(property_name)
    if not selected_configs:
        return []

    phase = next(
        str(row["phase"])
        for property_name in ("energy", "force")
        for row in selections[f"{crystal_system_name}_{property_name}"]
    )
    contcar_path = reference_root / phase / "CONTCAR"
    if not contcar_path.is_file():
        raise FileNotFoundError(f"Phase reference CONTCAR not found: {contcar_path}")
    reference_atoms = read(contcar_path)
    reference_symbols = reference_atoms.get_chemical_symbols()

    def closest_pair(atoms, first_symbol: str, second_symbol: str) -> tuple[float, int, int]:
        """Return the minimum-image closest pair, retaining the atom indices found."""
        first_indices = [index for index, symbol in enumerate(atoms.symbols) if symbol == first_symbol]
        second_indices = [index for index, symbol in enumerate(atoms.symbols) if symbol == second_symbol]
        if not first_indices or not second_indices:
            raise ValueError(f"Structure does not contain both {first_symbol} and {second_symbol}")
        return min(
            (float(atoms.get_distance(i, j, mic=True)), i, j)
            for i in first_indices for j in second_indices
        )

    closest_reference_pairs = {
        "H-Pb": closest_pair(reference_atoms, "H", "Pb"),
        "H-I": closest_pair(reference_atoms, "H", "I"),
    }

    rows: list[dict[str, object]] = []
    for config_id, details in sorted(selected_configs.items()):
        selected_for = details["selected_for"]
        family = str(details["family"])
        outlier_path = input_dir / phase / config_id / "vasprun.xml"
        if not outlier_path.is_file():
            raise FileNotFoundError(f"Outlier structure vasprun.xml not found: {outlier_path}")
        outlier_atoms = read(outlier_path, index=-1)
        outlier_symbols = outlier_atoms.get_chemical_symbols()
        if outlier_symbols != reference_symbols:
            raise ValueError(
                f"Atom order/species differ between {contcar_path} and {outlier_path}; "
                "cannot make an index-matched comparison."
            )
        selection_label = "both" if len(selected_for) == 2 else next(iter(selected_for))
        def append_row(
            pair_type: str, distance_method: str,
            reference_pair: tuple[float, int, int], outlier_pair: tuple[float, int, int],
        ) -> None:
            reference_distance, reference_i, reference_j = reference_pair
            outlier_distance, outlier_i, outlier_j = outlier_pair
            distance_change = outlier_distance - reference_distance
            rows.append({
                "config_id": config_id,
                "phase": phase,
                "crystal_system": crystal_system_name,
                "family": family,
                "outlier_selection": selection_label,
                "pair_type": pair_type,
                "distance_method": distance_method,
                "contcar_atom_i_zero_based": reference_i,
                "contcar_symbol_i": reference_symbols[reference_i],
                "contcar_atom_j_zero_based": reference_j,
                "contcar_symbol_j": reference_symbols[reference_j],
                "outlier_atom_i_zero_based": outlier_i,
                "outlier_symbol_i": outlier_symbols[outlier_i],
                "outlier_atom_j_zero_based": outlier_j,
                "outlier_symbol_j": outlier_symbols[outlier_j],
                "contcar_distance_A": reference_distance,
                "outlier_distance_A": outlier_distance,
                "signed_distance_change_A": distance_change,
                "absolute_distance_change_A": abs(distance_change),
            })

        # Inorganic Pb-I cage atoms retain their atom correspondence, so use
        # each original Pb-I pair (within the local-reference cutoff).
        for i, symbol_i in enumerate(reference_symbols):
            if symbol_i != "Pb":
                continue
            for j, symbol_j in enumerate(reference_symbols):
                if symbol_j != "I":
                    continue
                reference_pair = (float(reference_atoms.get_distance(i, j, mic=True)), i, j)
                if reference_pair[0] <= cutoff_A:
                    outlier_pair = (float(outlier_atoms.get_distance(i, j, mic=True)), i, j)
                    append_row("Pb-I", "index_matched", reference_pair, outlier_pair)

        # FA rotations make H indices non-corresponding.  Use the closest
        # H-Pb/H-I contact separately in the CONTCAR and outlier structures.
        for pair_type, first_symbol, second_symbol in (("H-Pb", "H", "Pb"), ("H-I", "H", "I")):
            append_row(
                pair_type,
                "minimum_pair",
                closest_reference_pairs[pair_type],
                closest_pair(outlier_atoms, first_symbol, second_symbol),
            )
    return rows


def write_pair_distance_comparison(
    rows: list[dict[str, object]], crystal_system_name: str, output_dir: Path,
) -> str | None:
    """Write pair-distance data and a three-panel CONTCAR-vs-outlier plot."""
    csv_filename = f"outlier_pair_distances_{crystal_system_name}.csv"
    with (output_dir / csv_filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_DISTANCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if not rows:
        return None

    plot_cache = output_dir / ".matplotlib"
    plot_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    family_colors = {
        "strong_strain": "tab:red",
        "global_rattle": "tab:orange",
        "cage_rattle": "tab:green",
        "fa_rotation_cage_rattle": "tab:purple",
        "combined": "tab:brown",
    }
    for axis, pair_type in zip(axes, ("Pb-I", "H-Pb", "H-I")):
        pair_rows = [row for row in rows if row["pair_type"] == pair_type]
        values = np.array([
            value
            for row in pair_rows
            for value in (row["contcar_distance_A"], row["outlier_distance_A"])
        ])
        lower, upper = float(values.min()), float(values.max())
        pad = max((upper - lower) * 0.05, 0.05)
        lower, upper = lower - pad, upper + pad
        axis.plot((lower, upper), (lower, upper), "k--", linewidth=1, label="y = x")
        for family, color in family_colors.items():
            selected_rows = [row for row in pair_rows if row["family"] == family]
            if selected_rows:
                axis.scatter(
                    [row["contcar_distance_A"] for row in selected_rows],
                    [row["outlier_distance_A"] for row in selected_rows],
                    s=13, alpha=0.50, color=color, label=family, rasterized=True,
                )
        axis.set(
            xlim=(lower, upper), ylim=(lower, upper), title=f"{pair_type} ({len(pair_rows)} pairs)",
            xlabel="Phase CONTCAR distance (A)", ylabel="Outlier structure distance (A)",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small", loc="upper left")
    figure.suptitle(f"{crystal_system_name.capitalize()}: CONTCAR vs outlier pair distances")
    plot_filename = f"outlier_pair_distance_comparison_{crystal_system_name}.png"
    figure.savefig(output_dir / plot_filename, dpi=220)
    plt.close(figure)
    return plot_filename


def plot_parity(results: list[FrameResult], phase: str, path: Path) -> None:
    # Some shared compute environments have a read-only home directory.  Keep
    # Matplotlib's cache beside the explicitly requested output files instead.
    plot_cache = path.parent / ".matplotlib"
    plot_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = (
        ("Energy / atom", "DFT energy (eV/atom)", "MLP energy (eV/atom)",
         np.array([item.energy_ref_eV_per_atom for item in results]),
         np.array([item.energy_pred_eV_per_atom for item in results])),
        ("Force components", "DFT force (eV/A)", "MLP force (eV/A)",
         np.concatenate([item.force_ref_eV_per_A.reshape(-1) for item in results]),
         np.concatenate([item.force_pred_eV_per_A.reshape(-1) for item in results])),
        ("Stress (Voigt)", "DFT stress (GPa)", "MLP stress (GPa)",
         np.concatenate([item.stress_ref_GPa for item in results]),
         np.concatenate([item.stress_pred_GPa for item in results])),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, (title, xlabel, ylabel, reference, prediction) in zip(axes, targets):
        lower, upper = min(reference.min(), prediction.min()), max(reference.max(), prediction.max())
        pad = max((upper - lower) * 0.05, 1.0e-6)
        lower, upper = lower - pad, upper + pad
        axis.plot((lower, upper), (lower, upper), "k--", linewidth=1, label="y = x")
        axis.scatter(reference, prediction, s=13, alpha=0.55, color="tab:blue", label=phase, rasterized=True)
        metrics = compute_metrics(reference, prediction)
        axis.text(
            0.04, 0.96,
            f"MAE = {metrics['mae']:.3g}\nRMSE = {metrics['rmse']:.3g}\nR² = {metrics['r2']:.4f}",
            transform=axis.transAxes, va="top", fontsize="small",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        axis.set(xlim=(lower, upper), ylim=(lower, upper), title=f"{phase}: {title}", xlabel=xlabel, ylabel=ylabel)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small", loc="lower right")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    metadata_files = sorted(args.input.glob("*/*/metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No parity metadata files found under {args.input}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        metadata_files = metadata_files[:args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    calculator = SevenNetCalculator(args.checkpoint, device=args.device)
    results: list[FrameResult] = []
    failures: list[dict[str, str]] = []
    for index, metadata_path in enumerate(metadata_files, start=1):
        try:
            results.append(evaluate_frame(metadata_path, calculator))
        except Exception as exc:
            failures.append({"config_id": metadata_path.parent.name, "directory": str(metadata_path.parent), "reason": str(exc)})
        print(f"[{index}/{len(metadata_files)}] {metadata_path.parent.name}", flush=True)

    with (args.output / "failures.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("config_id", "directory", "reason"))
        writer.writeheader()
        writer.writerows(failures)
    if failures and not args.allow_partial:
        raise RuntimeError(f"{len(failures)} frames failed validation; see {args.output / 'failures.csv'}")
    if not results:
        raise RuntimeError("No valid parity frames were evaluated")

    write_frame_csv(results, args.output / "parity_frames.csv")
    rows = metric_rows(results)
    write_metrics_csv(rows, args.output / "parity_metrics.csv")
    outlier_summary, outlier_selections = write_outlier_csvs(
        results,
        args.output,
        top_n=args.outlier_top_n,
        energy_threshold_meV_per_atom=args.energy_outlier_threshold_meV_per_atom,
        force_threshold_eV_per_A=args.force_outlier_threshold_eV_per_A,
    )
    for system in ("tetragonal", "orthorhombic"):
        distance_rows = pair_distance_rows(
            outlier_selections,
            args.input,
            ROOT,
            system,
            args.pair_distance_cutoff_A,
        )
        plot_filename = write_pair_distance_comparison(distance_rows, system, args.output)
        outlier_summary[f"{system}_pair_distances"] = {
            "file": f"outlier_pair_distances_{system}.csv",
            "plot": plot_filename,
            "n_pairs": len(distance_rows),
            "reference_distance_cutoff_A": args.pair_distance_cutoff_A,
        }
    (args.output / "outlier_summary.json").write_text(
        json.dumps(outlier_summary, indent=2) + "\n"
    )
    for phase in sorted({item.phase for item in results}):
        phase_results = [item for item in results if item.phase == phase]
        plot_parity(phase_results, phase, args.output / f"parity_plot_{phase}.png")
    overall = [row for row in rows if row["scope"] == "all"]
    print(f"Evaluated {len(results)} structures; skipped {len(failures)}.")
    for row in overall:
        print(f"{row['target']}: MAE={row['mae']:.6g} {row['unit']}, RMSE={row['rmse']:.6g}, R2={row['r2']:.6g}")
    for phase in sorted({item.phase for item in results}):
        print(f"Plot: {args.output / f'parity_plot_{phase}.png'}")
    for label, details in outlier_summary.items():
        if "n_selected" in details:
            print(f"Outliers ({label}): {details['n_selected']} -> {args.output / details['file']}")
        else:
            print(f"Pair distances ({label}): {details['n_pairs']} -> {args.output / details['file']}")


if __name__ == "__main__":
    main()
