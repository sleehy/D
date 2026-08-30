#!/usr/bin/env python3
"""Evaluate a SevenNet checkpoint on the independent FAPbI3 parity set.

The script uses the final VASP frame in each candidate directory as the DFT
reference, then writes a combined energy/force/stress parity plot and
machine-readable per-frame and aggregate metrics.  It also writes the
structures furthest from the y=x line, split into tetragonal/orthorhombic and
energy/force/stress CSV files.  For the selected structures it compares the distorted
structure with the DFT-relaxed phase CONTCAR: unit-cell edge lengths and the
minimum H-Pb/H-I contacts.  It also writes every index-matched neighbouring
Pb-I pair whose distorted/relaxed distance ratio differs sufficiently from 1,
but only for the structures selected by the energy/force/stress outlier criteria.
Every reported structural ratio is ``distorted / relaxed``.
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
STRESS_OUTLIER_FIELDS = (
    "config_id", "phase", "crystal_system", "family", "severity", "natoms",
    "voigt_component", "reference_stress_GPa", "prediction_stress_GPa",
    "signed_residual_GPa", "absolute_residual_GPa", "perpendicular_distance_GPa",
    "structure_stress_mae_GPa", "structure_stress_rmse_GPa",
)
STRUCTURE_RATIO_FIELDS = (
    "config_id", "phase", "crystal_system", "family", "severity", "outlier_selection",
    "absolute_energy_residual_meV_per_atom", "max_force_component_residual_eV_per_A",
    "max_stress_component_residual_GPa",
    "reference_cell_a_A", "reference_cell_b_A", "reference_cell_c_A",
    "outlier_cell_a_A", "outlier_cell_b_A", "outlier_cell_c_A",
    "cell_a_ratio_outlier_over_relaxed", "cell_b_ratio_outlier_over_relaxed",
    "cell_c_ratio_outlier_over_relaxed",
    "min_h_pb_relaxed_A", "min_h_pb_outlier_A", "min_h_pb_ratio_outlier_over_relaxed",
    "min_h_i_relaxed_A", "min_h_i_outlier_A", "min_h_i_ratio_outlier_over_relaxed",
)
PB_I_RATIO_FIELDS = (
    "config_id", "phase", "crystal_system", "family", "severity", "outlier_selection",
    "absolute_energy_residual_meV_per_atom", "max_force_component_residual_eV_per_A",
    "max_stress_component_residual_GPa",
    "pb_atom_index_zero_based", "i_atom_index_zero_based",
    "relaxed_pb_i_distance_A", "outlier_pb_i_distance_A",
    "distance_ratio_outlier_over_relaxed", "absolute_ratio_deviation_from_1",
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
        default=PROJECT_DIR / "checkpoint_fine_tuned_al_round1.pth",
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
        default=PROJECT_DIR / "parity_results2",
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
            "Number of highest-residual entries to write per crystal-system/property "
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
            "Write every individual force component whose absolute residual from y=x "
            "is at least this value. Overrides --outlier-top-n for force."
        ),
    )
    parser.add_argument(
        "--stress-outlier-threshold-GPa",
        type=float,
        default=None,
        help=(
            "Write every individual Voigt stress component whose absolute residual "
            "from y=x is at least this value. Overrides --outlier-top-n for stress."
        ),
    )
    parser.add_argument(
        "--outlier-selection",
        choices=("any", "both", "all"),
        default="any",
        help=(
            "Select structures exceeding any energy/force/stress criterion (any), "
            "both energy and force criteria (both), or all three criteria (all; "
            "default: any). When thresholds are omitted, the corresponding top-N "
            "selection is used."
        ),
    )
    parser.add_argument(
        "--pb-i-reference-cutoff-A",
        type=float,
        default=5.0,
        help=(
            "Treat Pb-I pairs no farther apart than this in the relaxed CONTCAR "
            "as neighbouring pairs (default: 5.0 A)."
        ),
    )
    parser.add_argument(
        "--pb-i-ratio-deviation-threshold",
        type=float,
        default=0.05,
        help=(
            "Write an index-matched Pb-I pair only when "
            "abs(distorted/relaxed - 1) is at least this value (default: 0.05)."
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


def force_outlier_rows(item: FrameResult) -> list[dict[str, object]]:
    """Return one outlier-candidate row for every atom and Cartesian component."""
    residual = item.force_pred_eV_per_A - item.force_ref_eV_per_A
    absolute_residual = np.abs(residual)
    structure_force_mae = float(np.mean(absolute_residual))
    structure_force_rmse = float(np.sqrt(np.mean(residual**2)))
    rows: list[dict[str, object]] = []
    for atom_index, component_index in np.ndindex(residual.shape):
        component_residual = float(residual[atom_index, component_index])
        absolute_component_residual = float(absolute_residual[atom_index, component_index])
        rows.append({
            "config_id": item.config_id,
            "phase": item.phase,
            "crystal_system": crystal_system(item.phase),
            "family": item.family,
            "severity": item.severity,
            "natoms": item.natoms,
            "atom_index_zero_based": int(atom_index),
            "cartesian_component": ("x", "y", "z")[component_index],
            "reference_force_eV_per_A": float(item.force_ref_eV_per_A[atom_index, component_index]),
            "prediction_force_eV_per_A": float(item.force_pred_eV_per_A[atom_index, component_index]),
            "signed_residual_eV_per_A": component_residual,
            "absolute_residual_eV_per_A": absolute_component_residual,
            "perpendicular_distance_eV_per_A": absolute_component_residual / np.sqrt(2),
            "structure_force_mae_eV_per_A": structure_force_mae,
            "structure_force_rmse_eV_per_A": structure_force_rmse,
        })
    return rows


def stress_outlier_rows(item: FrameResult) -> list[dict[str, object]]:
    """Return one outlier-candidate row for each independent Voigt component."""
    residual = item.stress_pred_GPa - item.stress_ref_GPa
    absolute_residual = np.abs(residual)
    structure_stress_mae = float(np.mean(absolute_residual))
    structure_stress_rmse = float(np.sqrt(np.mean(residual**2)))
    component_names = ("xx", "yy", "zz", "yz", "xz", "xy")
    return [
        {
            "config_id": item.config_id,
            "phase": item.phase,
            "crystal_system": crystal_system(item.phase),
            "family": item.family,
            "severity": item.severity,
            "natoms": item.natoms,
            "voigt_component": component_names[component_index],
            "reference_stress_GPa": float(item.stress_ref_GPa[component_index]),
            "prediction_stress_GPa": float(item.stress_pred_GPa[component_index]),
            "signed_residual_GPa": float(residual[component_index]),
            "absolute_residual_GPa": float(absolute_residual[component_index]),
            "perpendicular_distance_GPa": float(absolute_residual[component_index] / np.sqrt(2)),
            "structure_stress_mae_GPa": structure_stress_mae,
            "structure_stress_rmse_GPa": structure_stress_rmse,
        }
        for component_index in range(6)
    ]


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
    stress_threshold_GPa: float | None,
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    """Write tetragonal/orthorhombic energy, force, and stress outlier tables."""
    if top_n < 1:
        raise ValueError("--outlier-top-n must be at least 1")
    if energy_threshold_meV_per_atom is not None and energy_threshold_meV_per_atom < 0:
        raise ValueError("--energy-outlier-threshold-meV-per-atom must be non-negative")
    if force_threshold_eV_per_A is not None and force_threshold_eV_per_A < 0:
        raise ValueError("--force-outlier-threshold-eV-per-A must be non-negative")
    if stress_threshold_GPa is not None and stress_threshold_GPa < 0:
        raise ValueError("--stress-outlier-threshold-GPa must be non-negative")

    specifications = (
        ("energy", "absolute_residual_meV_per_atom", energy_threshold_meV_per_atom, ENERGY_OUTLIER_FIELDS),
        ("force", "absolute_residual_eV_per_A", force_threshold_eV_per_A, FORCE_OUTLIER_FIELDS),
        ("stress", "absolute_residual_GPa", stress_threshold_GPa, STRESS_OUTLIER_FIELDS),
    )
    summary: dict[str, dict[str, object]] = {}
    selections: dict[str, list[dict[str, object]]] = {}
    for system in ("tetragonal", "orthorhombic"):
        phase_results = [item for item in results if crystal_system(item.phase) == system]
        for property_name, metric, threshold, fieldnames in specifications:
            candidate_rows = (
                [energy_outlier_row(item) for item in phase_results]
                if property_name == "energy"
                else (
                    [row for item in phase_results for row in force_outlier_rows(item)]
                    if property_name == "force"
                    else [row for item in phase_results for row in stress_outlier_rows(item)]
                )
            )
            selected = select_outliers(
                candidate_rows,
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


def selected_outlier_configs(
    selections: dict[str, list[dict[str, object]]], crystal_system_name: str,
    selection_mode: str,
) -> dict[str, dict[str, object]]:
    """Combine energy/force/stress selections into unique structures for geometry analysis."""
    if selection_mode not in {"any", "both", "all"}:
        raise ValueError(f"Unknown outlier selection mode: {selection_mode}")

    selected: dict[str, dict[str, object]] = {}
    for property_name in ("energy", "force", "stress"):
        for row in selections[f"{crystal_system_name}_{property_name}"]:
            config_id = str(row["config_id"])
            details = selected.setdefault(
                config_id,
                {
                    "selected_for": set(),
                    "phase": str(row["phase"]),
                    "family": str(row["family"]),
                    "severity": float(row["severity"]),
                    "energy_row": None,
                    "force_rows": [],
                    "stress_rows": [],
                },
            )
            if (
                details["phase"] != str(row["phase"])
                or details["family"] != str(row["family"])
                or details["severity"] != float(row["severity"])
            ):
                raise ValueError(f"Inconsistent outlier metadata for {config_id}")
            details["selected_for"].add(property_name)
            if property_name == "energy":
                details["energy_row"] = row
            elif property_name == "force":
                details["force_rows"].append(row)
            else:
                details["stress_rows"].append(row)

    if selection_mode == "both":
        return {
            config_id: details
            for config_id, details in selected.items()
            if {"energy", "force"}.issubset(details["selected_for"])
        }
    if selection_mode == "all":
        return {
            config_id: details
            for config_id, details in selected.items()
            if details["selected_for"] == {"energy", "force", "stress"}
        }
    return selected


def outlier_selection_label(selected_for: set[str]) -> str:
    """Use a stable label for the property criteria that selected a structure."""
    return "+".join(property_name for property_name in ("energy", "force", "stress") if property_name in selected_for)


def closest_pair(atoms, first_symbol: str, second_symbol: str) -> tuple[float, int, int]:
    """Return the closest minimum-image pair and its zero-based atom indices."""
    first_indices = [index for index, symbol in enumerate(atoms.symbols) if symbol == first_symbol]
    second_indices = [index for index, symbol in enumerate(atoms.symbols) if symbol == second_symbol]
    if not first_indices or not second_indices:
        raise ValueError(f"Structure does not contain both {first_symbol} and {second_symbol}")
    return min(
        (float(atoms.get_distance(i, j, mic=True)), i, j)
        for i in first_indices for j in second_indices
    )


def neighbouring_pb_i_pairs(atoms, cutoff_A: float) -> list[tuple[float, int, int]]:
    """Return every Pb-I pair within the relaxed-structure neighbour cutoff."""
    if cutoff_A <= 0:
        raise ValueError("--pb-i-reference-cutoff-A must be positive")
    symbols = atoms.get_chemical_symbols()
    return [
        (distance, pb_index, i_index)
        for pb_index, symbol in enumerate(symbols)
        if symbol == "Pb"
        for i_index, i_symbol in enumerate(symbols)
        if i_symbol == "I"
        for distance in (float(atoms.get_distance(pb_index, i_index, mic=True)),)
        if distance <= cutoff_A
    ]


def structure_ratio_rows(
    selections: dict[str, list[dict[str, object]]], input_dir: Path, reference_root: Path,
    crystal_system_name: str, selection_mode: str,
) -> list[dict[str, object]]:
    """Summarise distorted-to-relaxed structural ratios once per selected structure."""
    selected_configs = selected_outlier_configs(
        selections, crystal_system_name, selection_mode
    )
    if not selected_configs:
        return []

    phases = {str(details["phase"]) for details in selected_configs.values()}
    if len(phases) != 1:
        raise ValueError(f"Expected one phase for {crystal_system_name}, found {phases}")
    phase = phases.pop()
    contcar_path = reference_root / phase / "CONTCAR"
    if not contcar_path.is_file():
        raise FileNotFoundError(f"Phase reference CONTCAR not found: {contcar_path}")
    relaxed_atoms = read(contcar_path)
    relaxed_symbols = relaxed_atoms.get_chemical_symbols()
    relaxed_cell_edges = np.linalg.norm(relaxed_atoms.cell.array, axis=1)
    if np.any(relaxed_cell_edges <= 0):
        raise ValueError(f"Relaxed reference has a zero-length cell edge: {contcar_path}")
    relaxed_h_pb = closest_pair(relaxed_atoms, "H", "Pb")[0]
    relaxed_h_i = closest_pair(relaxed_atoms, "H", "I")[0]

    rows: list[dict[str, object]] = []
    for config_id, details in sorted(selected_configs.items()):
        outlier_path = input_dir / phase / config_id / "vasprun.xml"
        if not outlier_path.is_file():
            raise FileNotFoundError(f"Outlier structure vasprun.xml not found: {outlier_path}")
        outlier_atoms = read(outlier_path, index=-1)
        if outlier_atoms.get_chemical_symbols() != relaxed_symbols:
            raise ValueError(
                f"Atom order/species differ between {contcar_path} and {outlier_path}; "
                "cannot make index-matched Pb-I comparisons."
            )
        outlier_cell_edges = np.linalg.norm(outlier_atoms.cell.array, axis=1)
        outlier_h_pb = closest_pair(outlier_atoms, "H", "Pb")[0]
        outlier_h_i = closest_pair(outlier_atoms, "H", "I")[0]
        selected_for = details["selected_for"]
        selection_label = outlier_selection_label(selected_for)
        energy_row = details["energy_row"]
        force_rows = details["force_rows"]
        stress_rows = details["stress_rows"]
        rows.append({
            "config_id": config_id,
            "phase": phase,
            "crystal_system": crystal_system_name,
            "family": details["family"],
            "severity": details["severity"],
            "outlier_selection": selection_label,
            "absolute_energy_residual_meV_per_atom": (
                energy_row["absolute_residual_meV_per_atom"] if energy_row else ""
            ),
            "max_force_component_residual_eV_per_A": (
                max(float(row["absolute_residual_eV_per_A"]) for row in force_rows)
                if force_rows else ""
            ),
            "max_stress_component_residual_GPa": (
                max(float(row["absolute_residual_GPa"]) for row in stress_rows)
                if stress_rows else ""
            ),
            "reference_cell_a_A": relaxed_cell_edges[0],
            "reference_cell_b_A": relaxed_cell_edges[1],
            "reference_cell_c_A": relaxed_cell_edges[2],
            "outlier_cell_a_A": outlier_cell_edges[0],
            "outlier_cell_b_A": outlier_cell_edges[1],
            "outlier_cell_c_A": outlier_cell_edges[2],
            "cell_a_ratio_outlier_over_relaxed": outlier_cell_edges[0] / relaxed_cell_edges[0],
            "cell_b_ratio_outlier_over_relaxed": outlier_cell_edges[1] / relaxed_cell_edges[1],
            "cell_c_ratio_outlier_over_relaxed": outlier_cell_edges[2] / relaxed_cell_edges[2],
            "min_h_pb_relaxed_A": relaxed_h_pb,
            "min_h_pb_outlier_A": outlier_h_pb,
            "min_h_pb_ratio_outlier_over_relaxed": outlier_h_pb / relaxed_h_pb,
            "min_h_i_relaxed_A": relaxed_h_i,
            "min_h_i_outlier_A": outlier_h_i,
            "min_h_i_ratio_outlier_over_relaxed": outlier_h_i / relaxed_h_i,
        })
    return rows


def write_structure_ratio_comparison(
    rows: list[dict[str, object]], crystal_system_name: str, output_dir: Path,
) -> str:
    """Write one compact distorted/relaxed structural-ratio record per outlier."""
    filename = f"outlier_structure_ratios_{crystal_system_name}.csv"
    with (output_dir / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRUCTURE_RATIO_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return filename


def pb_i_ratio_outlier_rows(
    selections: dict[str, list[dict[str, object]]], input_dir: Path, reference_root: Path,
    crystal_system_name: str, selection_mode: str, cutoff_A: float,
    ratio_deviation_threshold: float,
) -> list[dict[str, object]]:
    """Return anomalous Pb-I ratios only from energy/force/stress-selected structures."""
    if ratio_deviation_threshold < 0:
        raise ValueError("--pb-i-ratio-deviation-threshold must be non-negative")
    selected_configs = selected_outlier_configs(
        selections, crystal_system_name, selection_mode
    )
    if not selected_configs:
        return []

    phases = {str(details["phase"]) for details in selected_configs.values()}
    if len(phases) != 1:
        raise ValueError(f"Expected one phase for {crystal_system_name}, found {phases}")
    phase = phases.pop()
    contcar_path = reference_root / phase / "CONTCAR"
    if not contcar_path.is_file():
        raise FileNotFoundError(f"Phase reference CONTCAR not found: {contcar_path}")
    relaxed_atoms = read(contcar_path)
    relaxed_symbols = relaxed_atoms.get_chemical_symbols()
    relaxed_pairs = neighbouring_pb_i_pairs(relaxed_atoms, cutoff_A)
    if not relaxed_pairs:
        raise ValueError(f"No neighbouring Pb-I pairs found in {contcar_path}")
    tolerance = max(1.0e-12, ratio_deviation_threshold * 1.0e-12)

    rows: list[dict[str, object]] = []
    for config_id, details in sorted(selected_configs.items()):
        outlier_path = input_dir / phase / config_id / "vasprun.xml"
        if not outlier_path.is_file():
            raise FileNotFoundError(f"Outlier structure vasprun.xml not found: {outlier_path}")
        outlier_atoms = read(outlier_path, index=-1)
        if outlier_atoms.get_chemical_symbols() != relaxed_symbols:
            raise ValueError(
                f"Atom order/species differ between {contcar_path} and {outlier_path}; "
                "cannot make index-matched Pb-I comparisons."
            )
        selected_for = details["selected_for"]
        selection_label = outlier_selection_label(selected_for)
        energy_row = details["energy_row"]
        force_rows = details["force_rows"]
        stress_rows = details["stress_rows"]
        for relaxed_distance, pb_index, i_index in relaxed_pairs:
            outlier_distance = float(outlier_atoms.get_distance(pb_index, i_index, mic=True))
            ratio = outlier_distance / relaxed_distance
            deviation = abs(ratio - 1.0)
            if deviation < ratio_deviation_threshold - tolerance:
                continue
            rows.append({
                "config_id": config_id,
                "phase": phase,
                "crystal_system": crystal_system_name,
                "family": details["family"],
                "severity": details["severity"],
                "outlier_selection": selection_label,
                "absolute_energy_residual_meV_per_atom": (
                    energy_row["absolute_residual_meV_per_atom"] if energy_row else ""
                ),
                "max_force_component_residual_eV_per_A": (
                    max(float(row["absolute_residual_eV_per_A"]) for row in force_rows)
                    if force_rows else ""
                ),
                "max_stress_component_residual_GPa": (
                    max(float(row["absolute_residual_GPa"]) for row in stress_rows)
                    if stress_rows else ""
                ),
                "pb_atom_index_zero_based": pb_index,
                "i_atom_index_zero_based": i_index,
                "relaxed_pb_i_distance_A": relaxed_distance,
                "outlier_pb_i_distance_A": outlier_distance,
                "distance_ratio_outlier_over_relaxed": ratio,
                "absolute_ratio_deviation_from_1": deviation,
            })
    return rows


def write_pb_i_ratio_outliers(
    rows: list[dict[str, object]], crystal_system_name: str, output_dir: Path,
) -> str:
    """Write Pb-I ratio deviations from structures selected by any parity error."""
    filename = f"property_outlier_pb_i_ratios_{crystal_system_name}.csv"
    with (output_dir / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PB_I_RATIO_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return filename


def prepare_matplotlib(output_dir: Path):
    """Import a non-interactive Matplotlib backend with a writable cache directory."""
    plot_cache = output_dir / ".matplotlib"
    plot_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def draw_parity_axis(axis, *, phase: str, title: str, unit: str,
                     reference: np.ndarray, prediction: np.ndarray) -> None:
    """Draw one component-resolved DFT-versus-MLP parity panel."""
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
    axis.set(
        xlim=(lower, upper), ylim=(lower, upper), title=f"{phase}: {title}",
        xlabel=f"DFT {title.lower()} ({unit})", ylabel=f"MLP {title.lower()} ({unit})",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small", loc="lower right")


def plot_parity(results: list[FrameResult], phase: str, path: Path) -> None:
    """Write energy plus x/y/z-resolved force parity panels."""
    plt = prepare_matplotlib(path.parent)
    figure, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    draw_parity_axis(
        axes[0, 0], phase=phase, title="Energy / atom", unit="eV/atom",
        reference=np.array([item.energy_ref_eV_per_atom for item in results]),
        prediction=np.array([item.energy_pred_eV_per_atom for item in results]),
    )
    for component_index, component_name in enumerate(("x", "y", "z")):
        draw_parity_axis(
            axes.flat[component_index + 1], phase=phase,
            title=f"Force {component_name}", unit="eV/A",
            reference=np.concatenate([
                item.force_ref_eV_per_A[:, component_index] for item in results
            ]),
            prediction=np.concatenate([
                item.force_pred_eV_per_A[:, component_index] for item in results
            ]),
        )
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_stress_parity(results: list[FrameResult], phase: str, path: Path) -> None:
    """Write six Voigt-component-resolved stress parity panels."""
    plt = prepare_matplotlib(path.parent)
    figure, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for component_index, component_name in enumerate(("xx", "yy", "zz", "yz", "xz", "xy")):
        draw_parity_axis(
            axes.flat[component_index], phase=phase,
            title=f"Stress {component_name}", unit="GPa",
            reference=np.array([item.stress_ref_GPa[component_index] for item in results]),
            prediction=np.array([item.stress_pred_GPa[component_index] for item in results]),
        )
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
        stress_threshold_GPa=args.stress_outlier_threshold_GPa,
    )
    for system in ("tetragonal", "orthorhombic"):
        ratio_rows = structure_ratio_rows(
            outlier_selections,
            args.input,
            ROOT,
            system,
            args.outlier_selection,
        )
        ratio_filename = write_structure_ratio_comparison(ratio_rows, system, args.output)
        pb_i_rows = pb_i_ratio_outlier_rows(
            outlier_selections,
            args.input,
            ROOT,
            system,
            args.outlier_selection,
            args.pb_i_reference_cutoff_A,
            args.pb_i_ratio_deviation_threshold,
        )
        pb_i_ratio_filename = write_pb_i_ratio_outliers(
            pb_i_rows, system, args.output
        )
        outlier_summary[f"{system}_structure_ratios"] = {
            "file": ratio_filename,
            "n_selected_structures": len(ratio_rows),
            "ratio_definition": "distorted_structure / DFT_relaxed_CONTCAR",
            "outlier_selection": args.outlier_selection,
        }
        outlier_summary[f"{system}_property_outlier_pb_i_ratios"] = {
            "file": pb_i_ratio_filename,
            "n_pb_i_pairs_written": len(pb_i_rows),
            "ratio_definition": "distorted_structure / DFT_relaxed_CONTCAR",
            "reference_neighbour_cutoff_A": args.pb_i_reference_cutoff_A,
            "absolute_ratio_deviation_threshold": args.pb_i_ratio_deviation_threshold,
            "outlier_selection": args.outlier_selection,
        }
    (args.output / "outlier_summary.json").write_text(
        json.dumps(outlier_summary, indent=2) + "\n"
    )
    for phase in sorted({item.phase for item in results}):
        phase_results = [item for item in results if item.phase == phase]
        plot_parity(phase_results, phase, args.output / f"parity_plot_{phase}.png")
        plot_stress_parity(
            phase_results, phase, args.output / f"stress_parity_plot_{phase}.png"
        )
    overall = [row for row in rows if row["scope"] == "all"]
    print(f"Evaluated {len(results)} structures; skipped {len(failures)}.")
    for row in overall:
        print(f"{row['target']}: MAE={row['mae']:.6g} {row['unit']}, RMSE={row['rmse']:.6g}, R2={row['r2']:.6g}")
    for phase in sorted({item.phase for item in results}):
        print(f"Plot: {args.output / f'parity_plot_{phase}.png'}")
        print(f"Stress plot: {args.output / f'stress_parity_plot_{phase}.png'}")
    for label, details in outlier_summary.items():
        if "n_selected" in details:
            print(f"Outliers ({label}): {details['n_selected']} -> {args.output / details['file']}")
        elif "n_pb_i_pairs_written" in details:
            print(
                f"Pb-I ratios from energy/force/stress outliers ({label}): "
                f"{details['n_pb_i_pairs_written']} "
                f"-> {args.output / details['file']}"
            )
        else:
            print(
                f"Structural ratios ({label}): {details['n_selected_structures']} "
                f"-> {args.output / details['file']}"
            )


if __name__ == "__main__":
    main()
