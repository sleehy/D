#!/usr/bin/env python3
"""Generate diverse FAPbI3 structures for DFT single-point labeling.

The generator identifies each FA cation as the two nearest N atoms and five
nearest H atoms around every C atom using minimum-image distances.  FA cations
are unwrapped before rigid rotations, so molecules crossing a periodic boundary
are handled correctly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASES = {
    "O-FAPI3": ROOT / "O-FAPI3" / "CONTCAR",
    "t-FAPI3": ROOT / "t-FAPI3" / "CONTCAR",
}
SUPPORTED_MODES = (
    "small_rotation",
    "wide_rotation",
    "cage_rattle",
    "all_rattle",
    "strain_rattle",
    "mixed",
)

# Target number of structures PER PHASE.  With --extend, existing structures
# are preserved and only the missing indices are generated.
MODE_COUNTS = {
    "small_rotation": 18,
    "wide_rotation": 14,
    "cage_rattle": 17,
    "all_rattle": 17,
    "strain_rattle": 17,
    "mixed": 17,
}

# These ranges deliberately allow thermal/distortion broadening while rejecting
# broken or implausibly compressed bonds in the formamidinium (FA) cation.
# They are applied only to the bonded C--N, C--H, and N--H pairs identified in
# the relaxed source structure; nonbonded contacts remain the responsibility
# of ``collision_check`` below.
FA_BOND_LIMITS_A = {
    "C-N": (1.00, 1.65),
    "C-H": (0.80, 1.30),
    "N-H": (0.75, 1.35),
}
PB_I_BOND_LIMIT_A = (2.65, 3.85)
PB_I_NEIGHBOR_CUTOFF_A = 4.00

STATIC_INCAR = """\
SYSTEM = FAPbI3 MLP dataset single point
PREC   = Accurate
ENCUT  = 520
EDIFF  = 1E-7
NELM   = 200

IBRION = -1
NSW    = 0
ISIF   = 2
ISYM   = 0

ISTART = 0
ICHARG = 2
ISMEAR = 0
SIGMA  = 0.02
LREAL  = .FALSE.

IVDW   = 11
GGA    = PE

LWAVE  = .FALSE.
LCHARG = .FALSE.

KPAR  = 2
NCORE = 1
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "mlp_dataset_candidates",
        help="Candidate dataset directory.",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--extend",
        action="store_true",
        help=(
            "Preserve existing configurations and generate only the structures "
            "needed to reach MODE_COUNTS."
        ),
    )
    parser.add_argument(
        "--no-vasp-inputs",
        action="store_true",
        help="Write structures only, without INCAR/KPOINTS/POTCAR links.",
    )
    parser.add_argument(
        "--min-distance-scale",
        type=float,
        default=1.0,
        help="Scale all collision-filter thresholds (default: 1.0).",
    )
    return parser.parse_args()


def nearest_indices(
    atoms: Atoms, center: int, symbol: str, count: int
) -> tuple[list[int], list[float]]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    candidates = np.flatnonzero(symbols == symbol)
    distances = np.asarray(
        [atoms.get_distance(center, int(i), mic=True) for i in candidates]
    )
    order = np.argsort(distances)[:count]
    return candidates[order].astype(int).tolist(), distances[order].tolist()


def identify_fa_cations(atoms: Atoms) -> list[list[int]]:
    """Return [C, N, N, H, H, H, H, H] groups for all FA cations."""
    symbols = np.asarray(atoms.get_chemical_symbols())
    carbon_indices = np.flatnonzero(symbols == "C")
    if not len(carbon_indices):
        raise ValueError("No carbon atoms found; cannot identify FA cations.")

    groups: list[list[int]] = []
    used_n: set[int] = set()
    used_h: set[int] = set()
    for carbon in carbon_indices:
        nitrogens, n_dist = nearest_indices(atoms, int(carbon), "N", 2)
        hydrogens, h_dist = nearest_indices(atoms, int(carbon), "H", 5)
        if max(n_dist) > 1.8 or max(h_dist) > 2.5:
            raise ValueError(
                f"FA around C index {carbon} failed distance checks: "
                f"C-N={n_dist}, C-H={h_dist}"
            )
        if used_n.intersection(nitrogens) or used_h.intersection(hydrogens):
            raise ValueError("Automatic FA groups overlap; inspect the input structure.")
        used_n.update(nitrogens)
        used_h.update(hydrogens)
        groups.append([int(carbon), *nitrogens, *hydrogens])

    expected = {"C": len(groups), "N": 2 * len(groups), "H": 5 * len(groups)}
    for symbol, count in expected.items():
        actual = int(np.count_nonzero(symbols == symbol))
        if actual != count:
            raise ValueError(f"Expected {count} {symbol} atoms, found {actual}.")
    return groups


def identify_fa_bonds(
    atoms: Atoms, fa_groups: list[list[int]]
) -> list[tuple[int, int, str]]:
    """Return the C--N, C--H, and N--H bonds of the source FA cations.

    The returned atom-index topology is subsequently kept fixed while a
    candidate is distorted. Re-identifying nearest atoms after a global rattle
    could silently accept a proton that has moved to the wrong N atom.
    """
    bonds: list[tuple[int, int, str]] = []
    for group in fa_groups:
        carbon, *rest = group
        nitrogens = rest[:2]
        hydrogens = rest[2:]

        bonds.extend((carbon, nitrogen, "C-N") for nitrogen in nitrogens)
        carbon_hydrogen = min(
            hydrogens, key=lambda hydrogen: atoms.get_distance(carbon, hydrogen, mic=True)
        )
        bonds.append((carbon, carbon_hydrogen, "C-H"))

        remaining_hydrogens = [
            hydrogen for hydrogen in hydrogens if hydrogen != carbon_hydrogen
        ]
        assigned_hydrogens: set[int] = set()
        for nitrogen in nitrogens:
            nearest = sorted(
                remaining_hydrogens,
                key=lambda hydrogen: atoms.get_distance(nitrogen, hydrogen, mic=True),
            )[:2]
            if len(nearest) != 2 or assigned_hydrogens.intersection(nearest):
                raise ValueError("Could not assign two unique N-H bonds in an FA cation.")
            assigned_hydrogens.update(nearest)
            bonds.extend((nitrogen, hydrogen, "N-H") for hydrogen in nearest)

        if len(assigned_hydrogens) != len(remaining_hydrogens):
            raise ValueError("FA cation does not contain the expected four N-H bonds.")
    return bonds


def fa_geometry_check(
    atoms: Atoms, fa_bonds: list[tuple[int, int, str]]
) -> tuple[bool, dict[str, Any]]:
    """Reject candidates whose source FA bond topology has become unphysical."""
    distances: list[float] = []
    for first, second, bond_type in fa_bonds:
        distance = float(atoms.get_distance(first, second, mic=True))
        lower, upper = FA_BOND_LIMITS_A[bond_type]
        if not lower <= distance <= upper:
            return False, {
                "reason": "fa_bond_out_of_range",
                "bond_type": bond_type,
                "indices": [first, second],
                "symbols": [atoms[first].symbol, atoms[second].symbol],
                "distance_A": distance,
                "allowed_range_A": [lower, upper],
            }
        distances.append(distance)
    return True, {
        "fa_bond_distance_min_A": min(distances),
        "fa_bond_distance_max_A": max(distances),
    }


def identify_pb_i_bonds(atoms: Atoms) -> list[tuple[int, int, np.ndarray]]:
    """Return the six periodic Pb--I nearest-neighbor bonds for every Pb."""
    centers, neighbors, offsets = neighbor_list(
        "ijS", atoms, PB_I_NEIGHBOR_CUTOFF_A
    )
    bonds: list[tuple[int, int, np.ndarray]] = []
    for lead in (atom.index for atom in atoms if atom.symbol == "Pb"):
        candidates = []
        for center, neighbor, offset in zip(centers, neighbors, offsets):
            if center != lead or atoms[neighbor].symbol != "I":
                continue
            vector = atoms.positions[neighbor] + offset @ atoms.cell.array - atoms.positions[lead]
            candidates.append((float(np.linalg.norm(vector)), int(neighbor), offset.copy()))
        nearest = sorted(candidates, key=lambda item: item[0])[:6]
        if len(nearest) != 6:
            raise ValueError(f"Pb index {lead} does not have six I neighbors within 4.0 A.")
        bonds.extend((lead, iodine, offset) for _, iodine, offset in nearest)
    return bonds


def pb_i_geometry_check(atoms: Atoms) -> tuple[bool, dict[str, Any]]:
    """Reject candidates with compressed or broken Pb--I cage bonds."""
    lower, upper = PB_I_BOND_LIMIT_A
    distances: list[float] = []
    # Rebuild the periodic neighbor images after every perturbation. A wrapped
    # I coordinate can represent a different cell image from the source while
    # remaining the same physical nearest-neighbor Pb--I bond.
    try:
        pb_i_bonds = identify_pb_i_bonds(atoms)
    except ValueError as exc:
        return False, {"reason": "pb_i_coordination_invalid", "detail": str(exc)}
    for lead, iodine, offset in pb_i_bonds:
        vector = atoms.positions[iodine] + offset @ atoms.cell.array - atoms.positions[lead]
        distance = float(np.linalg.norm(vector))
        if not lower <= distance <= upper:
            return False, {
                "reason": "pb_i_bond_out_of_range",
                "bond_type": "Pb-I",
                "indices": [lead, iodine],
                "symbols": ["Pb", "I"],
                "distance_A": distance,
                "allowed_range_A": [lower, upper],
            }
        distances.append(distance)
    return True, {
        "pb_i_bond_distance_min_A": min(distances),
        "pb_i_bond_distance_max_A": max(distances),
    }


def geometry_check(
    atoms: Atoms,
    fa_bonds: list[tuple[int, int, str]],
) -> tuple[bool, dict[str, Any]]:
    """Apply the FA molecular and Pb--I cage geometry validity checks."""
    valid, fa_report = fa_geometry_check(atoms, fa_bonds)
    if not valid:
        return False, fa_report
    valid, pb_i_report = pb_i_geometry_check(atoms)
    if not valid:
        return False, pb_i_report
    return True, {**fa_report, **pb_i_report}


def axis_angle_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def uniform_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sample a rotation uniformly from SO(3) using a unit quaternion."""
    u1, u2, u3 = rng.random(3)
    qx = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    qy = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def rotate_fa_group(atoms: Atoms, group: list[int], rotation: np.ndarray) -> None:
    """Rigidly rotate one PBC-unwrapped FA cation about its center of mass."""
    anchor = group[0]
    vectors = atoms.get_distances(anchor, group, mic=True, vector=True)
    masses = atoms.get_masses()[group]
    com_vector = np.average(vectors, axis=0, weights=masses)
    centered = vectors - com_vector
    new_positions = atoms.positions[anchor] + com_vector + centered @ rotation.T
    atoms.positions[group] = new_positions


def apply_fa_rotations(
    atoms: Atoms,
    groups: list[list[int]],
    rng: np.random.Generator,
    rotation_kind: str,
) -> dict[str, Any]:
    number = int(rng.integers(1, len(groups) + 1))
    selected = sorted(rng.choice(len(groups), size=number, replace=False).tolist())
    records: list[dict[str, Any]] = []

    for group_number in selected:
        if rotation_kind == "small":
            angle = float(rng.uniform(-15.0, 15.0))
            axis = rng.normal(size=3)
            rotation = axis_angle_matrix(axis, angle)
            record = {
                "fa_group": group_number,
                "kind": "axis_angle",
                "angle_deg": round(angle, 6),
                "axis": np.round(axis / np.linalg.norm(axis), 8).tolist(),
            }
        elif rotation_kind == "medium":
            angle = float(rng.uniform(-45.0, 45.0))
            axis = rng.normal(size=3)
            rotation = axis_angle_matrix(axis, angle)
            record = {
                "fa_group": group_number,
                "kind": "axis_angle",
                "angle_deg": round(angle, 6),
                "axis": np.round(axis / np.linalg.norm(axis), 8).tolist(),
            }
        elif rotation_kind == "uniform":
            rotation = uniform_rotation_matrix(rng)
            record = {
                "fa_group": group_number,
                "kind": "uniform_SO3",
                "matrix": np.round(rotation, 8).tolist(),
            }
        else:
            raise ValueError(f"Unknown rotation kind: {rotation_kind}")

        rotate_fa_group(atoms, groups[group_number], rotation)
        records.append(record)

    return {"rotations": records}


def rattle_indices(
    atoms: Atoms,
    indices: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> None:
    displacement = rng.normal(scale=sigma, size=(len(indices), 3))
    atoms.positions[indices] += displacement


def random_strain(
    atoms: Atoms,
    rng: np.random.Generator,
    diagonal_limit: float,
    shear_limit: float,
) -> np.ndarray:
    strain = np.zeros((3, 3))
    strain[np.diag_indices(3)] = rng.uniform(
        -diagonal_limit, diagonal_limit, size=3
    )
    upper = rng.uniform(-shear_limit, shear_limit, size=3)
    strain[0, 1] = strain[1, 0] = upper[0]
    strain[0, 2] = strain[2, 0] = upper[1]
    strain[1, 2] = strain[2, 1] = upper[2]
    deformation = np.eye(3) + strain
    atoms.set_cell(atoms.cell.array @ deformation.T, scale_atoms=True)
    return strain


def perturb(
    source: Atoms,
    fa_groups: list[list[int]],
    mode: str,
    rng: np.random.Generator,
) -> tuple[Atoms, dict[str, Any]]:
    atoms = source.copy()
    symbols = np.asarray(atoms.get_chemical_symbols())
    cage = np.flatnonzero(np.isin(symbols, ["Pb", "I"]))
    all_indices = np.arange(len(atoms))
    details: dict[str, Any] = {}

    if mode == "small_rotation":
        details.update(apply_fa_rotations(atoms, fa_groups, rng, "small"))
    elif mode == "wide_rotation":
        details.update(apply_fa_rotations(atoms, fa_groups, rng, "uniform"))
    elif mode == "cage_rattle":
        sigma = float(rng.uniform(0.015, 0.050))
        rattle_indices(atoms, cage, sigma, rng)
        details.update({"rattle_target": "Pb_I", "rattle_sigma_A": sigma})
    elif mode == "all_rattle":
        sigma = float(rng.uniform(0.020, 0.070))
        rattle_indices(atoms, all_indices, sigma, rng)
        details.update({"rattle_target": "all", "rattle_sigma_A": sigma})
    elif mode == "strain_rattle":
        strain = random_strain(atoms, rng, diagonal_limit=0.020, shear_limit=0.010)
        sigma = float(rng.uniform(0.010, 0.035))
        rattle_indices(atoms, cage, sigma, rng)
        details.update(
            {
                "strain": np.round(strain, 8).tolist(),
                "rattle_target": "Pb_I",
                "rattle_sigma_A": sigma,
            }
        )
    elif mode == "mixed":
        strain = random_strain(atoms, rng, diagonal_limit=0.015, shear_limit=0.0075)
        details.update(apply_fa_rotations(atoms, fa_groups, rng, "medium"))
        sigma = float(rng.uniform(0.015, 0.050))
        rattle_indices(atoms, all_indices, sigma, rng)
        details.update(
            {
                "strain": np.round(strain, 8).tolist(),
                "rattle_target": "all",
                "rattle_sigma_A": sigma,
            }
        )
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")

    atoms.wrap()
    return atoms, details


PAIR_MINIMUMS = {
    frozenset(("H", "I")): 1.50,
    frozenset(("H", "Pb")): 1.50,
    frozenset(("C", "I")): 1.80,
    frozenset(("N", "I")): 1.80,
    frozenset(("C", "Pb")): 1.80,
    frozenset(("N", "Pb")): 1.80,
    frozenset(("I",)): 2.20,
    frozenset(("Pb", "I")): 2.30,
    frozenset(("Pb",)): 3.00,
}


def collision_check(
    atoms: Atoms, fa_groups: list[list[int]], scale: float
) -> tuple[bool, dict[str, Any]]:
    """Reject severe nonbonded contacts while allowing covalent FA bonds."""
    symbols = atoms.get_chemical_symbols()
    distances = atoms.get_all_distances(mic=True)
    same_fa = {
        frozenset((i, j))
        for group in fa_groups
        for offset, i in enumerate(group)
        for j in group[offset + 1 :]
    }
    closest = (math.inf, -1, -1)

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            distance = float(distances[i, j])
            if distance < closest[0]:
                closest = (distance, i, j)
            if frozenset((i, j)) in same_fa:
                threshold = 0.65
            else:
                threshold = PAIR_MINIMUMS.get(
                    frozenset((symbols[i], symbols[j])), 0.80
                )
            if distance < threshold * scale:
                return False, {
                    "reason": "short_contact",
                    "indices": [i, j],
                    "symbols": [symbols[i], symbols[j]],
                    "distance_A": distance,
                    "threshold_A": threshold * scale,
                }

    return True, {
        "minimum_distance_A": closest[0],
        "minimum_pair_indices": [closest[1], closest[2]],
        "minimum_pair_symbols": [symbols[closest[1]], symbols[closest[2]]],
    }


def prepare_vasp_inputs(config_dir: Path, phase: str) -> None:
    (config_dir / "INCAR").write_text(STATIC_INCAR)
    source_kpoints = ROOT / phase / "KPOINTS"
    (config_dir / "KPOINTS").write_text(source_kpoints.read_text())

    source_potcar = ROOT / "POTCAR"
    target = config_dir / "POTCAR"
    relative_target = os.path.relpath(source_potcar, start=config_dir)
    target.symlink_to(relative_target)


def output_is_available(output: Path) -> bool:
    return not output.exists() or not any(output.iterdir())


def extension_sample_seed(
    master_seed: int, phase: str, mode: str, index: int, attempt: int
) -> int:
    """Return a stable seed so interrupted extensions can be resumed exactly."""
    key = f"{master_seed}|extension-v1|{phase}|{mode}|{index}|{attempt}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], "little")


def existing_configurations(output: Path) -> dict[tuple[str, str, int], Path]:
    configurations: dict[tuple[str, str, int], Path] = {}
    for metadata_path in output.glob("*/*/metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        config_id = metadata.get("config_id", metadata_path.parent.name)
        phase = metadata.get("phase")
        mode = metadata.get("mode")
        if phase not in DEFAULT_PHASES or mode not in MODE_COUNTS:
            raise ValueError(f"Unsupported existing configuration: {config_id}")
        prefix = f"{phase}_{mode}_"
        suffix = config_id.removeprefix(prefix)
        if not config_id.startswith(prefix) or not suffix.isdigit():
            raise ValueError(f"Inconsistent metadata in {metadata_path}")
        index = int(suffix)
        key = (phase, mode, index)
        if key in configurations:
            raise ValueError(f"Duplicate existing configuration: {config_id}")
        configurations[key] = metadata_path.parent

    for phase in DEFAULT_PHASES:
        for mode, target_count in MODE_COUNTS.items():
            indices = sorted(
                index
                for existing_phase, existing_mode, index in configurations
                if existing_phase == phase and existing_mode == mode
            )
            if indices and indices != list(range(1, indices[-1] + 1)):
                raise ValueError(f"Existing indices are not contiguous for {phase}/{mode}")
            if indices and indices[-1] > target_count:
                raise ValueError(
                    f"{phase}/{mode} already has {indices[-1]} structures, "
                    f"above target {target_count}; refusing to remove any."
                )
    return configurations


def append_summary(
    atoms: Atoms,
    metadata: dict[str, Any],
    config_dir: Path,
    output: Path,
    all_structures: list[Atoms],
    rows: list[dict[str, Any]],
) -> None:
    extxyz_atoms = atoms.copy()
    extxyz_atoms.info.update(
        {
            "config_id": metadata["config_id"],
            "phase": metadata["phase"],
            "mode": metadata["mode"],
            "seed": metadata["seed"],
            "source": metadata["source"],
        }
    )
    all_structures.append(extxyz_atoms)
    rows.append(
        {
            "config_id": metadata["config_id"],
            "phase": metadata["phase"],
            "mode": metadata["mode"],
            "seed": metadata["seed"],
            "source": metadata["source"],
            "natoms": len(atoms),
            "rattle_sigma_A": metadata.get("rattle_sigma_A", ""),
            "n_rotated_fa": len(metadata.get("rotations", [])),
            "minimum_distance_A": metadata["minimum_distance_A"],
            "directory": str(config_dir.relative_to(output)),
        }
    )


def generate(args: argparse.Namespace) -> None:
    if not MODE_COUNTS:
        raise ValueError("MODE_COUNTS must contain at least one perturbation mode.")
    unknown_modes = set(MODE_COUNTS).difference(SUPPORTED_MODES)
    if unknown_modes:
        raise ValueError(f"Unknown modes in MODE_COUNTS: {sorted(unknown_modes)}")
    invalid_counts = {
        mode: count
        for mode, count in MODE_COUNTS.items()
        if not isinstance(count, int) or isinstance(count, bool) or count < 0
    }
    if invalid_counts:
        raise ValueError(
            f"MODE_COUNTS values must be non-negative integers: {invalid_counts}"
        )
    if sum(MODE_COUNTS.values()) == 0:
        raise ValueError("At least one MODE_COUNTS value must be greater than zero.")
    if not args.extend and not output_is_available(args.output):
        raise FileExistsError(
            f"{args.output} is not empty. Choose a new --output directory or "
            "use --extend to preserve and expand it."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(args.seed)
    existing = existing_configurations(args.output) if args.extend else {}
    existing_total = len(existing)
    all_structures: list[Atoms] = []
    rows: list[dict[str, Any]] = []

    for phase, source_path in DEFAULT_PHASES.items():
        source = read(source_path)
        source.pbc = True
        fa_groups = identify_fa_cations(source)
        fa_bonds = identify_fa_bonds(source, fa_groups)
        phase_dir = args.output / phase
        phase_dir.mkdir(exist_ok=args.extend)

        for mode, requested_count in MODE_COUNTS.items():
            for config_index in range(1, requested_count + 1):
                existing_dir = existing.get((phase, mode, config_index))
                if existing_dir is not None:
                    metadata = json.loads((existing_dir / "metadata.json").read_text())
                    poscar = existing_dir / "POSCAR"
                    if not poscar.is_file():
                        raise FileNotFoundError(
                            f"Existing configuration has no POSCAR: {existing_dir}"
                        )
                    append_summary(
                        read(poscar), metadata, existing_dir, args.output,
                        all_structures, rows
                    )
                    continue

                for attempt in range(1, 501):
                    if args.extend:
                        sample_seed = extension_sample_seed(
                            args.seed, phase, mode, config_index, attempt
                        )
                    else:
                        sample_seed = int(
                            master_rng.integers(
                                0, np.iinfo(np.uint32).max, dtype=np.uint32
                            )
                        )
                    sample_rng = np.random.default_rng(sample_seed)
                    atoms, details = perturb(source, fa_groups, mode, sample_rng)
                    valid, contact = collision_check(
                        atoms, fa_groups, args.min_distance_scale
                    )
                    if valid:
                        valid, geometry = geometry_check(atoms, fa_bonds)
                    if valid:
                        break
                else:
                    raise RuntimeError(
                        f"Could not generate enough valid {phase}/{mode} structures."
                    )

                config_id = f"{phase}_{mode}_{config_index:02d}"
                config_dir = phase_dir / config_id
                if config_dir.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite unrecognized directory: {config_dir}"
                    )
                config_dir.mkdir()
                write(
                    config_dir / "POSCAR",
                    atoms,
                    format="vasp",
                    direct=True,
                    sort=False,
                    vasp5=True,
                )
                if not args.no_vasp_inputs:
                    prepare_vasp_inputs(config_dir, phase)

                metadata = {
                    "config_id": config_id,
                    "phase": phase,
                    "mode": mode,
                    "seed": sample_seed,
                    "source": str(source_path.relative_to(ROOT)),
                    "formula": atoms.get_chemical_formula(),
                    "natoms": len(atoms),
                    "fa_groups": fa_groups,
                    "cell_A": np.round(atoms.cell.array, 10).tolist(),
                    **details,
                    **contact,
                    **geometry,
                }
                (config_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )

                append_summary(
                    atoms, metadata, config_dir, args.output, all_structures, rows
                )

    write(args.output / "structures.extxyz", all_structures, format="extxyz")
    with (args.output / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "generator": str(Path(__file__).relative_to(ROOT)),
        "seed": args.seed,
        "total_structures": len(rows),
        "structures_per_phase": sum(MODE_COUNTS.values()),
        "mode_counts_per_phase": MODE_COUNTS,
        "extended_existing_structures": existing_total,
        "new_structures_added": len(rows) - existing_total,
        "phases": {key: str(value.relative_to(ROOT)) for key, value in DEFAULT_PHASES.items()},
        "vasp_inputs_created": not args.no_vasp_inputs,
        "labels_present": False,
        "note": "Run DFT single-point calculations to add energy, forces, and stress labels.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Generated {len(rows)} candidate structures in {args.output}")
    for phase in DEFAULT_PHASES:
        count = sum(row["phase"] == phase for row in rows)
        print(f"  {phase}: {count}")
    print("These are unlabeled candidates; DFT single-point calculations are required.")


if __name__ == "__main__":
    generate(parse_args())
