#!/usr/bin/env python3
"""Generate a deliberately distorted, DFT-labeled parity-test set for FAPbI3.

The configurations are independent of the fine-tuning dataset.  Each phase
gets 25 structures: five each with large cell strain, global rattles, inorganic
cage rattles, FA rotations, and combined perturbations.  The collision filter
keeps unphysical atom overlaps out while retaining configurations sufficiently
far from the relaxed CONTCAR to make an informative parity plot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write

from generate_fapbi3_dataset import (
    DEFAULT_PHASES,
    ROOT,
    apply_fa_rotations,
    collision_check,
    identify_fa_cations,
    prepare_vasp_inputs,
    random_strain,
    rattle_indices,
)


# Ten configurations in each family give 50 structures per phase.  The
# boundaries deliberately overlap a little so parity errors can be examined as
# a function of distortion severity rather than at five isolated points.
FAMILIES = (
    "strong_strain",
    "global_rattle",
    "cage_rattle",
    "fa_rotation_cage_rattle",
    "combined",
)
PER_FAMILY = 10
FAMILY_LABELS = {
    "strong_strain": "strong_strain",
    "global_rattle": "global_rattle",
    "cage_rattle": "cage_rattle",
    "fa_rotation_cage_rattle": "fa_rotation_cage_rattle",
    "combined": "combined_strain_rattle_fa_rotation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "parity_dataset_candidates",
        help="Empty output directory (default: ROOT/parity_dataset_candidates).",
    )
    parser.add_argument(
        "--seed", type=int, default=20260810,
        help="Master seed used to make every configuration reproducible.",
    )
    parser.add_argument(
        "--no-vasp-inputs",
        action="store_true",
        help="Write POSCAR and metadata only.",
    )
    parser.add_argument(
        "--min-distance-scale",
        type=float,
        default=1.0,
        help="Scale collision-filter thresholds (default: 1.0).",
    )
    return parser.parse_args()


def sample_seed(master_seed: int, phase: str, family: str, index: int, attempt: int) -> int:
    """Derive an independent, stable seed for a configuration attempt."""
    key = f"{master_seed}|parity-v1|{phase}|{family}|{index}|{attempt}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def distortion_label(family: str) -> str:
    """Return the explicit, filesystem-safe label used in parity structure names."""
    try:
        return FAMILY_LABELS[family]
    except KeyError as exc:
        raise ValueError(f"Unknown distortion family: {family}") from exc


def parity_config_id(phase: str, config_number: int, family: str, severity: float) -> str:
    """Include the perturbation family and severity directly in the directory name."""
    return (
        f"{phase}_parity_{config_number:02d}__{distortion_label(family)}"
        f"__sev-{severity:.1f}"
    )


def perturb(
    source: Atoms,
    fa_groups: list[list[int]],
    family: str,
    severity: float,
    rng: np.random.Generator,
) -> tuple[Atoms, dict[str, Any]]:
    """Create one distortion, with *severity* spanning 0.2 to 1.0."""
    atoms = source.copy()
    symbols = np.asarray(atoms.get_chemical_symbols())
    cage = np.flatnonzero(np.isin(symbols, ("Pb", "I")))
    all_indices = np.arange(len(atoms))
    details: dict[str, Any] = {
        "distortion_family": family,
        "distortion_severity": round(float(severity), 3),
    }

    if family == "strong_strain":
        # 2.8--7.0% diagonal and 1.1--2.8% shear strain, plus cage motion.
        diagonal = 0.018 + 0.052 * severity
        shear = 0.007 + 0.021 * severity
        strain = random_strain(atoms, rng, diagonal, shear)
        sigma = 0.035 + 0.065 * severity
        rattle_indices(atoms, cage, sigma, rng)
        details.update(
            strain=np.round(strain, 8).tolist(),
            rattle_target="Pb_I",
            rattle_sigma_A=sigma,
        )
    elif family == "global_rattle":
        # Random Cartesian motion of every atom.  At the upper end this is
        # roughly twice the largest displacement used in the training-candidate
        # generator, subject to the contact filter below.
        sigma = 0.045 + 0.115 * severity
        rattle_indices(atoms, all_indices, sigma, rng)
        details.update(rattle_target="all", rattle_sigma_A=sigma)
    elif family == "cage_rattle":
        sigma = 0.060 + 0.140 * severity
        rattle_indices(atoms, cage, sigma, rng)
        details.update(rattle_target="Pb_I", rattle_sigma_A=sigma)
    elif family == "fa_rotation_cage_rattle":
        # Uniform SO(3) rotations are deliberately large while leaving each FA
        # molecule internally rigid.  The cage is also displaced so this is not
        # merely an orientational test.
        details.update(apply_fa_rotations(atoms, fa_groups, rng, "uniform"))
        sigma = 0.035 + 0.075 * severity
        rattle_indices(atoms, cage, sigma, rng)
        details.update(rattle_target="Pb_I", rattle_sigma_A=sigma)
    elif family == "combined":
        diagonal = 0.015 + 0.040 * severity
        shear = 0.006 + 0.017 * severity
        strain = random_strain(atoms, rng, diagonal, shear)
        details.update(apply_fa_rotations(atoms, fa_groups, rng, "medium"))
        sigma = 0.035 + 0.090 * severity
        rattle_indices(atoms, all_indices, sigma, rng)
        details.update(
            strain=np.round(strain, 8).tolist(),
            rattle_target="all",
            rattle_sigma_A=sigma,
        )
    else:
        raise ValueError(f"Unknown distortion family: {family}")

    atoms.wrap()
    return atoms, details


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output}. Choose a new directory."
        )
    if args.min_distance_scale <= 0:
        raise ValueError("--min-distance-scale must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    all_structures: list[Atoms] = []
    rows: list[dict[str, Any]] = []

    for phase, source_path in DEFAULT_PHASES.items():
        source = read(source_path)
        source.pbc = True
        fa_groups = identify_fa_cations(source)
        phase_dir = args.output / phase
        phase_dir.mkdir()

        config_number = 0
        for family in FAMILIES:
            for family_index in range(1, PER_FAMILY + 1):
                config_number += 1
                severity = family_index / PER_FAMILY
                for attempt in range(1, 1001):
                    seed = sample_seed(args.seed, phase, family, family_index, attempt)
                    atoms, details = perturb(
                        source, fa_groups, family, severity, np.random.default_rng(seed)
                    )
                    valid, contact = collision_check(
                        atoms, fa_groups, args.min_distance_scale
                    )
                    if valid:
                        break
                else:
                    raise RuntimeError(
                        f"Could not make a valid {phase}/{family}/{family_index:02d} structure."
                    )

                config_id = parity_config_id(phase, config_number, family, severity)
                config_dir = phase_dir / config_id
                config_dir.mkdir()
                write(
                    config_dir / "POSCAR", atoms, format="vasp", direct=True,
                    sort=False, vasp5=True,
                )
                if not args.no_vasp_inputs:
                    prepare_vasp_inputs(config_dir, phase)

                metadata = {
                    "config_id": config_id,
                    "phase": phase,
                    "mode": "parity",
                    "seed": seed,
                    "source": str(source_path.relative_to(ROOT)),
                    "formula": atoms.get_chemical_formula(),
                    "natoms": len(atoms),
                    "fa_groups": fa_groups,
                    "cell_A": np.round(atoms.cell.array, 10).tolist(),
                    **details,
                    "distortion_label": distortion_label(family),
                    **contact,
                }
                (config_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )
                extxyz_atoms = atoms.copy()
                extxyz_atoms.info.update(
                    config_id=config_id, phase=phase, mode="parity", seed=seed,
                    source=metadata["source"], distortion_family=family,
                    distortion_label=distortion_label(family), distortion_severity=severity,
                )
                all_structures.append(extxyz_atoms)
                rows.append(
                    {
                        "config_id": config_id,
                        "phase": phase,
                        "family": family,
                        "distortion_label": distortion_label(family),
                        "severity": severity,
                        "seed": seed,
                        "natoms": len(atoms),
                        "rattle_sigma_A": metadata.get("rattle_sigma_A", ""),
                        "n_rotated_fa": len(metadata.get("rotations", [])),
                        "minimum_distance_A": metadata["minimum_distance_A"],
                        "directory": str(config_dir.relative_to(args.output)),
                    }
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
        "structures_per_phase": len(FAMILIES) * PER_FAMILY,
        "families": list(FAMILIES),
        "per_family_per_phase": PER_FAMILY,
        "phases": {key: str(value.relative_to(ROOT)) for key, value in DEFAULT_PHASES.items()},
        "vasp_inputs_created": not args.no_vasp_inputs,
        "labels_present": False,
        "note": "Independent parity-test set; do not include these frames in fine-tuning.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(rows)} parity candidates in {args.output}")
    for phase in DEFAULT_PHASES:
        print(f"  {phase}: {sum(row['phase'] == phase for row in rows)}")


if __name__ == "__main__":
    main()
