#!/usr/bin/env python3
"""Generate non-overlapping FAPbI3 active-learning candidates.

The requested cohorts are sampled uniformly inside their severity intervals.
Twenty percent of each cohort reuse an exact per-structure seed from the
existing parity set, while the other eighty percent use a separate master
seed.  The midpoint severity grid keeps reused-seed structures distinct from
the parity set's 0.1-spaced severity values; an additional structure
fingerprint check prevents any accidental duplicate.
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

from generate_fapbi3_dataset import DEFAULT_PHASES, ROOT, collision_check, identify_fa_cations, prepare_vasp_inputs
from generate_parity_set import distortion_label, perturb


COHORTS = (
    ("t-FAPI3", "global_rattle", 25, 0.2, 0.6),
    ("t-FAPI3", "combined", 25, 0.2, 0.6),
    ("O-FAPI3", "global_rattle", 30, 0.2, 0.6),
    ("O-FAPI3", "combined", 20, 0.4, 0.7),
)
PARITY_MASTER_SEED = 20260810


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "active_learning_candidates",
        help="Empty output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--parity-input", type=Path, default=ROOT / "parity_dataset_candidates",
        help="Existing parity set used for seed reuse and duplicate exclusion.",
    )
    parser.add_argument(
        "--new-master-seed", type=int, default=20260901,
        help="Master seed for the 80%% of structures that do not reuse a parity seed.",
    )
    parser.add_argument(
        "--no-vasp-inputs", action="store_true",
        help="Write POSCAR and metadata only.",
    )
    parser.add_argument(
        "--min-distance-scale", type=float, default=1.0,
        help="Scale collision-filter thresholds (default: 1.0).",
    )
    return parser.parse_args()


def fingerprint(atoms: Atoms) -> str:
    """Hash symbols, cell, and wrapped fractional coordinates at high precision."""
    payload = "|".join(atoms.get_chemical_symbols()).encode()
    payload += np.round(atoms.cell.array, 10).tobytes()
    payload += np.round(atoms.get_scaled_positions(wrap=True), 10).tobytes()
    return hashlib.sha256(payload).hexdigest()


def midpoint_severities(low: float, high: float, count: int) -> list[float]:
    """Uniform values strictly inside an interval, avoiding parity's endpoints."""
    return [round(low + (high - low) * (index + 0.5) / count, 6) for index in range(count)]


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Choose ``count`` well-distributed zero-based indices from a sequence."""
    return [int((index + 0.5) * total / count) for index in range(count)]


def new_seed(master_seed: int, phase: str, family: str, index: int, attempt: int) -> int:
    key = f"{master_seed}|active-learning-v1|{phase}|{family}|{index}|{attempt}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def parity_seed_records(parity_input: Path) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], set[int], set[str]]:
    """Read parity metadata, its exact seeds, and fingerprints to exclude reuse."""
    records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    used_seeds: set[int] = set()
    fingerprints: set[str] = set()
    metadata_paths = sorted(parity_input.glob("*/*/metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No parity metadata files found below {parity_input}")
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        phase, family = str(metadata["phase"]), str(metadata["distortion_family"])
        record = {
            "config_id": str(metadata["config_id"]),
            "seed": int(metadata["seed"]),
            "severity": float(metadata["distortion_severity"]),
        }
        records.setdefault((phase, family), []).append(record)
        used_seeds.add(record["seed"])
        poscar_path = metadata_path.parent / "POSCAR"
        if not poscar_path.is_file():
            raise FileNotFoundError(f"Parity candidate POSCAR missing: {poscar_path}")
        fingerprints.add(fingerprint(read(poscar_path)))
    for items in records.values():
        items.sort(key=lambda item: (item["severity"], item["config_id"]))
    return records, used_seeds, fingerprints


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    if args.min_distance_scale <= 0:
        raise ValueError("--min-distance-scale must be positive")

    parity_records, parity_seeds, existing_fingerprints = parity_seed_records(args.parity_input)
    args.output.mkdir(parents=True, exist_ok=True)
    structures: list[Atoms] = []
    rows: list[dict[str, object]] = []
    used_new_seeds: set[int] = set()

    for phase, family, count, low, high in COHORTS:
        source_path = DEFAULT_PHASES[phase]
        source = read(source_path)
        source.pbc = True
        fa_groups = identify_fa_cations(source)
        phase_dir = args.output / phase
        phase_dir.mkdir(exist_ok=True)
        same_seed_count = count // 5
        reuse_indices = set(evenly_spaced_indices(count, same_seed_count))
        source_records = parity_records.get((phase, family), [])
        if len(source_records) < same_seed_count:
            raise ValueError(f"Need {same_seed_count} parity seeds for {phase}/{family}")
        reused_seed_records = [source_records[index] for index in evenly_spaced_indices(len(source_records), same_seed_count)]
        reused_seed_by_index = dict(zip(sorted(reuse_indices), reused_seed_records, strict=True))
        spare_reuse_records = [record for record in source_records if record not in reused_seed_records]
        used_reused_seeds: set[int] = set()

        for index, severity in enumerate(midpoint_severities(low, high, count), start=1):
            zero_based_index = index - 1
            requested_reuse_record = reused_seed_by_index.get(zero_based_index)
            reuse_record = None
            if requested_reuse_record:
                # A seed that is valid at its parity severity can fail the stricter
                # collision check at a new severity.  Fall back only to unused
                # parity seeds from this same phase/family, retaining the 20% ratio.
                for candidate_record in [requested_reuse_record, *spare_reuse_records]:
                    seed = int(candidate_record["seed"])
                    if seed in used_reused_seeds:
                        continue
                    atoms, details = perturb(source, fa_groups, family, severity, np.random.default_rng(seed))
                    valid, contact = collision_check(atoms, fa_groups, args.min_distance_scale)
                    if valid and fingerprint(atoms) not in existing_fingerprints:
                        reuse_record = candidate_record
                        used_reused_seeds.add(seed)
                        break
                if reuse_record is None:
                    raise RuntimeError(f"Could not reuse a unique valid parity seed for {phase}/{family}/{index:02d}")
            else:
                for attempt in range(1, 1001):
                    seed = new_seed(args.new_master_seed, phase, family, index, attempt)
                    if seed in parity_seeds or seed in used_new_seeds:
                        continue
                    atoms, details = perturb(source, fa_groups, family, severity, np.random.default_rng(seed))
                    valid, contact = collision_check(atoms, fa_groups, args.min_distance_scale)
                    if valid and fingerprint(atoms) not in existing_fingerprints:
                        break
                else:
                    raise RuntimeError(f"Could not generate unique valid {phase}/{family}/{index:02d}")

            seed_origin = "parity" if reuse_record else "new"
            source_config_id = str(reuse_record["config_id"]) if reuse_record else ""

            config_id = (
                f"{phase}_al_{family}_{index:02d}__{distortion_label(family)}"
                f"__sev-{severity:.3f}__seed-{seed_origin}"
            )
            config_dir = phase_dir / config_id
            config_dir.mkdir()
            write(config_dir / "POSCAR", atoms, format="vasp", direct=True, sort=False, vasp5=True)
            if not args.no_vasp_inputs:
                prepare_vasp_inputs(config_dir, phase)
            metadata = {
                "config_id": config_id,
                "phase": phase,
                "mode": "active_learning",
                "seed": seed,
                "seed_origin": seed_origin,
                "seed_origin_config_id": source_config_id,
                "parity_master_seed": PARITY_MASTER_SEED if reuse_record else None,
                "new_master_seed": None if reuse_record else args.new_master_seed,
                "source": str(source_path.relative_to(ROOT)),
                "formula": atoms.get_chemical_formula(),
                "natoms": len(atoms),
                "fa_groups": fa_groups,
                "cell_A": np.round(atoms.cell.array, 10).tolist(),
                **details,
                "distortion_label": distortion_label(family),
                **contact,
            }
            (config_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            extxyz_atoms = atoms.copy()
            extxyz_atoms.info.update(**metadata)
            structures.append(extxyz_atoms)
            rows.append({
                "config_id": config_id,
                "phase": phase,
                "family": family,
                "distortion_label": distortion_label(family),
                "severity": severity,
                "seed": seed,
                "seed_origin": seed_origin,
                "seed_origin_config_id": source_config_id,
                "minimum_distance_A": metadata["minimum_distance_A"],
                "directory": str(config_dir.relative_to(args.output)),
            })
            existing_fingerprints.add(fingerprint(atoms))
            if not reuse_record:
                used_new_seeds.add(seed)

    write(args.output / "structures.extxyz", structures, format="extxyz")
    with (args.output / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n_structures": len(rows),
        "n_parity_seed_reused": sum(row["seed_origin"] == "parity" for row in rows),
        "n_new_seed": sum(row["seed_origin"] == "new" for row in rows),
        "cohorts": [
            {"phase": phase, "family": family, "count": count, "severity_range": [low, high]}
            for phase, family, count, low, high in COHORTS
        ],
    }
    (args.output / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Generated {len(rows)} active-learning candidates in {args.output}")
    print(f"Parity-seed reuse: {summary['n_parity_seed_reused']} (20%); new seeds: {summary['n_new_seed']} (80%)")


if __name__ == "__main__":
    main()
