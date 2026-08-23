#!/usr/bin/env python3
"""Add distortion family and severity to existing parity-candidate names.

The script preserves the original identifier in ``original_config_id`` and
updates directory names, per-structure metadata, the root metadata CSV, and
the optional structures.extxyz together.  Run without ``--apply`` to preview.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ase.io import read, write

from generate_parity_set import ROOT, distortion_label, parity_config_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "parity_dataset_candidates",
        help="Parity-candidate directory to rename (default: %(default)s).",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Perform the rename and metadata updates. Without this flag, only preview them.",
    )
    return parser.parse_args()


def read_metadata(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    metadata_paths = sorted(args.input.glob("*/*/metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No metadata.json files found below {args.input}")

    plans: list[dict[str, object]] = []
    for metadata_path in metadata_paths:
        metadata = read_metadata(metadata_path)
        old_config_id = str(metadata["config_id"])
        original_config_id = str(metadata.get("original_config_id", old_config_id.split("__", 1)[0]))
        family = str(metadata["distortion_family"])
        severity = float(metadata["distortion_severity"])
        phase = str(metadata["phase"])
        config_number = int(original_config_id.rsplit("_", 1)[1])
        new_config_id = parity_config_id(phase, config_number, family, severity)
        old_dir = metadata_path.parent
        new_dir = old_dir.parent / new_config_id
        plans.append({
            "old_config_id": old_config_id,
            "original_config_id": original_config_id,
            "new_config_id": new_config_id,
            "old_dir": old_dir,
            "new_dir": new_dir,
            "metadata": metadata,
        })

    new_ids = [str(plan["new_config_id"]) for plan in plans]
    if len(new_ids) != len(set(new_ids)):
        raise ValueError("Renaming plan has duplicate target configuration IDs")
    for plan in plans:
        old_dir, new_dir = Path(plan["old_dir"]), Path(plan["new_dir"])
        if old_dir != new_dir and new_dir.exists():
            raise FileExistsError(f"Rename target already exists: {new_dir}")

    changed = [plan for plan in plans if plan["old_dir"] != plan["new_dir"]]
    for plan in changed:
        print(f"{Path(plan['old_dir']).name} -> {Path(plan['new_dir']).name}")
    print(f"{len(changed)} of {len(plans)} candidate directories require renaming.")
    if not args.apply:
        print("Preview only. Re-run with --apply to rename directories and update metadata.")
        return

    for plan in changed:
        Path(plan["old_dir"]).rename(Path(plan["new_dir"]))

    id_map = {str(plan["old_config_id"]): str(plan["new_config_id"]) for plan in plans}
    directory_map = {
        str(Path(plan["old_dir"]).relative_to(args.input)): str(Path(plan["new_dir"]).relative_to(args.input))
        for plan in plans
    }
    for plan in plans:
        metadata = dict(plan["metadata"])
        metadata["original_config_id"] = plan["original_config_id"]
        metadata["config_id"] = plan["new_config_id"]
        metadata["distortion_label"] = distortion_label(str(metadata["distortion_family"]))
        metadata_path = Path(plan["new_dir"]) / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    csv_path = args.input / "metadata.csv"
    if csv_path.is_file():
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0]) if rows else []
        for field in ("original_config_id", "distortion_label"):
            if field not in fieldnames:
                fieldnames.insert(fieldnames.index("config_id") + 1, field)
        for row in rows:
            old_config_id = row["config_id"]
            row["original_config_id"] = old_config_id.split("__", 1)[0]
            row["config_id"] = id_map[old_config_id]
            row["directory"] = directory_map[row["directory"]]
            row["distortion_label"] = distortion_label(row["family"])
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    extxyz_path = args.input / "structures.extxyz"
    if extxyz_path.is_file():
        structures = read(extxyz_path, index=":")
        for atoms in structures:
            old_config_id = str(atoms.info["config_id"])
            atoms.info["original_config_id"] = old_config_id.split("__", 1)[0]
            atoms.info["config_id"] = id_map[old_config_id]
            atoms.info["distortion_label"] = distortion_label(str(atoms.info["distortion_family"]))
        write(extxyz_path, structures, format="extxyz")

    mapping_path = args.input / "parity_config_id_rename_map.csv"
    with mapping_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("original_config_id", "config_id", "old_directory", "directory"))
        writer.writeheader()
        writer.writerows({
            "original_config_id": plan["original_config_id"],
            "config_id": plan["new_config_id"],
            "old_directory": Path(plan["old_dir"]).relative_to(args.input),
            "directory": Path(plan["new_dir"]).relative_to(args.input),
        } for plan in plans)
    print(f"Renamed {len(changed)} directories and updated {csv_path.name}, metadata.json, and structures.extxyz.")


if __name__ == "__main__":
    main()
