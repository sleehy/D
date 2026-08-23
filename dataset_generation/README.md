# FAPbI3 MLP dataset candidate generator

This generator creates perturbed structures from the current
`O-FAPI3/CONTCAR` and `t-FAPI3/CONTCAR`.

Edit `MODE_COUNTS` near the top of `generate_fapbi3_dataset.py` to control each
type independently. Every value is the number generated **per phase**:

```python
MODE_COUNTS = {
    "small_rotation": 18,
    "wide_rotation": 14,
    "cage_rattle": 17,
    "all_rattle": 17,
    "strain_rattle": 17,
    "mixed": 17,
}
```

These defaults generate 100 structures per phase and 200 in total. For example,
changing `wide_rotation` from 14 to 10 creates ten wide-rotation structures for
O-FAPI3 and ten for t-FAPI3 in a new dataset. Set a mode to zero to disable it.

| Mode | Perturbation |
|---|---|
| `small_rotation` | Rotate one or more FA cations by up to 15 degrees |
| `wide_rotation` | Uniform random SO(3) orientation for selected FA cations |
| `cage_rattle` | Randomly displace only Pb and I atoms |
| `all_rattle` | Randomly displace every atom |
| `strain_rattle` | Cell strain/shear plus Pb-I cage displacement |
| `mixed` | Moderate FA rotation plus strain and all-atom displacement |

FA cations are identified automatically from minimum-image C-N and C-H
distances. Each FA molecule is unwrapped across periodic boundaries before it is
rotated as a rigid body. Structures with severe short contacts are rejected.

## Generate candidates

From the repository root:

```bash
python dataset_generation/generate_fapbi3_dataset.py
```

To use a different random seed or output directory:

```bash
python dataset_generation/generate_fapbi3_dataset.py \
  --seed 12345 \
  --output mlp_dataset_candidates_seed12345
```

Without `--extend`, the output directory must be new or empty. This prevents
accidental replacement of existing DFT results.

To expand an existing candidate directory up to the current `MODE_COUNTS`
targets while preserving every existing configuration and VASP result:

```bash
python dataset_generation/generate_fapbi3_dataset.py --extend
```

Extension mode validates that existing mode indices are contiguous, refuses to
overwrite any unrecognized directory, and regenerates the root-level
`structures.extxyz`, `metadata.csv`, and `manifest.json` summaries. New sample
seeds are deterministic per phase, mode, index, and attempt, so extension can be
run again safely after an interruption.

## Output

Each configuration directory contains:

- `POSCAR`: perturbed structure
- `INCAR`: VASP single-point settings (`NSW=0`, `ISIF=2`)
- `KPOINTS`: copied from the corresponding phase
- `POTCAR`: relative symbolic link to the repository-level `POTCAR`
- `metadata.json`: seed and exact perturbation parameters

The output root also contains:

- `structures.extxyz`: all generated unlabeled candidate structures
- `metadata.csv`: flat summary
- `manifest.json`: generation settings

These structures are not an MLP training dataset until DFT energy, forces, and
stress have been calculated and stored as labels. Do not relax the perturbed
structures before collecting labels; run single-point calculations.

## Suggested DFT workflow

Run VASP separately in every configuration directory. After the calculations
finish, verify electronic convergence and collect:

- total energy
- atomic forces
- stress tensor

To run all configuration directories sequentially with two MPI ranks:

```bash
python dataset_generation/run_vasp_calculations.py
```

The runner skips calculations whose `OUTCAR` and `vasprun.xml` already satisfy
the collector's completion checks. If it is interrupted, run the same command
again: the completed directories are skipped and the interrupted directory is
run again. Output from each attempt is appended to `vasp.run.log` in that
configuration directory. Preview the work without launching VASP with
`--dry-run`, or explicitly start at a configuration with, for example,
`--start-at O-FAPI3_all_rattle_02`.

When all single-point calculations have finished, collect the labels:

```bash
python dataset_generation/collect_vasp_results.py
```

The collector requires a complete `OUTCAR`, an `EDIFF` convergence marker, and
`vasprun.xml` in each configuration directory. It writes:

- `mlp_dataset_candidates/labeled.extxyz`
- `mlp_dataset_candidates/labeled.csv`
- `mlp_dataset_candidates/collection_failures.csv`

By default it refuses to create a partial dataset. To intentionally collect only
the completed calculations:

```bash
python dataset_generation/collect_vasp_results.py --allow-partial
```

ASE uses its own stress sign convention when parsing VASP output; keep that
convention consistent with the MLP framework used for training.

Trajectory-correlated structures should stay in the same train/validation/test
split. As the dataset grows, split by parent structure, perturbation seed, or MD
trajectory instead of randomly splitting individual frames.
