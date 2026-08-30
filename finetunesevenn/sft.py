"""Fine-tune SevenNet from ``best.pth`` using data in ``dataset/``.

This source was restored from the accompanying CPython bytecode cache.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import random
import re

import torch
from sevenn import util
from sevenn.error_recorder import ErrorMetric, ErrorRecorder, get_err_type
from sevenn.logger import Logger
from sevenn.nn.scale import SpeciesWiseRescale
from sevenn.train.graph_dataset import SevenNetGraphDataset
from sevenn.train.trainer import Trainer
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "best.pth"
DEFAULT_DATASET_DIR = PROJECT_DIR / "dataset"
DEFAULT_OUTPUT = PROJECT_DIR / "checkpoint_fine_tuned.pth"
DEFAULT_METRICS = PROJECT_DIR / "finetune_metrics.csv"
DEFAULT_PLOT = PROJECT_DIR / "finetune_metrics.png"

# Keep these in a fixed order so the CSV and plot remain directly comparable
# across runs.  SevenNet reports the corresponding values per epoch.
ERROR_TARGETS = ("Energy", "Force", "Stress")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS,
        help="CSV file written with train/validation metrics after every epoch.",
    )
    parser.add_argument(
        "--plot-file",
        type=Path,
        default=DEFAULT_PLOT,
        help="PNG learning-curve plot written after fine-tuning.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument(
        "--best-metric",
        choices=("Energy_RMSE", "Force_RMSE", "Stress_RMSE", "TotalLoss"),
        default="Energy_RMSE",
        help=(
            "Validation metric minimized when selecting the saved checkpoint "
            "(default: Energy_RMSE)."
        ),
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Fraction of structures used for stratified validation (default: 0.2).",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help=(
            "JSON manifest for the fixed train/validation split "
            "(default: DATASET_DIR/stratified_split.json)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Training device (default: auto).",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Rebuild the cached SevenNet graph dataset.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not 0.0 < args.valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be between 0 and 1")


def find_dataset_files(dataset_dir: Path) -> list[Path]:
    files = sorted(dataset_dir.rglob("*.extxyz")) if dataset_dir.is_dir() else []
    if not files:
        raise FileNotFoundError(
            f"No .extxyz files found under {dataset_dir}. "
            "Put labeled training data in that directory first."
        )
    return files


def _extxyz_field(header: str, name: str) -> str | None:
    """Read an unquoted or quoted key=value field from an extxyz comment line."""
    match = re.search(rf'(?:^|\s){re.escape(name)}=(?:"([^"]*)"|(\S+))', header)
    if match is None:
        return None
    return match.group(1) if match.group(1) is not None else match.group(2)


def read_structure_metadata(files: list[Path]) -> list[dict[str, str]]:
    """Read labels used for splitting, retaining the extxyz file order."""
    records: list[dict[str, str]] = []
    for path in files:
        with path.open() as handle:
            structure_number = 0
            while line := handle.readline():
                try:
                    natoms = int(line.strip())
                except ValueError as exc:
                    raise ValueError(f"Invalid atom count in {path}: {line!r}") from exc
                header = handle.readline()
                if not header:
                    raise ValueError(f"Missing extxyz header after structure in {path}")
                structure_number += 1
                config_id = _extxyz_field(header, "config_id")
                phase = _extxyz_field(header, "phase")
                mode = _extxyz_field(header, "mode")
                if not all((config_id, phase, mode)):
                    raise ValueError(
                        "Stratified splitting requires config_id, phase, and mode in every "
                        f"extxyz header; missing field in {path}, structure {structure_number}."
                    )
                records.append(
                    {"config_id": config_id, "phase": phase, "mode": mode}
                )
                for _ in range(natoms):
                    if not handle.readline():
                        raise ValueError(f"Unexpected end of file in {path}")
    ids = [record["config_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("config_id values must be unique to create a persistent split.")
    return records


def allocate_counts(counts: dict[str, int], total: int, rng: random.Random) -> dict[str, int]:
    """Allocate *total* items proportionally using the largest-remainder method."""
    if total < 0 or total > sum(counts.values()):
        raise ValueError("Invalid split allocation target")
    population = sum(counts.values())
    raw = {key: count * total / population for key, count in counts.items()}
    allocation = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(allocation.values())
    # Shuffle first so equal fractional remainders are resolved reproducibly by seed.
    candidates = list(counts)
    rng.shuffle(candidates)
    candidates.sort(key=lambda key: raw[key] - allocation[key], reverse=True)
    for key in candidates[:remaining]:
        allocation[key] += 1
    return allocation


def stratified_split(
    records: list[dict[str, str]], valid_ratio: float, seed: int
) -> tuple[list[int], list[int], dict[str, int]]:
    """Split by phase, then by perturbation mode within each phase."""
    rng = random.Random(seed)
    by_phase: dict[str, dict[str, list[int]]] = {}
    for index, record in enumerate(records):
        by_phase.setdefault(record["phase"], {}).setdefault(record["mode"], []).append(index)

    num_valid = min(len(records) - 1, max(1, round(len(records) * valid_ratio)))
    phase_sizes = {phase: sum(map(len, modes.values())) for phase, modes in by_phase.items()}
    phase_targets = allocate_counts(phase_sizes, num_valid, rng)

    valid_indices: list[int] = []
    stratum_counts: dict[str, int] = {}
    for phase in sorted(by_phase):
        modes = by_phase[phase]
        mode_targets = allocate_counts(
            {mode: len(indices) for mode, indices in modes.items()},
            phase_targets[phase],
            rng,
        )
        for mode in sorted(modes):
            selected = rng.sample(modes[mode], mode_targets[mode])
            valid_indices.extend(selected)
            stratum_counts[f"{phase}/{mode}"] = mode_targets[mode]

    valid_indices.sort()
    valid_set = set(valid_indices)
    train_indices = [index for index in range(len(records)) if index not in valid_set]
    return train_indices, valid_indices, stratum_counts


def load_or_create_split(
    records: list[dict[str, str]], split_file: Path, valid_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """Persist split membership by config_id, so extxyz ordering cannot change it."""
    ids = [record["config_id"] for record in records]
    index_by_id = {config_id: index for index, config_id in enumerate(ids)}
    if split_file.is_file():
        with split_file.open() as handle:
            manifest = json.load(handle)
        valid_ids = manifest.get("valid_config_ids")
        train_ids = manifest.get("train_config_ids")
        if not isinstance(valid_ids, list) or not isinstance(train_ids, list):
            raise ValueError(f"Invalid split manifest: {split_file}")
        if set(valid_ids) | set(train_ids) != set(ids) or set(valid_ids) & set(train_ids):
            raise ValueError(
                f"Split manifest {split_file} does not match the current dataset. "
                "Choose a new --split-file to create a new split."
            )
        return [index_by_id[item] for item in train_ids], [index_by_id[item] for item in valid_ids]

    train_indices, valid_indices, stratum_counts = stratified_split(records, valid_ratio, seed)
    manifest = {
        "split_type": "stratified_by_phase_and_mode",
        "seed": seed,
        "valid_ratio": valid_ratio,
        "num_train": len(train_indices),
        "num_valid": len(valid_indices),
        "valid_count_by_phase_mode": stratum_counts,
        "train_config_ids": [ids[index] for index in train_indices],
        "valid_config_ids": [ids[index] for index in valid_indices],
    }
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with split_file.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return train_indices, valid_indices


def load_finetuning_model(checkpoint: Path):
    model, config = util.model_from_checkpoint(str(checkpoint))

    # SevenNet-0 uses SpeciesWiseRescale.  In that case, retain its
    # element-wise parameters but make them trainable for fine-tuning.
    #
    # This project's best.pth instead uses the scalar Rescale module.  Turning
    # that scalar into SpeciesWiseRescale would produce one scale value, then
    # incorrectly index it once per atom type.  Leave scalar Rescale intact;
    # it already has trainable shift/scale parameters.
    shift_scale = model._modules["rescale_atomic_energy"]
    if isinstance(shift_scale, SpeciesWiseRescale):
        model._modules["rescale_atomic_energy"] = SpeciesWiseRescale(
            shift_scale.shift.tolist(),
            shift_scale.scale.tolist(),
            train_shift_scale=True,
        )
    else:
        shift_scale.shift.requires_grad_(True)
        shift_scale.scale.requires_grad_(True)
    return model, config


class R2Score(ErrorMetric):
    """Streaming coefficient of determination for one SevenNet target."""

    def __init__(self, **kwargs) -> None:
        kwargs.pop("unit", None)
        super().__init__(unit=None, **kwargs)
        self.name = f"{self.name}_R2"
        self.reset()

    def update(self, output) -> None:
        y_ref, y_pred = self._retrieve(output)
        y_ref = y_ref.reshape(-1)
        y_pred = y_pred.reshape(-1)
        self._sum_squared_error += torch.sum((y_ref - y_pred) ** 2).item()
        self._sum_reference += torch.sum(y_ref).item()
        self._sum_squared_reference += torch.sum(y_ref**2).item()
        self._count += y_ref.numel()

    def get(self) -> float:
        if self._count == 0:
            return float("nan")
        total_variance = (
            self._sum_squared_reference
            - self._sum_reference**2 / self._count
        )
        if total_variance <= 0.0:
            return float("nan")
        return 1.0 - self._sum_squared_error / total_variance

    def reset(self) -> None:
        self._sum_squared_error = 0.0
        self._sum_reference = 0.0
        self._sum_squared_reference = 0.0
        self._count = 0


def error_recorder_with_r2(config: dict) -> ErrorRecorder:
    """Create the standard SevenNet recorder plus Energy/Force/Stress R²."""
    recorder = ErrorRecorder.from_config(config)
    for target in ERROR_TARGETS:
        recorder.metrics.append(R2Score(**get_err_type(target)))
    return recorder


def write_metrics_csv(history: list[dict[str, float]], path: Path) -> None:
    """Write a machine-readable epoch-versus-performance table."""
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_metrics(history: list[dict[str, float]], path: Path) -> None:
    """Plot errors plus separate train/validation R² panels by epoch."""
    if not history:
        return
    try:
        cache_dir = path.parent / ".matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        import matplotlib

        matplotlib.use("Agg")  # Allow training on headless compute nodes.
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it or provide a Python environment "
            "that includes matplotlib."
        ) from exc

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    marker_size = 2.5
    for axis, target in zip(axes.flat[:3], ERROR_TARGETS):
        # SevenNet reports energy errors in eV; display them in meV so that
        # the learning curve is easier to read without changing the CSV data.
        scale = 1_000.0 if target == "Energy" else 1.0
        ylabel = "RMSE / MAE (meV)" if target == "Energy" else "RMSE / MAE"
        for metric, style in (("RMSE", "-"), ("MAE", "--")):
            for split, color in (("train", "tab:blue"), ("valid", "tab:orange")):
                key = f"{split}_{target}_{metric}"
                axis.plot(
                    epochs,
                    [scale * row[key] for row in history],
                    color=color,
                    linestyle=style,
                    marker="o",
                    markersize=marker_size,
                    label=f"{split.title()} {metric}",
                )
        axis.set(title=target, xlabel="epoch", ylabel=ylabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize="small")

    loss_axis = axes.flat[3]
    for split, color in (("train", "tab:blue"), ("valid", "tab:orange")):
        loss_axis.plot(
            epochs,
            [row[f"{split}_TotalLoss"] for row in history],
            color=color,
            marker="o",
            markersize=marker_size,
            label=f"{split.title()} TotalLoss",
        )
    loss_axis.set(title="TotalLoss", xlabel="Fine-tuning epoch", ylabel="TotalLoss")
    loss_axis.grid(alpha=0.3)
    loss_axis.legend()

    for axis, split in zip(axes.flat[4:], ("train", "valid")):
        r2_values = []
        for target, color in zip(ERROR_TARGETS, ("tab:blue", "tab:orange", "tab:green")):
            values = [row[f"{split}_{target}_R2"] for row in history]
            r2_values.extend(values)
            axis.plot(
                epochs,
                values,
                color=color,
                marker="o",
                markersize=marker_size,
                label=target,
            )
        minimum = min(r2_values)
        lower = min(-0.05, minimum - 0.05 * max(1.0, abs(minimum)))
        axis.set(
            title=f"{split.title()} R²",
            xlabel="Fine-tuning epoch",
            ylabel="R²",
            ylim=(lower, 1.05),
        )
        axis.grid(alpha=0.3)
        axis.legend(fontsize="small")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset_files = find_dataset_files(args.dataset_dir)
    model, config = load_finetuning_model(args.checkpoint)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    config.update(
        {
            "device": device,
            "optimizer": "adam",
            "optim_param": {"lr": args.learning_rate},
            "scheduler": "reducelronplateau",
            "scheduler_param": {
                "mode": "min",
                "factor": 0.5,
                "patience": 5,
                "min_lr": 1.0e-5
            },
            "best_metric": args.best_metric,
            "is_ddp": False,
        }
    )

    dataset = SevenNetGraphDataset(
        config["cutoff"],
        root=str(PROJECT_DIR),
        files=[str(path) for path in dataset_files],
        processed_name="finetune.pt",
        force_reload=args.force_reload,
    )
    if len(dataset) < 2:
        raise ValueError("At least two structures are required for train/validation splitting.")

    records = read_structure_metadata(dataset_files)
    if len(records) != len(dataset):
        raise ValueError(
            "The number of extxyz structures does not match the graph dataset "
            f"({len(records)} != {len(dataset)}). Rebuild with --force-reload."
        )
    split_seed = int(config.get("random_seed", 7))
    split_file = args.split_file or args.dataset_dir / "stratified_split.json"
    train_indices, valid_indices = load_or_create_split(
        records, split_file, args.valid_ratio, split_seed
    )
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    num_train, num_valid = len(train_dataset), len(valid_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset files: {len(dataset_files)}")
    print(f"Structures: {len(dataset)} (train={num_train}, valid={num_valid})")
    print(f"Stratified split manifest: {split_file}")
    print(f"Device: {device}")
    print(f"Epoch metrics CSV: {args.metrics_file}")
    print(f"Learning-curve plot: {args.plot_file}")

    trainer = Trainer.from_config(model, config)
    train_recorder = error_recorder_with_r2(config)
    valid_recorder = deepcopy(train_recorder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_value = float("inf")
    best_epoch: int | None = None
    metrics_history: list[dict[str, float]] = []

    logger = Logger()
    logger.screen = True
    with logger:
        logger.greeting()
        for epoch in range(1, args.epochs + 1):
            logger.timer_start("epoch")
            learning_rate = trainer.get_lr()
            logger.writeline(
                f"Epoch {epoch}/{args.epochs}  Learning rate: {learning_rate:.6f}"
            )
            trainer.run_one_epoch(train_loader, is_train=True, error_recorder=train_recorder, wrap_tqdm=True)
            train_metrics = train_recorder.get_metric_dict(with_unit=False)
            train_error = train_recorder.epoch_forward()
            trainer.run_one_epoch(valid_loader, is_train=False, error_recorder=valid_recorder, wrap_tqdm=True)
            valid_metrics = valid_recorder.get_metric_dict(with_unit=False)
            valid_error = valid_recorder.epoch_forward()
            trainer.scheduler_step(valid_metrics[args.best_metric])
            epoch_metrics = {"epoch": epoch, "learning_rate": learning_rate}
            epoch_metrics.update(
                {f"train_{name}": value for name, value in train_metrics.items()}
            )
            epoch_metrics.update(
                {f"valid_{name}": value for name, value in valid_metrics.items()}
            )
            metrics_history.append(epoch_metrics)
            # Persist progress at every epoch: an interrupted GPU job still leaves
            # a complete record up to its final completed epoch.
            write_metrics_csv(metrics_history, args.metrics_file)
            plot_metrics(metrics_history, args.plot_file)
            logger.bar()
            logger.write_full_table([train_error, valid_error], ["Train", "Valid"])
            current_value = valid_metrics[args.best_metric]
            if current_value < best_value:
                best_value = current_value
                best_epoch = epoch
                trainer.write_checkpoint(str(args.output), config=config, epoch=epoch)
                logger.writeline(
                    f"Saved best checkpoint: epoch {epoch} "
                    f"validation {args.best_metric}={best_value:.8g}"
                )
            logger.timer_end("epoch", message=f"Epoch {epoch} elapsed")

    if best_epoch is None:
        raise RuntimeError(
            f"No finite validation {args.best_metric} was produced; checkpoint was not saved."
        )
    print(
        f"Best fine-tuned checkpoint saved to: {args.output} "
        f"(epoch {best_epoch}, validation {args.best_metric}={best_value:.8g})"
    )
    print(f"Epoch metrics written to: {args.metrics_file}")
    print(f"Learning-curve plot written to: {args.plot_file}")


if __name__ == "__main__":
    main()
