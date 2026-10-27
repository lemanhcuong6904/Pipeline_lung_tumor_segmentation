"""Training lifecycle helpers shared by trainer entry points."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scheduler_step_counts(*, loader_batches: int, train_batches_per_epoch: int | None, epochs: int, warmup_fraction: float) -> tuple[int, int]:
    batches = loader_batches if train_batches_per_epoch is None else min(loader_batches, train_batches_per_epoch)
    if batches < 1 or epochs < 1 or not 0 <= warmup_fraction <= 1:
        raise ValueError("invalid scheduler step configuration")
    total_steps = batches * epochs
    return total_steps, math.ceil(total_steps * warmup_fraction)


def warmup_cosine_step_lambda(total_steps: int, warmup_steps: int, lr: float, lr_min: float) -> Callable[[int], float]:
    if total_steps < 1 or warmup_steps < 0 or warmup_steps > total_steps or lr <= 0 or not 0 <= lr_min <= lr:
        raise ValueError("invalid warm-up/cosine scheduler configuration")
    minimum = lr_min / lr
    def schedule(step: int) -> float:
        step = min(max(step, 0), total_steps - 1)
        if warmup_steps and step < warmup_steps:
            return minimum + (1 - minimum) * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
        return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))
    return schedule


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)


def load_metrics_history(path: str | Path) -> list[dict[str, float | int]]:
    """Read an epoch-metric history, returning an empty one when absent."""
    path = Path(path)
    if not path.is_file():
        return []
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list):
        raise ValueError(f"metrics history must be a list: {path}")
    return history


def append_metrics_history(
    path: str | Path,
    history: list[dict[str, float | int]],
    values: dict[str, float | int],
) -> list[dict[str, float | int]]:
    """Upsert one epoch's metrics and atomically persist the full history."""
    epoch = values["epoch"]
    updated = [row for row in history if row.get("epoch") != epoch] + [values]
    updated.sort(key=lambda row: int(row["epoch"]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return updated
