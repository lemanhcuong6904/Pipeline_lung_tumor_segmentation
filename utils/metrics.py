"""Metrics for binary segmentation."""

from __future__ import annotations

import torch


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Return thresholded binary micro-Dice for a batch or complete epoch."""
    predictions = torch.sigmoid(logits) >= threshold
    targets = targets >= 0.5
    intersection = (predictions & targets).sum().item()
    denominator = predictions.sum().item() + targets.sum().item()
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator
