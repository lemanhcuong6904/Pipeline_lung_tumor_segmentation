"""Loss functions for binary classification from logits."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    """Weighted BCE modulated by the probability assigned to the wrong class."""

    def __init__(self, pos_weight: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        if not math.isfinite(pos_weight) or pos_weight <= 0:
            raise ValueError("POS_WEIGHT must be positive and finite")
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("FOCAL_GAMMA must be non-negative and finite")
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        # Keep the modulation factor in float32 with AMP to avoid early
        # saturation in float16/bfloat16 sigmoid outputs.
        modulation_logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
        wrong_probability = (targets * torch.sigmoid(-modulation_logits)
                             + (1 - targets) * torch.sigmoid(modulation_logits))
        return (loss * wrong_probability.pow(self.gamma)).mean()


def create_classification_loss(name: str, pos_weight: float, gamma: float,
                               device: torch.device) -> nn.Module:
    """Create the configured loss on the training device."""
    if not math.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError("POS_WEIGHT must be positive and finite")
    if name == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    if name == "focal":
        return BinaryFocalLoss(pos_weight=pos_weight, gamma=gamma).to(device)
    raise ValueError("LOSS must be 'weighted_bce' or 'focal'")
