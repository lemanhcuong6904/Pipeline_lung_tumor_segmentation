"""Tumor losses: BCE, class-weighted batch micro-Dice and optional boundary loss."""

import math

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import nn


def signed_distance_maps(targets: torch.Tensor, max_distance: float) -> torch.Tensor:
    """Return clipped, normalized signed 2D distance maps on the CPU.

    Distances are measured in pixels after augmentation. The inside of a tumor
    is negative, the outside positive, and tumor-free slices contribute zero.
    """
    if not math.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be finite and positive")
    if targets.device.type != "cpu" or targets.ndim != 4 or targets.shape[1] != 1:
        raise ValueError(
            "targets must be a CPU tensor shaped (batch, 1, height, width)"
        )
    masks = targets.detach().numpy()[:, 0] > 0.5
    maps = np.zeros(masks.shape, dtype=np.float32)
    for index, mask in enumerate(masks):
        if not mask.any():
            continue
        # Padding supplies exterior background even if the tumor touches an edge.
        inside = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        outside = (
            distance_transform_edt(~mask) if not mask.all() else np.zeros_like(inside)
        )
        signed = outside - inside
        maps[index] = np.clip(signed, -max_distance, max_distance) / max_distance
    return torch.from_numpy(maps[:, None])


class BoundaryLoss(nn.Module):
    """Mean probability times signed distance, including zero maps for empty slices."""

    def forward(
        self, logits: torch.Tensor, distance_maps: torch.Tensor
    ) -> torch.Tensor:
        if logits.shape != distance_maps.shape:
            raise ValueError("logits and distance maps must have identical shapes")
        return (
            logits.float().sigmoid() * distance_maps.to(logits.device).float()
        ).mean()


class BCEWeightedDiceLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        dice_weight: float,
        background_weight: float,
        foreground_weight: float,
        smooth: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        if (
            pos_weight <= 0
            or dice_weight < 0
            or background_weight <= 0
            or foreground_weight <= 0
            or smooth <= 0
        ):
            raise ValueError("invalid BCE or Dice loss weights/smoothing")
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=device)
        )
        self.dice_weight, self.smooth = dice_weight, smooth
        self.register_buffer(
            "dice_class_weights",
            torch.tensor([background_weight, foreground_weight], device=device),
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce_loss = self.bce(logits, targets)
        targets = targets.float()
        # Keep the original formula while avoiding FP16 overflow in the
        # batch-wide pixel sums when mixed precision is enabled.
        foreground = logits.float().sigmoid()
        probabilities = torch.cat((1.0 - foreground, foreground), dim=1)
        class_targets = torch.cat((1.0 - targets, targets), dim=1)
        reduce_dims = (0, *range(2, probabilities.ndim))
        intersection = (probabilities * class_targets).sum(dim=reduce_dims)
        denominator = probabilities.sum(dim=reduce_dims) + class_targets.sum(
            dim=reduce_dims
        )
        class_dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        valid = class_targets.sum(dim=reduce_dims) > 0
        weights = self.dice_class_weights.to(dtype=class_dice.dtype)
        weighted = (class_dice * valid * weights).sum() / (
            (valid * weights).sum().clamp_min(torch.finfo(class_dice.dtype).eps)
        )
        dice_loss = 1.0 - weighted
        return bce_loss + self.dice_weight * dice_loss, bce_loss, dice_loss
