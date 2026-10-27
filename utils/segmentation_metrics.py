"""NumPy metrics and volume assembly for binary lung segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


def binary_overlap_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """Return binary Dice and IoU, treating two empty masks as a perfect match."""
    pred, truth = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have identical shapes")
    intersection = int(np.logical_and(pred, truth).sum())
    pred_count, target_count = int(pred.sum()), int(truth.sum())
    union = pred_count + target_count - intersection
    return {
        "dice": (
            1.0
            if not pred_count + target_count
            else 2.0 * intersection / (pred_count + target_count)
        ),
        "iou": 1.0 if not union else intersection / union,
    }


def _surface_points(mask_zyx: np.ndarray, spacing_xyz: Sequence[float]) -> np.ndarray:
    surface = np.asarray(mask_zyx, bool) & ~binary_erosion(mask_zyx, border_value=0)
    return np.argwhere(surface).astype(np.float64) * np.asarray(
        tuple(spacing_xyz)[::-1]
    )


def binary_volume_metrics(
    prediction_zyx: np.ndarray, target_zyx: np.ndarray, spacing_xyz: Sequence[float]
) -> dict[str, float]:
    """Return overlap plus spacing-aware ASSD/HD95 in millimetres for ZYX volumes."""
    metrics = binary_overlap_metrics(prediction_zyx, target_zyx)
    if not np.any(prediction_zyx) or not np.any(target_zyx):
        return metrics | {"assd": float("nan"), "hd95": float("nan")}
    pred_points, target_points = _surface_points(
        prediction_zyx, spacing_xyz
    ), _surface_points(target_zyx, spacing_xyz)
    distances = np.concatenate(
        (
            cKDTree(target_points).query(pred_points)[0],
            cKDTree(pred_points).query(target_points)[0],
        )
    )
    return metrics | {
        "assd": float(distances.mean()),
        "hd95": float(np.percentile(distances, 95)),
    }


@dataclass(frozen=True)
class SliceRecord:
    case_id: str
    slice_index: int
    prediction: np.ndarray
    target: np.ndarray


def aggregate_case_slices(
    records: Sequence[SliceRecord],
) -> dict[str, dict[str, np.ndarray]]:
    """Sort thresholded slices and return contiguous case volumes in ZYX order."""
    grouped: dict[str, list[SliceRecord]] = {}
    for record in records:
        grouped.setdefault(record.case_id, []).append(record)
    volumes: dict[str, dict[str, np.ndarray]] = {}
    for case_id, case_records in grouped.items():
        ordered = sorted(case_records, key=lambda item: item.slice_index)
        indices = [item.slice_index for item in ordered]
        if indices != list(range(len(ordered))):
            raise ValueError(
                f"case {case_id} has non-contiguous slice indices: {indices}"
            )
        volumes[case_id] = {
            "prediction": np.stack([item.prediction.astype(bool) for item in ordered]),
            "target": np.stack([item.target.astype(bool) for item in ordered]),
        }
    return volumes


def finite_mean(values: Sequence[float]) -> float:
    """Mean finite values, or NaN if no values are finite."""
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(finite.mean()) if finite.size else float("nan")
