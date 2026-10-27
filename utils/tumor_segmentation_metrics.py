"""Streaming slice and case overlap metrics for binary tumor segmentation."""

from __future__ import annotations

import math

import torch


def _scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    both_empty = tp + fp + fn == 0
    return {
        "dice": 1.0 if both_empty else 2 * tp / (2 * tp + fp + fn),
        "iou": 1.0 if both_empty else tp / (tp + fp + fn),
        "recall": 1.0 if both_empty else (tp / (tp + fn) if tp + fn else 0.0),
        "precision": 1.0 if both_empty else (tp / (tp + fp) if tp + fp else 0.0),
        "fp": float(fp),
        "fn": float(fn),
    }


class TumorSegmentationMetrics:
    """Accumulate counts rather than retaining full-size masks for an epoch."""

    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []
        self.tumor_dice: list[float] = []
        self.cases: dict[str, list[int]] = {}
        self.total = [0, 0, 0]
        self.negative_count = self.correct_negative_count = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        case_ids: list[str],
        slice_indices: list[int],
    ) -> None:
        if (
            prediction.shape != target.shape
            or prediction.ndim != 4
            or prediction.shape[1] != 1
        ):
            raise ValueError("prediction and target must have shape (B, 1, H, W)")
        if len(case_ids) != len(prediction) or len(slice_indices) != len(prediction):
            raise ValueError("missing case or slice metadata")
        pred, truth = prediction.bool(), target.bool()
        dims = (1, 2, 3)
        tp = (pred & truth).sum(dim=dims).cpu().tolist()
        fp = (pred & ~truth).sum(dim=dims).cpu().tolist()
        fn = (~pred & truth).sum(dim=dims).cpu().tolist()
        for case, t, f, n in zip(case_ids, tp, fp, fn, strict=True):
            self.rows.append(_scores(t, f, n))
            if t + n:
                self.tumor_dice.append(self.rows[-1]["dice"])
            else:
                self.negative_count += 1
                self.correct_negative_count += int(f == 0)
            counts = self.cases.setdefault(str(case), [0, 0, 0])
            for i, value in enumerate((t, f, n)):
                counts[i] += value
                self.total[i] += value

    def compute(self, *, include_cases: bool = False) -> dict[str, float]:
        if not self.rows:
            raise ValueError("no segmentation predictions")
        mean = lambda values: sum(values) / len(values) if values else math.nan
        result = {
            "dice": mean([r["dice"] for r in self.rows]),
            "dice_micro": _scores(*self.total)["dice"],
            "iou_2d": mean([r["iou"] for r in self.rows]),
            "recall_2d": mean([r["recall"] for r in self.rows]),
            "precision_2d": mean([r["precision"] for r in self.rows]),
            "fp_2d": mean([r["fp"] for r in self.rows]),
            "fn_2d": mean([r["fn"] for r in self.rows]),
            "dice_tumor": mean(self.tumor_dice),
            "negative_slice_specificity": self.correct_negative_count
            / self.negative_count
            if self.negative_count
            else math.nan,
        }
        if include_cases:
            scores = [_scores(*counts) for counts in self.cases.values()]
            result.update(
                dice_3d=mean([row["dice"] for row in scores]),
                iou_3d=mean([row["iou"] for row in scores]),
                case_dice_3d_tumor=mean(
                    [
                        _scores(*counts)["dice"]
                        for counts in self.cases.values()
                        if counts[0] + counts[2] > 0
                    ]
                ),
            )
        return result
