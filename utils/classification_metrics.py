"""Confusion-matrix metrics for binary classification from logits."""

import torch


class BinaryClassificationMetrics:
    """Accumulate counts over batches; undefined ratios are reported as zero."""

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have the same shape")
        prediction = torch.sigmoid(logits.detach().float()) >= self.threshold
        target = targets.detach() >= 0.5
        self.tp += int((prediction & target).sum().item())
        self.fp += int((prediction & ~target).sum().item())
        self.fn += int((~prediction & target).sum().item())
        self.tn += int((~prediction & ~target).sum().item())

    def compute(self) -> dict[str, float]:
        total = self.tp + self.fp + self.fn + self.tn
        precision_denominator = self.tp + self.fp
        recall_denominator = self.tp + self.fn
        f1_denominator = 2 * self.tp + self.fp + self.fn
        return {
            "acc": (self.tp + self.tn) / total if total else 0.0,
            "precision": self.tp / precision_denominator if precision_denominator else 0.0,
            "recall": self.tp / recall_denominator if recall_denominator else 0.0,
            "f1": 2 * self.tp / f1_denominator if f1_denominator else 0.0,
        }
