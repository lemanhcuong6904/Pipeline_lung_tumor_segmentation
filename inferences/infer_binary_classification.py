"""Config-only ResNet-50 inference and slice-level classification evaluation."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from data.binary_classification_dataset import BinaryClassificationDataset
from models.resnet50 import ResNet50
from models.resnet50_pytorch import ResNet50PyTorch
from utils.classification_metrics import BinaryClassificationMetrics
from utils.training import load_checkpoint
from utils.inference_reporting import InferenceProfiler, finite_std, rounded_output
from utils.segmentation_metrics import finite_mean


DATASET = "nsclc_radiomics"  # "nsclc_radiomics" or "nsclc_radiogenomics".
if DATASET not in {"nsclc_radiomics", "nsclc_radiogenomics"}:
    raise ValueError(f"unsupported DATASET: {DATASET}")
DATASET_ROOT = ROOT_DIR / "data" / DATASET
EXPERIMENTS_ROOT = ROOT_DIR / "experiments" if DATASET == "nsclc_radiomics" else ROOT_DIR / "experiments" / DATASET

CFG: dict[str, Any] = {
    "CHECKPOINT_PATH": str(EXPERIMENTS_ROOT / ("resnet50_pytorch_binary_classification" if DATASET == "nsclc_radiomics" else "resnet50_pytorch_binary_classification_weighted_BCELoss") / "best.pt"),
    "DATASET_ROOT": str(DATASET_ROOT),
    "SPLIT_PATH": str(DATASET_ROOT / "config" / "test.txt"),
    "LABELS_CSV": str(DATASET_ROOT / "lung_cropped" / "config" / "test_labels.csv"),
    "OUTPUT_DIR": str((ROOT_DIR / "output" if DATASET == "nsclc_radiomics" else ROOT_DIR / "output" / DATASET) / ("resnet50_pytorch_binary_classification" if DATASET == "nsclc_radiomics" else "resnet50_pytorch_binary_classification_weighted_BCELoss") / "test"),
    "BATCH_SIZE": 16,
    "NUM_WORKERS": 0,
    "PIN_MEMORY": True,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "THRESHOLD": 0.5,
    "PLOT_THRESHOLD_SWEEP": True,
    "MEASURE_INFERENCE": False,
    "WARMUP_BATCHES": 2,
    "USE_K_EXPANSION": False,
    "K_EXPANSION": [2, 4, 6, 8, 10, 12],
}


class ClassificationInferenceDataset(Dataset):
    """Attach case and slice identity to the existing unaugmented dataset."""

    def __init__(self, dataset_root: str | Path, split_path: str | Path, labels_csv: str | Path) -> None:
        self.dataset = BinaryClassificationDataset(dataset_root, split_path, labels_csv=labels_csv)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, target = self.dataset[index]
        case_id, image_path, _, _ = self.dataset.samples[index]
        return {"image": image, "target": target, "case_id": case_id, "slice": image_path.name}


def _metrics_with_counts(metrics: BinaryClassificationMetrics) -> dict[str, float | int]:
    negatives = metrics.tn + metrics.fp
    return {
        **metrics.compute(),
        "specificity": metrics.tn / negatives if negatives else 0.0,
        "tp": metrics.tp, "fp": metrics.fp, "fn": metrics.fn, "tn": metrics.tn,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rounded_output(rows))


def _threshold_sweep(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Compute global slice metrics at 101 thresholds, including both endpoints."""
    results = []
    for step in range(101):
        threshold = step / 100
        metrics = BinaryClassificationMetrics()
        for row in rows:
            prediction = row["probability"] >= threshold
            target = bool(row["target"])
            key = "tp" if prediction and target else "fp" if prediction else "fn" if target else "tn"
            setattr(metrics, key, getattr(metrics, key) + 1)
        results.append({"threshold": threshold, **metrics.compute()})
    return results


def _plot_threshold_sweep(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    sweep = _threshold_sweep(rows)
    with plt.rc_context({"font.family": "serif", "font.serif": ["cmr10"],
                         "font.size": 16, "axes.labelsize": 18,
                         "xtick.labelsize": 15, "ytick.labelsize": 15,
                         "legend.fontsize": 16, "axes.formatter.use_mathtext": True}):
        fig, ax = plt.subplots(figsize=(12, 7))
        try:
            ax.set_facecolor("white")
            thresholds = [point["threshold"] for point in sweep]
            for key, label, color in (("acc", "Accuracy", "#166c88"),
                                      ("precision", "Precision", "#ed704e"),
                                      ("recall", "Recall", "#5a9e68"),
                                      ("f1", "F1-score", "#9c6bb0")):
                values = [point[key] for point in sweep]
                ax.plot(thresholds, values, label=label, color=color, linewidth=2.5)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Probability threshold", ylabel="Score")
            ax.grid(True, color="#b9c9da", alpha=0.4)
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(path, dpi=200)
        finally:
            plt.close(fig)


def _model_from_state(state: dict[str, torch.Tensor]) -> torch.nn.Module:
    """Select the architecture saved by the custom or TorchVision trainer."""
    if any(key.startswith("backbone.") for key in state):
        return ResNet50PyTorch(pretrained=False)
    return ResNet50()


def expand_predictions(rows: list[dict[str, Any]], radius: int) -> list[dict[str, Any]]:
    """Expand positive predictions by original slice index, separately per case."""
    if type(radius) is not int or radius < 0:
        raise ValueError("K_EXPANSION values must be non-negative integers")
    positives: dict[str, list[int]] = {}
    indices = []
    for row in rows:
        stem = Path(str(row["slice"])).stem
        if not stem.isdecimal():
            raise ValueError(f"slice filename must contain a numeric index: {row['slice']}")
        index = int(stem)
        indices.append(index)
        if row["prediction"]:
            positives.setdefault(str(row["case_id"]), []).append(index)
    return [
        dict(row, prediction_raw=int(row["prediction"]), prediction=int(any(
            abs(index - z) <= radius for z in positives.get(str(row["case_id"]), [])
        )))
        for row, index in zip(rows, indices, strict=True)
    ]


def _expanded_metrics(rows: list[dict[str, Any]], threshold: float, radius: int):
    total = BinaryClassificationMetrics()
    cases: dict[str, BinaryClassificationMetrics] = {}
    for row in rows:
        prediction, target = bool(row["prediction"]), bool(row["target"])
        key = "tp" if prediction and target else "fp" if prediction else "fn" if target else "tn"
        case = cases.setdefault(str(row["case_id"]), BinaryClassificationMetrics())
        setattr(case, key, getattr(case, key) + 1)
        setattr(total, key, getattr(total, key) + 1)
    case_rows = [
        {"case_id": case_id, "slice_count": m.tp + m.fp + m.fn + m.tn, **_metrics_with_counts(m)}
        for case_id, m in sorted(cases.items())
    ]
    summary = {"case_count": len(case_rows), "slice_count": len(rows), "threshold": threshold,
               "neighbor_radius": radius, **_metrics_with_counts(total)}
    for key in ("acc", "precision", "recall", "f1", "specificity"):
        values = [row[key] for row in case_rows]
        summary[f"{key}_case_mean"] = finite_mean(values)
        summary[f"{key}_case_std"] = finite_std(values)
    return case_rows, summary


def run_inference(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate lung-containing slices and write per-slice, per-case and overall reports.

    Overall and per-case metrics are computed from slice-level confusion counts.
    """
    threshold = float(cfg["THRESHOLD"])
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not isinstance(cfg.get("USE_K_EXPANSION", False), bool):
        raise ValueError("USE_K_EXPANSION must be True or False")
    radii = cfg.get("K_EXPANSION", []) if cfg.get("USE_K_EXPANSION", False) else []
    if cfg.get("USE_K_EXPANSION", False) and (
        not isinstance(radii, (list, tuple)) or not radii
        or any(type(k) is not int or k < 0 for k in radii) or len(set(radii)) != len(radii)
    ):
        raise ValueError("K_EXPANSION must contain unique non-negative integers")
    checkpoint_path = Path(cfg["CHECKPOINT_PATH"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    device = torch.device(str(cfg["DEVICE"]))
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    model = _model_from_state(checkpoint["model_state"])
    model.load_state_dict(checkpoint["model_state"])
    del checkpoint
    model.to(device).eval()
    profiler = InferenceProfiler(cfg.get("MEASURE_INFERENCE", False), device, cfg.get("WARMUP_BATCHES", 2))

    dataset = ClassificationInferenceDataset(cfg["DATASET_ROOT"], cfg["SPLIT_PATH"], cfg["LABELS_CSV"])
    loader = DataLoader(dataset, batch_size=int(cfg["BATCH_SIZE"]), shuffle=False,
                        num_workers=int(cfg["NUM_WORKERS"]), pin_memory=bool(cfg["PIN_MEMORY"]))
    overall = BinaryClassificationMetrics(threshold)
    by_case: dict[str, BinaryClassificationMetrics] = {}
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference", unit="batch", dynamic_ncols=True):
            images = batch["image"].to(device, non_blocking=True)
            logits = profiler.run("classifier", lambda: model(images), batch_size=images.shape[0]).float().cpu()
            targets = batch["target"]
            overall.update(logits, targets)
            probabilities_tensor = torch.sigmoid(logits).flatten()
            probabilities = probabilities_tensor.tolist()
            predictions = (probabilities_tensor >= threshold).tolist()
            for index, (case_id, name, probability, prediction) in enumerate(
                    zip(batch["case_id"], batch["slice"], probabilities, predictions, strict=True)):
                case_metrics = by_case.setdefault(case_id, BinaryClassificationMetrics(threshold))
                case_metrics.update(logits[index:index + 1], targets[index:index + 1])
                rows.append({"case_id": case_id, "slice": name, "target": int(targets[index].item()),
                             "probability": probability, "prediction": int(prediction)})

    case_rows = [
        {"case_id": case_id, "slice_count": metrics.tp + metrics.fp + metrics.fn + metrics.tn,
         **_metrics_with_counts(metrics)}
        for case_id, metrics in sorted(by_case.items())
    ]
    summary: dict[str, Any] = {"case_count": len(case_rows), "slice_count": len(rows),
                               "threshold": threshold, **_metrics_with_counts(overall)}
    for key in ("acc", "precision", "recall", "f1", "specificity"):
        values = [row[key] for row in case_rows]
        summary[f"{key}_case_mean"] = finite_mean(values)
        summary[f"{key}_case_std"] = finite_std(values)
    if profiler.enabled:
        summary["inference_profile"] = profiler.report()
    output_dir = Path(cfg["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "slice_predictions.csv", rows)
    _write_csv(output_dir / "per_case_metrics.csv", case_rows)
    (output_dir / "mean_metrics.json").write_text(json.dumps(rounded_output(summary), indent=2) + "\n", encoding="utf-8")
    if cfg.get("PLOT_THRESHOLD_SWEEP", False):
        _plot_threshold_sweep(rows, output_dir / "threshold_sweep.png")
    for radius in radii:
        expanded = expand_predictions(rows, radius)
        expanded_cases, expanded_summary = _expanded_metrics(expanded, threshold, radius)
        if profiler.enabled:
            expanded_summary["inference_profile"] = profiler.report()
            expanded_summary["inference_profile"]["neighbor_radius_scope"] = "shared_classifier_forward"
        directory = output_dir / f"k{radius}"
        directory.mkdir(parents=True, exist_ok=True)
        _write_csv(directory / "slice_predictions.csv", expanded)
        _write_csv(directory / "per_case_metrics.csv", expanded_cases)
        (directory / "mean_metrics.json").write_text(
            json.dumps(rounded_output(expanded_summary), indent=2) + "\n", encoding="utf-8"
        )
    return rows, summary


def main() -> None:
    _, summary = run_inference(dict(CFG))
    print(json.dumps(rounded_output(summary), indent=2))


if __name__ == "__main__":
    main()
