"""Config-only TransUNet inference and 2D/3D tumor segmentation evaluation."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import SimpleITK as sitk
import torch
from tqdm.auto import tqdm

from data.tumor_segmentation_dataset import ABLATION_MODES, create_dataloader
from models.transunet import SingleWindowTransUNet, TransUNet
from utils.segmentation_metrics import (
    binary_overlap_metrics,
    binary_volume_metrics,
    finite_mean,
)
from utils.training import load_checkpoint
from utils.inference_reporting import InferenceProfiler, finite_std, rounded_output

DATASET = "nsclc_radiogenomics"  # "nsclc_radiomics" or "nsclc_radiogenomics".
if DATASET not in {"nsclc_radiomics", "nsclc_radiogenomics"}:
    raise ValueError(f"unsupported DATASET: {DATASET}")
DATASET_ROOT = ROOT_DIR / "data" / DATASET
EXPERIMENTS_ROOT = ROOT_DIR / "experiments" if DATASET == "nsclc_radiomics" else ROOT_DIR / "experiments" / DATASET

CFG: dict[str, Any] = {
    # auto: from checkpoint; baseline: no crop; lung_roi: cropped, 1 window;
    # multi_window: cropped, 3 windows.
    "ABLATION": "auto",
    "CHECKPOINT_PATH": str(
        EXPERIMENTS_ROOT
        / "nsclc_radiogenomics_transunet_baseline_wide" #("transunet_multi_window_tumor_channel_attention" if DATASET == "nsclc_radiomics" else "transunet_lung_roi_wide")
        / "best.pt"
    ),
    "DATASET_ROOT": str(DATASET_ROOT),
    "SPLIT_PATH": str(DATASET_ROOT / "config" / "test.txt"),
    "FULL_NIFTI_ROOT": str(DATASET_ROOT / "nifti"),
    "CROPPED_NIFTI_ROOT": str(
        DATASET_ROOT / "lung_cropped" / "nifti"
    ),
    "OUTPUT_DIR": str(
        (ROOT_DIR / "output" if DATASET == "nsclc_radiomics" else ROOT_DIR / "output" / DATASET)
        / "nsclc_radiogenomics_transunet_baseline_wide"
        / "test"
    ),
    "BATCH_SIZE": 8,
    "NUM_WORKERS": 0,
    "PIN_MEMORY": True,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "THRESHOLD": 0.5,
    "PLOT_THRESHOLD_SWEEP": False,  # Plot mean per-case Dice and IoU across thresholds.
    "MEASURE_INFERENCE": True,
    "WARMUP_BATCHES": 5,
}


def _json_safe(value: Any) -> Any:
    return rounded_output(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_json_safe(rows))



SWEEP_THRESHOLDS = np.arange(101, dtype=np.float32) / np.float32(100)


def _threshold_sweep(cases: dict[str, dict[str, Any]]) -> list[dict[str, float]]:
    """Mean per-case 3D overlaps; unprocessed slices stay negative."""
    dice_values, iou_values = [], []
    for case in cases.values():
        pred = np.cumsum(case["sweep_predictions"][:0:-1])[::-1]
        intersection = np.cumsum(case["sweep_intersections"][:0:-1])[::-1]
        truth = int(case["target"].sum())
        denominator = pred + truth
        union = denominator - intersection
        dice_values.append(np.divide(2 * intersection, denominator,
                                     out=np.ones(101, dtype=float), where=denominator != 0))
        iou_values.append(np.divide(intersection, union,
                                    out=np.ones(101, dtype=float), where=union != 0))
    dice_mean = np.mean(dice_values, axis=0)
    iou_mean = np.mean(iou_values, axis=0)
    return [{"threshold": i / 100, "dice_3d": float(dice_mean[i]),
             "iou_3d": float(iou_mean[i])} for i in range(101)]


def _plot_threshold_sweep(rows: list[dict[str, float]], path: Path) -> None:
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "serif", "font.serif": ["cmr10"],
                         "font.size": 16, "axes.labelsize": 18,
                         "xtick.labelsize": 15, "ytick.labelsize": 15,
                         "legend.fontsize": 16, "axes.formatter.use_mathtext": True}):
        fig, ax = plt.subplots(figsize=(12, 7))
        try:
            thresholds = [row["threshold"] for row in rows]
            for key, label, color in (("dice_3d", "Dice 3D", "#166c88"),
                                      ("iou_3d", "IoU 3D", "#ed704e")):
                ax.plot(thresholds, [row[key] for row in rows], label=label,
                        color=color, linewidth=2.5)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Probability threshold",
                   ylabel="Mean per-case score")
            ax.grid(True, color="#b9c9da", alpha=0.4)
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(path, dpi=200)
        finally:
            plt.close(fig)

def _case_reference(
    nifti_root: Path, case_id: str, *, cropped: bool = True
) -> dict[str, Any]:
    case_root = nifti_root / case_id
    reference = sitk.ReadImage(str(case_root / "tumor_mask.nii.gz"))
    if reference.GetDimension() != 3:
        raise ValueError(f"expected a 3D tumor mask for {case_id}")
    spacing = tuple(float(value) for value in reference.GetSpacing())
    if any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError(f"invalid spacing for {case_id}")
    start = 0
    if cropped:
        metadata = json.loads((case_root / "crop.json").read_text(encoding="utf-8"))
        start = metadata["source_index_xyz"][2]
        depth = metadata["source_size_xyz"][2]
        if not isinstance(start, int) or start < 0 or depth != reference.GetSize()[2]:
            raise ValueError(f"invalid crop metadata for {case_id}")
    target = sitk.GetArrayFromImage(reference) > 0
    return {
        "target": target,
        "prediction": np.zeros_like(target),
        "spacing": spacing,
        "start": start,
        "seen": set(),
        "slices": [],
    }


def checkpoint_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Recover optional attention flags from weights in older checkpoints."""
    if not isinstance(checkpoint.get("model_config"), dict):
        raise ValueError("checkpoint does not contain model_config")  # noqa: TRY004
    model_cfg = dict(checkpoint["model_config"])
    state = checkpoint["model_state"]
    if model_cfg.get("in_channels", 3) == 1:
        model_cfg["backbone_pretrained"] = False
        return model_cfg
    model_cfg.setdefault(
        "cross_attention_enabled",
        any(k.startswith("cross_window_attention.") for k in state),
    )
    had_window_attention = model_cfg.pop("window_attention_enabled", False)
    model_cfg.pop("window_attention_reduction", None)
    if had_window_attention or any(
        k.startswith("window_channel_attention.") for k in state
    ):
        raise ValueError(
            "checkpoint uses removed window attention and cannot be loaded"
        )
    model_cfg.setdefault(
        "channel_attention_enabled",
        any(k.startswith("channel_attention.") for k in state),
    )
    model_cfg["backbone_pretrained"] = False
    return model_cfg


def run_inference(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    threshold = float(cfg["THRESHOLD"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    checkpoint_path = Path(str(cfg["CHECKPOINT_PATH"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    model_cfg = checkpoint_model_config(checkpoint)
    channels = int(model_cfg.get("in_channels", 3))
    requested_mode = str(cfg.get("ABLATION", "auto"))
    checkpoint_mode = checkpoint.get("config", {}).get("ABLATION")
    if requested_mode == "auto":
        mode = checkpoint_mode or ("multi_window" if channels == 3 else None)
        if mode is None:
            raise ValueError(
                "1-channel checkpoint has no ABLATION; set CFG['ABLATION'] explicitly"
            )
    else:
        mode = requested_mode
        if checkpoint_mode and mode != checkpoint_mode:
            raise ValueError("ABLATION does not match the checkpoint training mode")
    if mode not in ABLATION_MODES or channels != (3 if mode == "multi_window" else 1):
        raise ValueError("invalid ABLATION or mismatch with checkpoint input channels")
    model = (TransUNet if channels == 3 else SingleWindowTransUNet)(**model_cfg)
    model.load_state_dict(checkpoint["model_state"])
    del checkpoint
    device = torch.device(str(cfg["DEVICE"]))
    model = model.to(device).eval()
    profiler = InferenceProfiler(cfg.get("MEASURE_INFERENCE", False), device, cfg.get("WARMUP_BATCHES", 2))
    loader = create_dataloader(
        cfg["DATASET_ROOT"],
        cfg["SPLIT_PATH"],
        mode=mode,
        batch_size=int(cfg["BATCH_SIZE"]),
        num_workers=int(cfg["NUM_WORKERS"]),
        pin_memory=bool(cfg["PIN_MEMORY"]),
        shuffle=False,
        return_metadata=True,
    )
    nifti_root = (
        Path(str(cfg["FULL_NIFTI_ROOT"]))
        if mode == "baseline"
        else Path(str(cfg["CROPPED_NIFTI_ROOT"]))
    )
    sweep_enabled = bool(cfg.get("PLOT_THRESHOLD_SWEEP", False))
    cases: dict[str, dict[str, Any]] = {}
    slice_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference", unit="batch", dynamic_ncols=True):
            images = batch["image"].to(device, non_blocking=True)
            logits = profiler.run("tumor", lambda: model(images), batch_size=images.shape[0])
            if (
                logits.ndim != 4
                or logits.shape[1] != 1
                or logits.shape != batch["mask"].shape
            ):
                raise ValueError(
                    "model output and masks must have matching shape (B, 1, H, W)"
                )
            probabilities = torch.sigmoid(logits).cpu().numpy()[:, 0]
            predictions = probabilities >= threshold
            targets = batch["mask"].numpy()[:, 0] >= 0.5
            for case_id, source_index, prediction, target, probability in zip(
                batch["case_id"],
                batch["slice_index"].tolist(),
                predictions,
                targets,
                probabilities,
                strict=True,
            ):
                case_id, source_index = str(case_id), int(source_index)
                if case_id not in cases:
                    cases[case_id] = _case_reference(
                        nifti_root, case_id, cropped=mode != "baseline"
                    )
                case = cases[case_id]
                if sweep_enabled and "sweep_predictions" not in case:
                    case["sweep_predictions"] = np.zeros(102, dtype=np.int64)
                    case["sweep_intersections"] = np.zeros(102, dtype=np.int64)
                z = source_index - case["start"]
                if not 0 <= z < case["target"].shape[0] or z in case["seen"]:
                    raise ValueError(
                        f"invalid or duplicate slice index {source_index} for {case_id}"
                    )
                if target.shape != case["target"].shape[1:] or not np.array_equal(
                    target, case["target"][z]
                ):
                    raise ValueError(
                        f"PNG tumor mask does not match NIfTI for {case_id}/{source_index:04d}"
                    )
                case["seen"].add(z)
                case["prediction"][z] = prediction
                if sweep_enabled:
                    bins = np.searchsorted(SWEEP_THRESHOLDS, probability, side="right")
                    case["sweep_predictions"] += np.bincount(bins.ravel(), minlength=102)
                    case["sweep_intersections"] += np.bincount(bins[target], minlength=102)
                scores = binary_overlap_metrics(prediction, target)
                row = {
                    "case_id": case_id,
                    "slice_index": source_index,
                    "target_foreground_pixels": int(target.sum()),
                    "pred_foreground_pixels": int(prediction.sum()),
                    **scores,
                }
                case["slices"].append(row)
                slice_rows.append(row)
    expected_cases = {
        line.strip()
        for line in Path(cfg["SPLIT_PATH"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if set(cases) != expected_cases:
        raise ValueError(
            f"missing cases or unexpected cases in inference: {sorted(expected_cases ^ set(cases))}"
        )
    case_rows: list[dict[str, Any]] = []
    for case_id, case in sorted(cases.items()):
        spacing = case["spacing"]
        scores_3d = binary_volume_metrics(case["prediction"], case["target"], spacing)
        case_rows.append(
            {
                "case_id": case_id,
                "slice_count": len(case["slices"]),
                "volume_slice_count": case["target"].shape[0],
                "spacing_x_mm": spacing[0],
                "spacing_y_mm": spacing[1],
                "spacing_z_mm": spacing[2],
                "pred_foreground_voxels": int(case["prediction"].sum()),
                "target_foreground_voxels": int(case["target"].sum()),
                **{
                    f"{key}_2d": finite_mean([row[key] for row in case["slices"]])
                    for key in ("dice", "iou")
                },
                **{f"{key}_3d": value for key, value in scores_3d.items()},
            }
        )
    summary: dict[str, Any] = {
        "case_count": len(case_rows),
        "slice_count": len(slice_rows),
        "threshold": threshold,
    }
    for key in ("dice", "iou"):
        all_values = [row[key] for row in slice_rows]
        summary[f"{key}_2d"] = finite_mean(all_values)
        summary[f"{key}_2d_std"] = finite_std(all_values)
        positive = [
            row[key] for row in slice_rows if row["target_foreground_pixels"] > 0
        ]
        summary[f"{key}_2d_tumor"] = finite_mean(positive)
        summary[f"{key}_2d_tumor_std"] = finite_std(positive)
    for key in ("dice", "iou", "assd", "hd95"):
        values = [float(row[f"{key}_3d"]) for row in case_rows]
        summary[f"{key}_3d"] = finite_mean(values)
        summary[f"{key}_3d_std"] = finite_std(values)
        if key in {"assd", "hd95"}:
            summary[f"{key}_3d_valid_cases"] = sum(np.isfinite(values))
    if profiler.enabled:
        summary["inference_profile"] = profiler.report()
    output_dir = Path(str(cfg["OUTPUT_DIR"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_slice_metrics.csv", slice_rows)
    _write_csv(output_dir / "per_case_metrics.csv", case_rows)
    (output_dir / "mean_metrics.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    if sweep_enabled:
        sweep = _threshold_sweep(cases)
        _write_csv(output_dir / "threshold_sweep.csv", sweep)
        _plot_threshold_sweep(sweep, output_dir / "threshold_sweep.png")
    return case_rows, summary


def main() -> None:
    _, summary = run_inference(dict(CFG))
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
