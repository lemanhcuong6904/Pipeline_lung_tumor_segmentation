"""Config-only UNet inference and 2D/3D evaluation for lung segmentation."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.lung_segmentation_dataset import LungSegmentationInferenceDataset
from models.unet import UNet
from utils.segmentation_metrics import (
    SliceRecord,
    aggregate_case_slices,
    binary_overlap_metrics,
    binary_volume_metrics,
    finite_mean,
)
from utils.training import load_checkpoint
from utils.inference_reporting import InferenceProfiler, finite_std, rounded_output

DATASET = "nsclc_radiomics"  # "nsclc_radiomics" or "nsclc_radiogenomics".
if DATASET not in {"nsclc_radiomics", "nsclc_radiogenomics"}:
    raise ValueError(f"unsupported DATASET: {DATASET}")
DATASET_ROOT = ROOT_DIR / "data" / DATASET
EXPERIMENTS_ROOT = ROOT_DIR / "experiments" if DATASET == "nsclc_radiomics" else ROOT_DIR / "experiments" / DATASET

CFG: dict[str, Any] = {
    "CHECKPOINT_PATH": str(
        EXPERIMENTS_ROOT / "unet_lung_segmentation" / "best.pt"
    ),
    "DATASET_ROOT": str(DATASET_ROOT),
    "SPLIT_PATH": str(DATASET_ROOT / "config" / "test.txt"),
    "NIFTI_ROOT": str(DATASET_ROOT / "nifti"),
    "OUTPUT_DIR": str((ROOT_DIR / "output" if DATASET == "nsclc_radiomics" else ROOT_DIR / "output" / DATASET) / "unet_lung_segmentation" / "test"),
    "BATCH_SIZE": 16,
    "NUM_WORKERS": 0,
    "PIN_MEMORY": True,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "THRESHOLD": 0.5,
    "MEASURE_INFERENCE": False,
    "WARMUP_BATCHES": 2,
}


def _json_safe(value: Any) -> Any:
    return rounded_output(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_json_safe(rows))


def run_inference(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate the configured checkpoint; only CSV and JSON reports are written."""
    device = torch.device(str(cfg["DEVICE"]))
    checkpoint_path = Path(str(cfg["CHECKPOINT_PATH"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    model = UNet().to(device)
    model.load_state_dict(load_checkpoint(checkpoint_path, device)["model_state"])
    model.eval()
    profiler = InferenceProfiler(
        cfg.get("MEASURE_INFERENCE", False), device, cfg.get("WARMUP_BATCHES", 2)
    )
    dataset = LungSegmentationInferenceDataset(cfg["DATASET_ROOT"], cfg["SPLIT_PATH"])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(cfg["NUM_WORKERS"]),
        pin_memory=bool(cfg["PIN_MEMORY"]),
    )
    records: list[SliceRecord] = []
    slice_metrics: dict[str, list[dict[str, float]]] = {}
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference", unit="batch", dynamic_ncols=True):
            images = batch["image"].to(device, non_blocking=True)
            logits = profiler.run("lung", lambda: model(images), batch_size=images.shape[0])
            predictions = (
                (torch.sigmoid(logits) >= float(cfg["THRESHOLD"])).cpu().numpy()[:, 0]
            )
            targets = batch["mask"].numpy()[:, 0] >= 0.5
            for case_id, index, prediction, target in zip(
                batch["case_id"],
                batch["slice_index"].tolist(),
                predictions,
                targets,
                strict=True,
            ):
                case_id = str(case_id)
                records.append(SliceRecord(case_id, int(index), prediction, target))
                slice_metrics.setdefault(case_id, []).append(
                    binary_overlap_metrics(prediction, target)
                )
    volumes = aggregate_case_slices(records)
    nifti_root = Path(str(cfg["NIFTI_ROOT"]))
    case_rows: list[dict[str, Any]] = []
    for case_id, volume in sorted(volumes.items()):
        reference_path = nifti_root / case_id / "lung_mask.nii.gz"
        reference = sitk.ReadImage(str(reference_path))
        expected_shape = tuple(reversed(reference.GetSize()))
        if volume["prediction"].shape != expected_shape:
            raise ValueError(
                f"volume shape mismatch for {case_id}: {volume['prediction'].shape} vs {expected_shape}"
            )
        spacing = tuple(float(item) for item in reference.GetSpacing())
        metrics_3d = binary_volume_metrics(
            volume["prediction"], volume["target"], spacing
        )
        metrics_2d = {
            f"{key}_2d": finite_mean([item[key] for item in slice_metrics[case_id]])
            for key in ("dice", "iou")
        }
        case_rows.append(
            {
                "case_id": case_id,
                "slice_count": len(slice_metrics[case_id]),
                "spacing_x_mm": spacing[0],
                "spacing_y_mm": spacing[1],
                "spacing_z_mm": spacing[2],
                "pred_foreground_voxels": int(volume["prediction"].sum()),
                "target_foreground_voxels": int(volume["target"].sum()),
                **metrics_2d,
                **{f"{key}_3d": value for key, value in metrics_3d.items()},
            }
        )
    all_slice_metrics = [
        metric for values in slice_metrics.values() for metric in values
    ]
    summary: dict[str, Any] = {
        "case_count": len(case_rows),
        "slice_count": len(records),
        "threshold": float(cfg["THRESHOLD"]),
    }
    for key in ("dice", "iou"):
        values = [item[key] for item in all_slice_metrics]
        summary[f"{key}_2d"] = finite_mean(values)
        summary[f"{key}_2d_std"] = finite_std(values)
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
    _write_csv(output_dir / "per_case_metrics.csv", case_rows)
    (output_dir / "mean_metrics.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    return case_rows, summary


def main() -> None:
    _, summary = run_inference(dict(CFG))
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
