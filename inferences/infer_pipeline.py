"""Evaluate lung-ROI, classifier and multi-window pipeline ablations.

Uses predicted lung masks and reports segmentation metrics on the full CT grid.
No ground-truth mask is used to determine the crop or select slices.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from tqdm.auto import tqdm

from inferences.infer_tumor_segmentation import (
    _json_safe,
    _write_csv,
    checkpoint_model_config,
)
from models.transunet import SingleWindowTransUNet, TransUNet
from models.unet import UNet
from scripts.crop_nsclc_lungs import (
    clean_lung_mask,
    _cleaned_lung_crop_region,
    crop_and_resize,
)
from utils.segmentation_metrics import (
    binary_overlap_metrics,
    binary_volume_metrics,
    finite_mean,
)
from utils.training import load_checkpoint
from utils.inference_reporting import InferenceProfiler, finite_std

DATASET = "nsclc_radiogenomics"  # "nsclc_radiomics" or "nsclc_radiogenomics".
if DATASET not in {"nsclc_radiomics", "nsclc_radiogenomics"}:
    raise ValueError(f"unsupported DATASET: {DATASET}")
DATASET_ROOT = ROOT_DIR / "data" / DATASET
EXPERIMENTS_ROOT = ROOT_DIR / "experiments" if DATASET == "nsclc_radiomics" else ROOT_DIR / "experiments" / DATASET

CFG: dict[str, Any] = {
    "USE_CLASSIFIER": True,  # ResNet50 selects slices for tumor segmentation
    "USE_MULTI_WINDOW": True,  # False: wide only; True: wide, lung, mediastinal
    "USE_NEIGHBOR_EXPANSION": False,  # Requires USE_CLASSIFIER
    "NEIGHBOR_RADII": [2, 4, 6, 8, 10, 12],  # One report per radius when enabled
    "DATASET_ROOT": str(DATASET_ROOT),
    "SPLIT_PATH": str(DATASET_ROOT / "config" / "test.txt"),
    "OUTPUT_ROOT": str((ROOT_DIR / "output" if DATASET == "nsclc_radiomics" else ROOT_DIR / "output" / DATASET) / "pipeline" / "test"),
    "EXP_NAME": "lungSeg_resnet50_transUnet3",  # Empty: name based on flags; otherwise use this folder name.
    "LUNG_CHECKPOINT": r"experiments\unet_lung_segmentation\best.pt",
    "CLASSIFIER_CHECKPOINT": r"experiments\nsclc_radiogenomics\nsclc_radiogenomics_resnet50_pytorch_binary_classification\best.pt",
    "SINGLE_WINDOW_CHECKPOINT": str(
        EXPERIMENTS_ROOT / "nsclc_radiogenomics_transunet_lung_roi_wide" / "best.pt"
    ),
    "MULTI_WINDOW_CHECKPOINT": str(
        EXPERIMENTS_ROOT
        / ("transunet_multi_window_tumor_channel_attention" if DATASET == "nsclc_radiomics" else "nsclc_radiogenomics_transunet_multi_window_tumor_channel_attention")
        / "best.pt"
    ),
    "LUNG_THRESHOLD": 0.5,
    "CLASSIFIER_THRESHOLD": 0.5,
    "TUMOR_THRESHOLD": 0.5,
    "LUNG_MARGIN_MM": 10.0,
    "MIN_COMPONENT_FRACTION": 0.01,
    "BATCH_SIZE": 8,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "AMP_DTYPE": "none",  # none | bf16 | fp16 (CUDA only)
    "MEASURE_INFERENCE": True,
    "WARMUP_BATCHES": 5,
}


def select_slices(candidates: list[int], positives: set[int], radius: int) -> set[int]:
    """Expand using source CT indices, intersecting predicted-lung slices."""
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
        raise ValueError("neighbor radius must be a non-negative integer")
    return {
        z
        for z in candidates
        if any(abs(z - positive) <= radius for positive in positives)
    }


def _checkpoint(path: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return load_checkpoint(path, torch.device("cpu"))


def load_models(cfg: dict[str, Any], device: torch.device):
    lung = UNet()
    lung.load_state_dict(_checkpoint(cfg["LUNG_CHECKPOINT"])["model_state"])
    classifier = None
    if cfg["USE_CLASSIFIER"]:
        checkpoint = _checkpoint(cfg["CLASSIFIER_CHECKPOINT"])
        state = checkpoint["model_state"]
        if any(key.startswith("backbone.") for key in state):
            from models.resnet50_pytorch import ResNet50PyTorch

            classifier = ResNet50PyTorch(pretrained=False)
        else:
            from models.resnet50 import ResNet50

            classifier = ResNet50()
        classifier.load_state_dict(state)
    multi = cfg["USE_MULTI_WINDOW"]
    checkpoint = _checkpoint(
        cfg["MULTI_WINDOW_CHECKPOINT"] if multi else cfg["SINGLE_WINDOW_CHECKPOINT"]
    )
    model_cfg = checkpoint_model_config(checkpoint)
    expected_mode, channels = ("multi_window", 3) if multi else ("lung_roi", 1)
    mode = checkpoint.get("config", {}).get("ABLATION")
    if model_cfg.get("in_channels", 3) != channels or (mode and mode != expected_mode):
        raise ValueError(
            f"tumor checkpoint must be trained for {expected_mode} with {channels} channels"
        )
    tumor = (TransUNet if multi else SingleWindowTransUNet)(**model_cfg)
    tumor.load_state_dict(checkpoint["model_state"])
    return tuple(
        model.to(device).eval() if model is not None else None
        for model in (lung, classifier, tumor)
    )


def _read_windows(
    root: Path, case_id: str, reference: sitk.Image, multi: bool
) -> list[sitk.Image]:
    directories = (
        ("images", "images_lung", "images_mediastinal") if multi else ("images",)
    )
    depth, height, width = reversed(reference.GetSize())
    result = []
    for directory in directories:
        folder = root / "processed" / case_id / directory
        files = sorted(folder.glob("*.png"))
        expected = [f"{z:04d}.png" for z in range(depth)]
        if [p.name for p in files] != expected:
            raise ValueError(f"missing or unexpected full CT slices in {folder}")
        arrays = []
        for path in files:
            with Image.open(path) as image:
                arrays.append(np.asarray(image.convert("L"), dtype=np.uint8))
        array = np.stack(arrays)
        if array.shape != (depth, height, width):
            raise ValueError(
                f"slice shape does not match NIfTI for {case_id}/{directory}"
            )
        volume = sitk.GetImageFromArray(array)
        volume.CopyInformation(reference)
        result.append(volume)
    return result


def _predict(
    model: torch.nn.Module,
    array: np.ndarray,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype | None,
    profiler: InferenceProfiler,
    stage: str,
) -> torch.Tensor:
    results = []
    for start in range(0, len(array), batch_size):
        batch = (
            torch.from_numpy(np.ascontiguousarray(array[start : start + batch_size]))
            .float()
            .to(device)
            / 255.0
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, enabled=dtype is not None, dtype=dtype
        ):
            logits = profiler.run(stage, lambda: model(batch), batch_size=batch.shape[0])
        results.append(torch.sigmoid(logits).float().cpu())
    return torch.cat(results) if results else torch.empty(0)


def run_pipeline(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    exp_name = cfg.get("EXP_NAME", "")
    if not isinstance(exp_name, str) or (exp_name and (
        exp_name in {".", ".."} or Path(exp_name).name != exp_name
        or PureWindowsPath(exp_name).name != exp_name
    )):
        raise ValueError("EXP_NAME must be a folder name, not a path")
    for key in ("USE_CLASSIFIER", "USE_MULTI_WINDOW", "USE_NEIGHBOR_EXPANSION"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be True or False")
    if cfg["USE_NEIGHBOR_EXPANSION"] and not cfg["USE_CLASSIFIER"]:
        raise ValueError("USE_NEIGHBOR_EXPANSION requires USE_CLASSIFIER")
    for key in ("LUNG_THRESHOLD", "CLASSIFIER_THRESHOLD", "TUMOR_THRESHOLD"):
        value = float(cfg[key])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    radii = cfg["NEIGHBOR_RADII"] if cfg["USE_NEIGHBOR_EXPANSION"] else [0]
    if (
        not isinstance(radii, (list, tuple))
        or not radii
        or len(set(radii)) != len(radii)
    ):
        raise ValueError("NEIGHBOR_RADII must be a non-empty list of unique radii")
    for radius in radii:
        select_slices([], set(), radius)
    if cfg.get("MEASURE_INFERENCE", False) and cfg["USE_NEIGHBOR_EXPANSION"] and len(radii) > 1:
        results = {}
        for radius in radii:
            print(f"Measuring pipeline separately for neighbor radius k={radius}")
            results.update(run_pipeline({**cfg, "NEIGHBOR_RADII": [radius]}))
        return results
    if int(cfg["BATCH_SIZE"]) < 1:
        raise ValueError("BATCH_SIZE must be positive")
    device = torch.device(str(cfg["DEVICE"]))
    dtypes = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}
    if cfg["AMP_DTYPE"] not in dtypes:
        raise ValueError("AMP_DTYPE must be none, bf16 or fp16")
    dtype = dtypes[cfg["AMP_DTYPE"]]
    if dtype is not None and device.type != "cuda":
        raise ValueError("AMP_DTYPE requires CUDA")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support bf16")
    case_ids = [
        s.strip()
        for s in Path(cfg["SPLIT_PATH"]).read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("split must contain unique case IDs")
    if any(Path(case).name != case or case in {".", ".."} for case in case_ids):
        raise ValueError("invalid case ID")
    lung_model, classifier, tumor_model = load_models(cfg, device)
    profiler = InferenceProfiler(cfg.get("MEASURE_INFERENCE", False), device, cfg.get("WARMUP_BATCHES", 2))
    name = exp_name
    if not name:
        name = "lung_roi_multi" if cfg["USE_MULTI_WINDOW"] else "lung_roi_wide"
        if cfg["USE_CLASSIFIER"]:
            name += "_classifier"
    names = {
        k: f"{name}_k{k}" if cfg["USE_NEIGHBOR_EXPANSION"] else name for k in radii
    }
    case_rows = {k: [] for k in radii}
    slice_rows = {k: [] for k in radii}
    root = Path(cfg["DATASET_ROOT"])
    for case_id in tqdm(case_ids, desc="Pipeline", unit="case", dynamic_ncols=True):
        reference = sitk.ReadImage(str(root / "nifti" / case_id / "image.nii.gz"))
        target_image = sitk.ReadImage(
            str(root / "nifti" / case_id / "tumor_mask.nii.gz")
        )
        if reference.GetDimension() != 3 or not reference.IsSameImageGeometryAs(
            target_image
        ):
            raise ValueError(f"image and tumor mask geometry mismatch for {case_id}")
        windows = _read_windows(root, case_id, reference, cfg["USE_MULTI_WINDOW"])
        target = sitk.GetArrayFromImage(target_image) > 0
        wide = sitk.GetArrayFromImage(windows[0])
        lung_prob = _predict(
            lung_model, wide[:, None], int(cfg["BATCH_SIZE"]), device, dtype, profiler, "lung"
        )
        if lung_prob.shape != (len(wide), 1, *wide.shape[1:]):
            raise ValueError("lung model must return one mask per CT slice")
        lung_mask = (lung_prob.numpy()[:, 0] >= float(cfg["LUNG_THRESHOLD"])).astype(
            np.uint8
        )
        selected: dict[int, set[int]] = {k: set() for k in radii}
        probabilities: dict[int, float] = {}
        candidate_indices: list[int] = []
        roi_index = roi_size = None
        prediction_by_z: dict[int, np.ndarray] = {}
        roi_reference = None
        if np.any(lung_mask):
            mask_image = sitk.GetImageFromArray(lung_mask)
            mask_image.CopyInformation(reference)
            cleaned = clean_lung_mask(mask_image, float(cfg["MIN_COMPONENT_FRACTION"]))
            roi_index, roi_size = _cleaned_lung_crop_region(
                cleaned, float(cfg["LUNG_MARGIN_MM"])
            )
            roi_windows = [
                crop_and_resize(window, roi_index, roi_size, is_label=False)
                for window in windows
            ]
            roi_reference = roi_windows[0]
            roi_lung = (
                sitk.GetArrayFromImage(
                    crop_and_resize(cleaned, roi_index, roi_size, is_label=True)
                )
                > 0
            )
            candidate_indices = [
                roi_index[2] + z for z in range(roi_size[2]) if roi_lung[z].any()
            ]
            roi_arrays = np.stack(
                [sitk.GetArrayFromImage(window) for window in roi_windows], axis=1
            )
            if classifier is None:
                selected = {k: set(candidate_indices) for k in radii}
            elif candidate_indices:
                local = [z - roi_index[2] for z in candidate_indices]
                cls = _predict(
                    classifier,
                    roi_arrays[local, :1],
                    int(cfg["BATCH_SIZE"]),
                    device,
                    dtype,
                    profiler,
                    "classifier",
                )
                if cls.numel() != len(candidate_indices):
                    raise ValueError("classifier must return one logit per slice")
                probabilities = dict(
                    zip(candidate_indices, cls.reshape(-1).tolist(), strict=True)
                )
                if any(not math.isfinite(p) for p in probabilities.values()):
                    raise ValueError("classifier returned a non-finite probability")
                positives = {
                    z
                    for z, p in probabilities.items()
                    if p >= float(cfg["CLASSIFIER_THRESHOLD"])
                }
                selected = {
                    k: select_slices(candidate_indices, positives, k) for k in radii
                }
            union = sorted(set().union(*selected.values()))
            if union:
                predictions = _predict(
                    tumor_model,
                    roi_arrays[[z - roi_index[2] for z in union]],
                    int(cfg["BATCH_SIZE"]),
                    device,
                    dtype,
                    profiler,
                    "tumor",
                )
                if predictions.shape != (len(union), 1, 256, 256):
                    raise ValueError(
                        "tumor model must return one 256x256 mask per slice"
                    )
                prediction_by_z = {
                    z: (mask[0].numpy() >= float(cfg["TUMOR_THRESHOLD"]))
                    for z, mask in zip(union, predictions, strict=True)
                }
        for radius in radii:
            if roi_reference is None:
                prediction = np.zeros_like(target)
            else:
                roi_prediction = np.zeros((roi_size[2], 256, 256), dtype=np.uint8)
                for z in selected[radius]:
                    roi_prediction[z - roi_index[2]] = prediction_by_z[z]
                roi_image = sitk.GetImageFromArray(roi_prediction)
                roi_image.CopyInformation(roi_reference)
                full_image = sitk.Resample(
                    roi_image,
                    reference,
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor,
                    0,
                    sitk.sitkUInt8,
                )
                prediction = sitk.GetArrayFromImage(full_image) > 0
            scores = binary_volume_metrics(prediction, target, reference.GetSpacing())
            case_rows[radius].append(
                {
                    "case_id": case_id,
                    "lung_slices": len(candidate_indices),
                    "classifier_positive_slices": (
                        sum(
                            p >= float(cfg["CLASSIFIER_THRESHOLD"])
                            for p in probabilities.values()
                        )
                        if classifier is not None
                        else None
                    ),
                    "selected_slices": len(selected[radius]),
                    "volume_slice_count": len(target),
                    "target_foreground_voxels": int(target.sum()),
                    "pred_foreground_voxels": int(prediction.sum()),
                    **{f"{key}_3d": value for key, value in scores.items()},
                }
            )
            for z in range(len(target)):
                scores_2d = binary_overlap_metrics(prediction[z], target[z])
                slice_rows[radius].append(
                    {
                        "case_id": case_id,
                        "slice_index": z,
                        "lung_present": z in candidate_indices,
                        "classifier_probability": probabilities.get(z),
                        "selected_for_segmentation": z in selected[radius],
                        "target_foreground_pixels": int(target[z].sum()),
                        "pred_foreground_pixels": int(prediction[z].sum()),
                        **scores_2d,
                    }
                )
    results = {}
    for radius in radii:
        cases, slices = case_rows[radius], slice_rows[radius]
        summary = {
            "neighbor_radius": radius if cfg["USE_NEIGHBOR_EXPANSION"] else None,
            "case_count": len(cases),
            "slice_count": len(slices),
            "lung_slices": sum(row["lung_slices"] for row in cases),
            "selected_slices": sum(row["selected_slices"] for row in cases),
            "classification_enabled": cfg["USE_CLASSIFIER"],
            "multi_window_enabled": cfg["USE_MULTI_WINDOW"],
            "neighbor_expansion_enabled": cfg["USE_NEIGHBOR_EXPANSION"],
            "evaluation_grid": "full_ct",
            "config": dict(cfg),
        }
        for key in ("dice", "iou"):
            values = [row[key] for row in slices]
            positive = [row[key] for row in slices if row["target_foreground_pixels"] > 0]
            summary[f"{key}_2d"] = finite_mean(values)
            summary[f"{key}_2d_std"] = finite_std(values)
            summary[f"{key}_2d_tumor"] = finite_mean(positive)
            summary[f"{key}_2d_tumor_std"] = finite_std(positive)
        for key in ("dice", "iou", "assd", "hd95"):
            values = [row[f"{key}_3d"] for row in cases]
            summary[f"{key}_3d"] = finite_mean(values)
            summary[f"{key}_3d_std"] = finite_std(values)
            if key in ("assd", "hd95"):
                summary[f"{key}_3d_valid_cases"] = sum(np.isfinite(values))
        if profiler.enabled:
            summary["inference_profile"] = profiler.report()
        out = Path(cfg["OUTPUT_ROOT"]) / names[radius]
        out.mkdir(parents=True, exist_ok=True)
        _write_csv(out / "per_case_metrics.csv", cases)
        _write_csv(out / "per_slice_metrics.csv", slices)
        (out / "mean_metrics.json").write_text(
            json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
        )
        results[names[radius]] = summary
    return results


def main() -> None:
    print(json.dumps(_json_safe(run_pipeline(dict(CFG))), indent=2))


if __name__ == "__main__":
    main()
