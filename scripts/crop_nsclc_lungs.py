"""Create lung-cropped NSCLC Radiomics or Radiogenomics NIfTI volumes and paired PNG slices.

The source is data/<selected dataset>/nifti. PNG filenames retain the source Z
indices; crop.json records the mapping to the zero-based output NIfTI volume.
"""

from __future__ import annotations

import json
import math
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

DATASET = "nsclc_radiogenomics"  # "nsclc_radiomics" or "nsclc_radiogenomics".
DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / DATASET
CASE_IDS: list[str] | None = None  # None processes every case; otherwise list case IDs.
MARGIN_MM = 10.0  # In-plane margin; Z is cropped to the lung extent.
MIN_COMPONENT_FRACTION = 0.01  # Fraction of the largest 3D component.
SKIP_EXISTING = False  # Skip cases with both output directories already present.
OVERWRITE = True  # Replace existing outputs; cannot be combined with SKIP_EXISTING.
MIN_FREE_GB = 1.0  # Stop before the next case if less disk space is available.


def clean_lung_mask(mask: sitk.Image, min_component_fraction: float = MIN_COMPONENT_FRACTION) -> sitk.Image:
    """Remove 3D islands below 1% of the largest component and single voxels."""
    if mask.GetDimension() != 3:
        raise ValueError("mask must be 3D")
    if not math.isfinite(min_component_fraction) or not 0 <= min_component_fraction <= 1:
        raise ValueError("min_component_fraction must be between 0 and 1")
    labels = sitk.ConnectedComponent(sitk.Cast(mask > 0, sitk.sitkUInt8), True)
    ordered = sitk.RelabelComponent(labels, sortByObjectSize=True)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(ordered)
    if not stats.HasLabel(1):
        raise ValueError("lung mask is empty")
    minimum = max(2, math.ceil(stats.GetNumberOfPixels(1) * min_component_fraction))
    retained = sitk.RelabelComponent(ordered, minimumObjectSize=minimum, sortByObjectSize=True)
    cleaned = sitk.Cast(retained > 0, sitk.sitkUInt8)
    if minimum > stats.GetNumberOfPixels(1):
        raise ValueError("lung mask is empty after removing small components")
    return cleaned


def lung_crop_region(mask: sitk.Image, margin_mm: float = 10.0, *,
                     min_component_fraction: float = MIN_COMPONENT_FRACTION) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Square XY lung bbox with physical margin; crop Z to lung extent."""
    mask = clean_lung_mask(mask, min_component_fraction)
    return _cleaned_lung_crop_region(mask, margin_mm)


def _cleaned_lung_crop_region(mask: sitk.Image, margin_mm: float) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if mask.GetDimension() != 3 or not math.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("mask must be 3D and margin_mm must be finite and non-negative")
    spacing = mask.GetSpacing()[:2]
    if any(not math.isfinite(s) or s <= 0 for s in spacing):
        raise ValueError("invalid mask spacing")
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(mask > 0, sitk.sitkUInt8))
    if not stats.HasLabel(1):
        raise ValueError("lung mask is empty")
    x, y, z, w, h, depth = stats.GetBoundingBox(1)
    width, height, _ = mask.GetSize()
    if max(w, h) > min(width, height):
        raise ValueError("lung bounding box cannot fit into a square")
    margin = math.ceil(margin_mm / min(spacing))
    side = min(max(w, h) + 2 * margin, min(width, height))
    x = min(max(x - (side - w) // 2, 0), width - side)
    y = min(max(y - (side - h) // 2, 0), height - side)
    return (x, y, z), (side, side, depth)


def crop_and_resize(image: sitk.Image, index: tuple[int, int, int],
                    size: tuple[int, int, int], *, is_label: bool) -> sitk.Image:
    cropped = sitk.RegionOfInterest(image, size, index)
    spacing = cropped.GetSpacing()
    return sitk.Resample(
        cropped, (256, 256, size[2]), sitk.Transform(),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
        cropped.GetOrigin(), (spacing[0] * size[0] / 256, spacing[1] * size[1] / 256, spacing[2]),
        cropped.GetDirection(), 0, cropped.GetPixelID(),
    )


def _rename_with_retry(source: Path, target: Path) -> None:
    """Retry temporary Windows access/sharing violations when moving outputs."""
    for attempt in range(6):
        try:
            source.rename(target)
            return
        except OSError as error:
            if (getattr(error, "winerror", None) not in (5, 32, 33)
                    or target.exists() or attempt == 5):
                raise
            time.sleep(0.5)


def crop_case(dataset_root: str | Path, case_id: str, margin_mm: float = 10.0, *,
              min_component_fraction: float = MIN_COMPONENT_FRACTION, overwrite: bool = False) -> None:
    """Write a case; replace existing output only when overwrite is enabled."""
    root = Path(dataset_root)
    if not case_id or Path(case_id).name != case_id or case_id in {".", ".."}:
        raise ValueError("invalid case ID")
    source = root / "nifti" / case_id
    output = root / "lung_cropped"
    processed, nifti = output / "processed" / case_id, output / "nifti" / case_id
    if (processed.exists() or nifti.exists()) and not overwrite:
        raise FileExistsError(f"output already exists for {case_id}: {output}")
    volumes = [sitk.ReadImage(str(source / f"{name}.nii.gz"))
               for name in ("image", "lung_mask", "tumor_mask")]
    image, lung, tumor = volumes
    if image.GetDimension() != 3 or not image.IsSameImageGeometryAs(lung) or not image.IsSameImageGeometryAs(tumor):
        raise ValueError(f"image and mask geometry mismatch for {case_id}")
    if np.any((sitk.GetArrayViewFromImage(tumor) > 0) & (sitk.GetArrayViewFromImage(lung) == 0)):
        raise ValueError(f"tumor mask extends outside lung mask for {case_id}")
    cleaned_lung = clean_lung_mask(lung, min_component_fraction)
    index, size = _cleaned_lung_crop_region(cleaned_lung, margin_mm)
    tumor_array = sitk.GetArrayViewFromImage(tumor)
    x, y, z = index
    w, h, depth = size
    inside = tumor_array[z:z + depth, y:y + h, x:x + w]
    if np.count_nonzero(inside) != np.count_nonzero(tumor_array):
        raise ValueError(f"crop would exclude tumor voxels for {case_id}")
    # Keep the pipeline's existing convention: tumor voxels also count as lung
    # for slice selection, but they do not influence the lung bounding box.
    lung_for_output = sitk.Cast((cleaned_lung > 0) | (tumor > 0), sitk.sitkUInt8)
    cropped = [crop_and_resize(v, index, size, is_label=(i != 0))
               for i, v in enumerate((image, lung_for_output, tumor))]
    processed.parent.mkdir(parents=True, exist_ok=True)
    nifti.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    stage_processed = processed.parent / f".{case_id}.{token}.tmp"
    stage_nifti = nifti.parent / f".{case_id}.{token}.tmp"
    backups: list[tuple[Path, Path]] = []
    moved: list[Path] = []
    try:
        stage_nifti.mkdir()
        for name, volume in zip(("image", "lung_mask", "tumor_mask"), cropped):
            sitk.WriteImage(volume, str(stage_nifti / f"{name}.nii.gz"))
        (stage_nifti / "crop.json").write_text(json.dumps({"source_index_xyz": index, "source_size_xyz": size,
                                                           "margin_xy_mm": margin_mm,
                                                           "min_component_fraction": min_component_fraction}, indent=2), encoding="utf-8")
        for directory, volume, is_label in zip(("images", "lung_labels", "tumor_labels"), cropped, (False, True, True)):
            folder = stage_processed / directory
            folder.mkdir(parents=True)
            array = sitk.GetArrayFromImage(volume)
            for z, slice_array in enumerate(array, start=index[2]):
                png = (slice_array > 0).astype(np.uint8) * 255 if is_label else slice_array.astype(np.uint8)
                Image.fromarray(png).save(folder / f"{z:04d}.png")
        for target in (nifti, processed):
            if target.exists():
                backup = target.parent / f".{case_id}.{token}.backup"
                _rename_with_retry(target, backup)
                backups.append((target, backup))
        _rename_with_retry(stage_nifti, nifti)
        moved.append(nifti)
        _rename_with_retry(stage_processed, processed)
        moved.append(processed)
    except Exception:
        for target in reversed(moved):
            shutil.rmtree(target)
        for target, backup in reversed(backups):
            _rename_with_retry(backup, target)
        raise
    else:
        for _, backup in backups:
            shutil.rmtree(backup)
    finally:
        shutil.rmtree(stage_processed, ignore_errors=True)
        shutil.rmtree(stage_nifti, ignore_errors=True)


def main() -> int:
    if SKIP_EXISTING and OVERWRITE:
        raise ValueError("SKIP_EXISTING and OVERWRITE cannot both be True")
    root = Path(DATASET_ROOT)
    source = root / "nifti"
    if not source.is_dir():
        raise FileNotFoundError(f"source directory not found: {source}")
    if not math.isfinite(MIN_FREE_GB) or MIN_FREE_GB < 0:
        raise ValueError("MIN_FREE_GB must be finite and non-negative")
    cases = sorted(CASE_IDS if CASE_IDS is not None else (p.name for p in source.iterdir() if p.is_dir()))
    failures = 0
    for number, case in enumerate(cases, 1):
        processed = root / "lung_cropped" / "processed" / case
        nifti = root / "lung_cropped" / "nifti" / case
        if SKIP_EXISTING and processed.is_dir() and nifti.is_dir():
            print(f"[{number}/{len(cases)}] SKIP {case}", flush=True)
            continue
        if shutil.disk_usage(root).free < MIN_FREE_GB * 1_000_000_000:
            print(f"ERROR: free disk space is below {MIN_FREE_GB:g} GB; stopped before {case}", flush=True)
            return 1
        try:
            crop_case(root, case, MARGIN_MM,
                      min_component_fraction=MIN_COMPONENT_FRACTION, overwrite=OVERWRITE)
            print(f"[{number}/{len(cases)}] OK {case}", flush=True)
        except Exception as error:
            failures += 1
            print(f"[{number}/{len(cases)}] ERROR {case}: {error}", flush=True)
    if OVERWRITE:
        print("Existing label CSVs may be stale; regenerate them before training.", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
