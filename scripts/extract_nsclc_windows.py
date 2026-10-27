"""Create lung and mediastinal PNGs from original NSCLC Radiomics or Radiogenomics CT DICOMs.

The existing 8-bit NIfTI volumes cannot recover HU outside the wide window.
The output follows the existing resize, crop, and slice numbering conventions.
"""
from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import SimpleITK as sitk
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.crop_nsclc_lungs import _rename_with_retry, crop_and_resize
from scripts.preprocess_nsclc import _load_ct, _same_geometry, normalize_ct_image, output_case_id, resize_full_ct_xy

DATASET = "nsclc_radiogenomics"  # "nsclc_radiomics" or "nsclc_radiogenomics".
RAW_ROOT = {"nsclc_radiomics": Path(r"D:\NSCLC-Radiomics"),
            "nsclc_radiogenomics": Path(r"D:\NSCLC-Radiogenomics")}[DATASET]
DATASET_ROOT = ROOT_DIR / "data" / DATASET
CASE_IDS: list[str] | None = None  # None: all processed cases; or e.g. ["LUNG-001"].
OVERWRITE = True  # Replace existing window directories when True.
WINDOWS = {"images_lung": (-600, 1500), "images_mediastinal": (40, 400)}


def extract_case(ct: sitk.Image, dataset_root: Path, case_id: str, *, overwrite: bool = False) -> None:
    """Write both windows for a case without modifying existing images or labels."""
    root = Path(dataset_root)
    if not case_id or Path(case_id).name != case_id or case_id in {".", ".."}:
        raise ValueError("invalid case ID")
    full_root = root / "processed" / case_id
    crop_root = root / "lung_cropped" / "processed" / case_id
    full_reference = sitk.ReadImage(str(root / "nifti" / case_id / "image.nii.gz"))
    crop_reference = sitk.ReadImage(str(root / "lung_cropped" / "nifti" / case_id / "image.nii.gz"))
    metadata = json.loads((root / "lung_cropped" / "nifti" / case_id / "crop.json").read_text(encoding="utf-8"))
    index, size = tuple(metadata["source_index_xyz"]), tuple(metadata["source_size_xyz"])

    # Validate grids and the complete PNG name set before creating output.
    z_spacing = full_reference.GetSpacing()[2]
    grid = resize_full_ct_xy(normalize_ct_image(ct, -700, 500), False, z_spacing)
    cropped_grid = crop_and_resize(grid, index, size, is_label=False)
    if not _same_geometry(grid, full_reference) or not _same_geometry(cropped_grid, crop_reference):
        raise ValueError(f"CT geometry does not match existing output for {case_id}")
    for directory, start, depth in ((full_root, 0, grid.GetSize()[2]), (crop_root, index[2], size[2])):
        expected = {f"{z:04d}.png" for z in range(start, start + depth)}
        actual = {path.name for path in (directory / "images").glob("*.png")}
        if actual != expected:
            raise ValueError(f"existing slice filenames do not match CT for {directory}")
    targets = [directory / name for directory in (full_root, crop_root) for name in WINDOWS]
    if not overwrite and any(target.exists() for target in targets):
        raise FileExistsError(f"window output already exists for {case_id}; set OVERWRITE = True")

    token = uuid.uuid4().hex
    stages = {target: target.parent / f".{target.name}.{token}.tmp" for target in targets}
    backups, moved = [], []
    try:
        for target in targets:
            stages[target].mkdir()
        for name, (level, width) in WINDOWS.items():
            low, high = level - width / 2, level + width / 2
            full = resize_full_ct_xy(normalize_ct_image(ct, low, high), False, z_spacing)
            crop = crop_and_resize(full, index, size, is_label=False)
            for directory, volume, start in ((full_root, full, 0), (crop_root, crop, index[2])):
                for z, image in enumerate(sitk.GetArrayViewFromImage(volume), start=start):
                    Image.fromarray(image).save(stages[directory / name] / f"{z:04d}.png")
        for target in targets:
            if target.exists():
                backup = target.parent / f".{target.name}.{token}.backup"
                _rename_with_retry(target, backup)
                backups.append((target, backup))
            _rename_with_retry(stages[target], target)
            moved.append(target)
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
        for stage in stages.values():
            shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    raw_root = Path(RAW_ROOT)
    dataset_root = Path(DATASET_ROOT)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw directory not found: {raw_root}")
    processed = dataset_root / "processed"
    if not processed.is_dir():
        raise FileNotFoundError(f"processed directory not found: {processed}")
    raw_cases = {output_case_id(path.name): path for path in raw_root.iterdir() if path.is_dir()}
    cases = sorted(set(CASE_IDS) if CASE_IDS is not None else (p.name for p in processed.iterdir() if p.is_dir()))
    if not cases:
        raise ValueError("no cases found")
    failures = 0
    for number, case_id in enumerate(cases, 1):
        try:
            if case_id not in raw_cases:
                raise FileNotFoundError(f"original DICOM case not found: {case_id}")
            ct, _ = _load_ct(raw_cases[case_id])
            extract_case(ct, dataset_root, case_id, overwrite=OVERWRITE)
            print(f"[{number}/{len(cases)}] OK {case_id}", flush=True)
        except Exception as error:
            failures += 1
            print(f"[{number}/{len(cases)}] ERROR {case_id}: {type(error).__name__}: {error}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
