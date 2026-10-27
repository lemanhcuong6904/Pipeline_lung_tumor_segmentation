"""Shared DICOM preprocessing utilities for the two NSCLC datasets.

The output grid always contains the whole acquired CT field of view.  Only
the in-plane dimensions are resized to 256 by 256; Radiogenomics optionally
uses a fixed Z spacing as well.
"""

from __future__ import annotations

import csv
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image, ImageDraw

HU_WINDOW_LOW = -700
HU_WINDOW_HIGH = 500
LUNG_SEGMENT_LABELS = {"lung"}
RADIOMICS_TUMOR_SEGMENT_LABELS = {"neoplasm, primary"}
RADIOGENOMICS_TUMOR_SEGMENT_LABELS = {"heart", "tissue", "segmentation"}
TOTALSEGMENTATOR_LUNG_TASK_ID = 291
TOTALSEGMENTATOR_FAST_TASK_ID = 297
TOTALSEGMENTATOR_LUNG_ROIS = (
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
)
TOTALSEGMENTATOR_LUNG_LABEL_IDS = frozenset((10, 11, 12, 13, 14))
TOTALSEGMENTATOR_SPLIT_MIN_DEPTH = 150
REPORT_FIELDS = (
    "raw_case_id",
    "case_id",
    "status",
    "message",
    "ct_slices",
    "original_size",
    "original_shape_dhw",
    "original_spacing",
    "original_origin",
    "original_direction",
    "crop_index",
    "crop_size",
    "crop_z_to_lung_bbox",
    "final_size",
    "final_shape_dhw",
    "final_spacing",
    "final_origin",
    "final_direction",
    "seg_path",
    "lung_source",
    "lung_labels",
    "removed_lung_components",
    "tumor_source",
    "tumor_labels",
    "totalsegmentator_input",
    "totalsegmentator_output",
    "totalsegmentator_force_split",
)


def output_case_id(raw_case_id: str) -> str:
    return raw_case_id.replace("LUNG1-", "LUNG-", 1)


def normalize_hu_to_uint8(array: np.ndarray, low: int, high: int) -> np.ndarray:
    if low >= high:
        raise ValueError("HU lower bound must be lower than upper bound")
    clipped = np.clip(array, low, high).astype(np.float32)
    return np.rint((clipped - low) * 255.0 / (high - low)).astype(np.uint8)


def normalize_ct_image(
    ct: sitk.Image, low: int = HU_WINDOW_LOW, high: int = HU_WINDOW_HIGH
) -> sitk.Image:
    result = sitk.GetImageFromArray(
        normalize_hu_to_uint8(sitk.GetArrayFromImage(ct), low, high)
    )
    result.CopyInformation(ct)
    return result


def clip_hu_image(ct: sitk.Image, low: int, high: int) -> sitk.Image:
    if low >= high:
        raise ValueError("HU lower bound must be lower than upper bound")
    result = sitk.GetImageFromArray(
        np.clip(sitk.GetArrayFromImage(ct), low, high).astype(np.int16)
    )
    result.CopyInformation(ct)
    return result


def _same_geometry(first: sitk.Image, second: sitk.Image) -> bool:
    return (
        first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing())
        and np.allclose(first.GetOrigin(), second.GetOrigin())
        and np.allclose(first.GetDirection(), second.GetDirection())
    )


def geometry_metadata(original: sitk.Image, final: sitk.Image) -> dict[str, str]:
    """Return stable CT-grid metadata for ``processing_report.csv``."""
    original_size, final_size = original.GetSize(), final.GetSize()
    return {
        "ct_slices": str(original_size[2]),
        "original_size": str(original_size),
        "original_shape_dhw": str(
            (original_size[2], original_size[1], original_size[0])
        ),
        "original_spacing": str(original.GetSpacing()),
        "original_origin": str(original.GetOrigin()),
        "original_direction": str(original.GetDirection()),
        "final_size": str(final_size),
        "final_shape_dhw": str((final_size[2], final_size[1], final_size[0])),
        "final_spacing": str(final.GetSpacing()),
        "final_origin": str(final.GetOrigin()),
        "final_direction": str(final.GetDirection()),
    }


def nnunet_device(device: str | None) -> str:
    """Map the public TotalSegmentator spelling to its nnU-Net API spelling."""
    normalized = (device or "cuda").casefold()
    aliases = {"gpu": "cuda", "cuda": "cuda", "cpu": "cpu", "mps": "mps"}
    if normalized not in aliases:
        raise ValueError("device must be one of: gpu, cuda, cpu, or mps")
    return aliases[normalized]


def should_force_split_totalseg(size_xyz: tuple[int, int, int]) -> bool:
    """Split unusually long CT volumes to limit Task 291 host-memory use."""
    return size_xyz[2] >= TOTALSEGMENTATOR_SPLIT_MIN_DEPTH


def resample_ct_for_totalseg(ct_hu: sitk.Image, spacing_mm: float) -> sitk.Image:
    """Create the low-resolution CT grid consumed by TotalSegmentator.

    Resampling before nnU-Net avoids its expensive multi-class upsampling back
    to the native 512x512 grid.  The final binary lung mask is resampled once
    to the native CT grid after inference.
    """
    if spacing_mm <= 0:
        raise ValueError("TotalSegmentator spacing must be positive")
    size, spacing = ct_hu.GetSize(), ct_hu.GetSpacing()
    target_size = tuple(
        max(1, int(round((size[axis] - 1) * spacing[axis] / spacing_mm)) + 1)
        for axis in range(3)
    )
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing((spacing_mm, spacing_mm, spacing_mm))
    resampler.SetOutputOrigin(ct_hu.GetOrigin())
    resampler.SetOutputDirection(ct_hu.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1024)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(ct_hu)


def write_nnunet_input(image: sitk.Image, work_root: Path) -> Path:
    """Write one CT in nnU-Net's required single-modality directory layout."""
    input_dir = work_root / "nnunet_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(input_dir / "s01_0000.nii.gz"))
    return input_dir


def lung_lobe_binary_mask(task_291_labels: np.ndarray) -> np.ndarray:
    """Keep only the five lung-lobe labels emitted by TotalSegmentator Task 291."""
    return np.isin(task_291_labels, tuple(TOTALSEGMENTATOR_LUNG_LABEL_IDS)).astype(
        np.uint8
    )


def remove_small_lung_components(lung_mask: sitk.Image, minimum_voxels: int) -> tuple[sitk.Image, int]:
    """Use the same native-grid lung-fragment cleanup as the old project."""
    connected = sitk.ConnectedComponent(sitk.Cast(lung_mask > 0, sitk.sitkUInt8))
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(connected)
    kept = [label for label in statistics.GetLabels() if statistics.GetNumberOfPixels(label) >= minimum_voxels]
    if not kept:
        raise ValueError("lung mask has no component above the minimum size")
    cleaned = sitk.GetImageFromArray(np.isin(sitk.GetArrayFromImage(connected), kept).astype(np.uint8))
    cleaned.CopyInformation(lung_mask)
    return cleaned, len(statistics.GetLabels()) - len(kept)


def clean_tumor_mask(tumor_mask: sitk.Image, *, fill_holes: bool, keep_largest_component: bool, minimum_component_voxels: int) -> tuple[sitk.Image, dict[str, int]]:
    """Port the old Radiogenomics tumour cleanup before output resizing."""
    cleaned = sitk.Cast(tumor_mask > 0, sitk.sitkUInt8)
    original_voxels = int(sitk.GetArrayViewFromImage(cleaned).sum())
    if fill_holes:
        cleaned = sitk.BinaryFillhole(cleaned, fullyConnected=True)
    connected_filter = sitk.ConnectedComponentImageFilter()
    connected_filter.FullyConnectedOn()
    connected = connected_filter.Execute(cleaned)
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(connected)
    labels = list(statistics.GetLabels())
    if keep_largest_component and labels:
        kept = [max(labels, key=statistics.GetNumberOfPixels)]
    elif minimum_component_voxels:
        kept = [label for label in labels if statistics.GetNumberOfPixels(label) >= minimum_component_voxels]
    else:
        kept = labels
    array = np.isin(sitk.GetArrayFromImage(connected), kept).astype(np.uint8)
    cleaned = sitk.GetImageFromArray(array)
    cleaned.CopyInformation(tumor_mask)
    return cleaned, {
        "tumor_original_voxels": original_voxels,
        "tumor_cleaned_voxels": int(array.sum()),
        "tumor_components_before_cleanup": len(labels),
        "tumor_components_removed": len(labels) - len(kept),
        "tumor_kept_largest_component": int(keep_largest_component),
        "tumor_min_component_voxels": minimum_component_voxels,
    }


def _totalseg_z_chunks(image: sitk.Image, split: bool) -> list[tuple[int, sitk.Image]]:
    """Return one or two full-XY chunks, retaining their Z positions."""
    size = image.GetSize()
    if not split:
        return [(0, image)]
    first_depth = size[2] // 2
    return [
        (0, sitk.RegionOfInterest(image, (size[0], size[1], first_depth), (0, 0, 0))),
        (
            first_depth,
            sitk.RegionOfInterest(
                image, (size[0], size[1], size[2] - first_depth), (0, 0, first_depth)
            ),
        ),
    ]


def resize_full_ct_xy(
    image: sitk.Image, is_label: bool, output_z_spacing_mm: float | None = None
) -> sitk.Image:
    """Resize the complete CT grid to 256x256, never applying a lung crop."""
    if output_z_spacing_mm is not None and output_z_spacing_mm <= 0:
        raise ValueError("output_z_spacing_mm must be positive or None")
    size = image.GetSize()
    spacing = image.GetSpacing()
    z_size = (
        size[2]
        if output_z_spacing_mm is None
        else max(1, round((size[2] - 1) * spacing[2] / output_z_spacing_mm) + 1)
    )
    z_spacing = (
        spacing[2] if output_z_spacing_mm is None else float(output_z_spacing_mm)
    )
    target_size = (256, 256, z_size)
    target_spacing = (spacing[0] * size[0] / 256, spacing[1] * size[1] / 256, z_spacing)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(image)


def write_case_outputs(
    processed_root: Path,
    nifti_root: Path,
    case_id: str,
    image: sitk.Image,
    lung_mask: sitk.Image,
    tumor_mask: sitk.Image,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write paired PNG slices and full-resolution NIfTI outputs."""
    if not (_same_geometry(image, lung_mask) and _same_geometry(image, tumor_mask)):
        raise ValueError("image, lung mask, and tumor mask geometry must match")
    final_processed, final_nifti = processed_root / case_id, nifti_root / case_id
    if (final_processed.exists() or final_nifti.exists()) and not overwrite:
        raise FileExistsError(f"output already exists for {case_id}")
    processed_root.mkdir(parents=True, exist_ok=True)
    nifti_root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    tmp_processed, tmp_nifti = (
        processed_root / f".{case_id}.{token}.tmp",
        nifti_root / f".{case_id}.{token}.tmp",
    )
    try:
        combined_lung = sitk.Cast((lung_mask > 0) | (tumor_mask > 0), sitk.sitkUInt8)
        arrays = {
            "images": sitk.GetArrayFromImage(image).astype(np.uint8),
            "lung_labels": sitk.GetArrayFromImage(combined_lung).astype(np.uint8) * 255,
            "tumor_labels": (sitk.GetArrayFromImage(tumor_mask) > 0).astype(np.uint8)
            * 255,
        }
        for name, array in arrays.items():
            directory = tmp_processed / name
            directory.mkdir(parents=True, exist_ok=True)
            for index, image_slice in enumerate(array):
                Image.fromarray(image_slice).save(directory / f"{index:04d}.png")
        tmp_nifti.mkdir(parents=True)
        sitk.WriteImage(image, str(tmp_nifti / "image.nii.gz"))
        sitk.WriteImage(combined_lung, str(tmp_nifti / "lung_mask.nii.gz"))
        sitk.WriteImage(
            sitk.Cast(tumor_mask > 0, sitk.sitkUInt8),
            str(tmp_nifti / "tumor_mask.nii.gz"),
        )
        if final_processed.exists():
            shutil.rmtree(final_processed)
        if final_nifti.exists():
            shutil.rmtree(final_nifti)
        tmp_processed.replace(final_processed)
        tmp_nifti.replace(final_nifti)
    except Exception:
        shutil.rmtree(tmp_processed, ignore_errors=True)
        shutil.rmtree(tmp_nifti, ignore_errors=True)
        raise


def _dicom_headers(case_dir: Path) -> Iterable[tuple[Path, pydicom.Dataset]]:
    for path in case_dir.rglob("*.dcm"):
        try:
            yield path, pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue


def _load_ct(case_dir: Path) -> tuple[sitk.Image, list[Path]]:
    series: dict[str, list[Path]] = {}
    for path, dataset in _dicom_headers(case_dir):
        if getattr(dataset, "Modality", "") == "CT":
            series.setdefault(str(dataset.SeriesInstanceUID), []).append(path)
    if not series:
        raise ValueError("no CT DICOM series found")
    paths = max(series.values(), key=len)
    headers = [
        pydicom.dcmread(path, stop_before_pixels=True, force=True) for path in paths
    ]
    orientation = np.asarray(headers[0].ImageOrientationPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    paths = [
        path
        for _, path in sorted(
            zip(
                (
                    float(
                        np.dot(
                            np.asarray(header.ImagePositionPatient, dtype=float), normal
                        )
                    )
                    for header in headers
                ),
                paths,
            ),
            key=lambda item: item[0],
        )
    ]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in paths])
    return reader.Execute(), paths


def _find_dicom(case_dir: Path, modality: str) -> Path:
    for path, dataset in _dicom_headers(case_dir):
        if getattr(dataset, "Modality", "") == modality:
            return path
    raise ValueError(f"no DICOM {modality} object found")


def _decode_seg(
    seg_path: Path, labels: set[str], ct: sitk.Image
) -> tuple[sitk.Image, list[str]]:
    dataset = pydicom.dcmread(seg_path, force=True)
    selected = [
        int(segment.SegmentNumber)
        for segment in dataset.SegmentSequence
        if str(getattr(segment, "SegmentLabel", "")).strip().casefold() in labels
    ]
    if not selected:
        raise ValueError(f"SEG labels not found: {sorted(labels)}")
    names = [
        str(segment.SegmentLabel)
        for segment in dataset.SegmentSequence
        if int(segment.SegmentNumber) in selected
    ]
    mask = np.zeros(sitk.GetArrayFromImage(ct).shape, dtype=np.uint8)
    shared = (
        getattr(
            dataset.SharedFunctionalGroupsSequence[0],
            "SegmentIdentificationSequence",
            None,
        )
        if getattr(dataset, "SharedFunctionalGroupsSequence", None)
        else None
    )
    for frame, groups in zip(
        dataset.pixel_array, dataset.PerFrameFunctionalGroupsSequence, strict=True
    ):
        identification = (
            getattr(groups, "SegmentIdentificationSequence", None) or shared
        )
        if (
            not identification
            or int(identification[0].ReferencedSegmentNumber) not in selected
        ):
            continue
        position = tuple(
            float(value)
            for value in groups.PlanePositionSequence[0].ImagePositionPatient
        )
        x, y, z = ct.TransformPhysicalPointToIndex(position)
        if (
            x != 0
            or y != 0
            or not 0 <= z < ct.GetSize()[2]
            or frame.shape != (ct.GetSize()[1], ct.GetSize()[0])
        ):
            raise ValueError("SEG is not aligned to the CT grid")
        mask[z] = np.maximum(mask[z], (frame > 0).astype(np.uint8))
    result = sitk.GetImageFromArray(mask)
    result.CopyInformation(ct)
    return result, names


def _decode_rtstruct(
    rtstruct_path: Path, ct: sitk.Image
) -> tuple[sitk.Image, list[str]]:
    dataset = pydicom.dcmread(rtstruct_path, force=True)
    roi_names = {
        int(item.ROINumber): str(item.ROIName)
        for item in dataset.StructureSetROISequence
    }
    selected = {
        number for number, name in roi_names.items() if name.casefold() == "gtv-1"
    } or {
        number
        for number, name in roi_names.items()
        if name.casefold().startswith("gtv")
    }
    if not selected:
        raise ValueError("RTSTRUCT does not contain a GTV contour")
    array = np.zeros(sitk.GetArrayFromImage(ct).shape, dtype=np.uint8)
    for roi in dataset.ROIContourSequence:
        if int(roi.ReferencedROINumber) not in selected:
            continue
        for contour in getattr(roi, "ContourSequence", []):
            points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            indices = [
                ct.TransformPhysicalPointToContinuousIndex(tuple(point))
                for point in points
            ]
            z = int(round(np.mean([index[2] for index in indices])))
            if 0 <= z < ct.GetSize()[2]:
                polygon = Image.new("L", (ct.GetSize()[0], ct.GetSize()[1]), 0)
                ImageDraw.Draw(polygon).polygon(
                    [(index[0], index[1]) for index in indices], fill=1
                )
                array[z] = np.maximum(array[z], np.asarray(polygon, dtype=np.uint8))
    if not array.any():
        raise ValueError("RTSTRUCT GTV contours did not rasterize onto CT")
    result = sitk.GetImageFromArray(array)
    result.CopyInformation(ct)
    return result, [roi_names[number] for number in sorted(selected)]


def _tumor_mask(
    case_dir: Path, ct: sitk.Image, labels: set[str]
) -> tuple[sitk.Image, list[str], str, str]:
    try:
        seg_path = _find_dicom(case_dir, "SEG")
        mask, names = _decode_seg(seg_path, labels, ct)
        return mask, names, "DICOM SEG", str(seg_path)
    except ValueError:
        rtstruct = _find_dicom(case_dir, "RTSTRUCT")
        mask, names = _decode_rtstruct(rtstruct, ct)
        return mask, names, "RTSTRUCT", ""


def segment_lung_with_totalsegmentator(
    ct_hu: sitk.Image,
    case_id: str,
    staging_root: Path,
    *,
    device: str | None,
    overwrite: bool,
    resample_mm: float = 3.0,
    force_split: bool = False,
) -> tuple[sitk.Image, Path, Path, bool]:
    """Run the Task 291 lung model on a low-resolution CT, then restore its grid.

    This deliberately calls ``nnUNetv2_predict`` directly instead of the
    TotalSegmentator convenience wrapper.  The wrapper runs extra whole-body
    post-processing and can spawn several progress bars/workers for long CTs.
    """
    case_root = staging_root / case_id
    input_path, output_dir = (
        case_root / "ct_hu_clipped.nii.gz",
        case_root / "totalsegmentator",
    )
    mask_path = output_dir / "lung_mask.nii.gz"
    case_root.mkdir(parents=True, exist_ok=True)
    if mask_path.exists() and not overwrite:
        mask = sitk.ReadImage(str(mask_path))
        if not _same_geometry(mask, ct_hu):
            mask = sitk.Resample(
                mask,
                ct_hu,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8,
            )
        return sitk.Cast(mask > 0, sitk.sitkUInt8), input_path, output_dir, force_split
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    model_input = resample_ct_for_totalseg(ct_hu, resample_mm)
    sitk.WriteImage(model_input, str(input_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from totalsegmentator.config import setup_nnunet
        from totalsegmentator.libs import download_pretrained_weights

        setup_nnunet()
        from totalsegmentator.nnunet import nnUNetv2_predict
    except ImportError as error:
        raise RuntimeError(
            "TotalSegmentator was not found; install TotalSegmentator."
        ) from error
    if resample_mm <= 0:
        raise ValueError("TotalSegmentator resample_mm must be positive")
    # Task 291 is the five-lung-lobe model.  Do not call the public
    # ``task='total'`` wrapper: it starts at Task 292 and downloads a
    # different total-body checkpoint even when roi_subset contains lungs.
    download_pretrained_weights(TOTALSEGMENTATOR_LUNG_TASK_ID)
    split_volume = force_split or should_force_split_totalseg(model_input.GetSize())
    labels = np.zeros(sitk.GetArrayFromImage(model_input).shape, dtype=np.uint8)
    for chunk_index, (z_start, chunk) in enumerate(
        _totalseg_z_chunks(model_input, split_volume)
    ):
        chunk_root = output_dir / f"chunk_{chunk_index:02d}"
        nnunet_input = write_nnunet_input(chunk, chunk_root)
        nnunet_output = chunk_root / "nnunet_output"
        nnunet_output.mkdir(parents=True, exist_ok=True)
        try:
            nnUNetv2_predict(
                nnunet_input,
                nnunet_output,
                TOTALSEGMENTATOR_LUNG_TASK_ID,
                model="3d_fullres",
                folds=[0],
                trainer="nnUNetTrainerNoMirroring",
                tta=False,
                num_threads_preprocessing=1,
                num_threads_nifti_save=1,
                device=nnunet_device(device),
                quiet=False,
            )
            part = sitk.ReadImage(str(nnunet_output / "s01.nii.gz"))
            part_labels = sitk.GetArrayFromImage(part).astype(np.uint8)
            labels[z_start : z_start + part_labels.shape[0]] = part_labels
        finally:
            shutil.rmtree(chunk_root, ignore_errors=True)
    part = sitk.GetImageFromArray(labels)
    part.CopyInformation(model_input)
    if not _same_geometry(part, ct_hu):
        part = sitk.Resample(
            part, ct_hu, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
        )
    union = lung_lobe_binary_mask(sitk.GetArrayFromImage(part))
    if not union.any():
        raise ValueError("TotalSegmentator generated an empty lung mask")
    mask = sitk.GetImageFromArray(union)
    mask.CopyInformation(ct_hu)
    sitk.WriteImage(mask, str(mask_path))
    return mask, input_path, output_dir, split_volume


def segment_lung_with_totalsegmentator_legacy(
    ct_hu: sitk.Image, case_id: str, staging_root: Path, *, task_id: int,
    trainer: str, resample_mm: float, device: str | None, overwrite: bool,
    lung_rois: Sequence[str],
) -> tuple[sitk.Image, Path, Path]:
    """Exact TotalSegmentator wrapper path used by the old Radiogenomics project."""
    case_root = staging_root / case_id
    input_path, output_dir = case_root / "ct_hu_clipped.nii.gz", case_root / "totalsegmentator"
    mask_path = output_dir / "lung_mask.nii.gz"
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    case_root.mkdir(parents=True, exist_ok=True)
    if overwrite or not input_path.exists():
        sitk.WriteImage(ct_hu, str(input_path))
    if mask_path.exists() and not overwrite:
        mask = sitk.ReadImage(str(mask_path))
        if not _same_geometry(mask, ct_hu):
            mask = sitk.Resample(mask, ct_hu, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        return sitk.Cast(mask > 0, sitk.sitkUInt8), input_path, output_dir
    try:
        from totalsegmentator.config import setup_nnunet
        from totalsegmentator.libs import download_pretrained_weights

        setup_nnunet()
        from totalsegmentator.nnunet import nnUNet_predict_image
    except ImportError as error:
        raise RuntimeError("TotalSegmentator was not found; install TotalSegmentator.") from error
    download_pretrained_weights(task_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lung_lobes_", dir=output_dir, ignore_cleanup_errors=True) as temp_dir:
        output_path = Path(temp_dir) / "lung_lobes.nii.gz"
        for attempt in range(1, 4):
            try:
                nnUNet_predict_image(
                    input_path, output_path, task_id, model="3d_fullres", folds=[0],
                    trainer=trainer, tta=False, multilabel_image=True,
                    resample=resample_mm, task_name="total", roi_subset=list(lung_rois),
                    device=nnunet_device(device), quiet=False,
                )
                break
            except PermissionError as error:
                if getattr(error, "winerror", None) != 32 or attempt == 3:
                    raise
                time.sleep(2)
        lobe_paths = sorted(path for path in Path(temp_dir).glob("lung_*.nii*") if path.is_file())
        if not lobe_paths:
            raise ValueError("TotalSegmentator produced no lung-lobe mask")
        combined = np.zeros(sitk.GetArrayFromImage(ct_hu).shape, dtype=np.uint8)
        for path in lobe_paths:
            lobe = sitk.ReadImage(str(path))
            if not _same_geometry(lobe, ct_hu):
                lobe = sitk.Resample(lobe, ct_hu, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            combined |= (sitk.GetArrayFromImage(lobe) > 0).astype(np.uint8)
        if not combined.any():
            raise ValueError("TotalSegmentator generated an empty lung mask")
        mask = sitk.GetImageFromArray(combined)
        mask.CopyInformation(ct_hu)
        sitk.WriteImage(mask, str(mask_path))
    return mask, input_path, output_dir


def process_radiogenomics_case(
    case_dir: Path, processed_root: Path, nifti_root: Path, staging_root: Path, *,
    overwrite: bool, task_id: int, device: str | None, trainer: str,
    resample_mm: float, lung_rois: Sequence[str], output_z_spacing_mm: float | None,
    reuse_existing_lung_mask: bool, tumor_fill_holes: bool,
    tumor_keep_largest_component: bool, tumor_min_component_voxels: int,
    hu_low: int = HU_WINDOW_LOW, hu_high: int = HU_WINDOW_HIGH,
    dry_run: bool = False,
) -> dict[str, str]:
    """Old Radiogenomics pipeline with full-FOV resize and persisted lung labels."""
    case_id = output_case_id(case_dir.name)
    result: dict[str, str] = {"raw_case_id": case_dir.name, "case_id": case_id, "status": "ok", "message": ""}
    if ((processed_root / case_id).exists() or (nifti_root / case_id).exists()) and not overwrite:
        result.update(status="skipped", message="output already exists")
        return result
    ct, ct_paths = _load_ct(case_dir)
    if dry_run:
        raise RuntimeError("dry run cannot create a required TotalSegmentator lung mask")
    lung, input_path, total_output = segment_lung_with_totalsegmentator_legacy(
        clip_hu_image(ct, hu_low, hu_high), case_id, staging_root, task_id=task_id,
        trainer=trainer, resample_mm=resample_mm, device=device,
        overwrite=overwrite and not reuse_existing_lung_mask, lung_rois=lung_rois,
    )
    tumor, tumor_names, tumor_source, tumor_seg = _tumor_mask(case_dir, ct, RADIOGENOMICS_TUMOR_SEGMENT_LABELS)
    tumor, tumor_cleanup = clean_tumor_mask(
        tumor, fill_holes=tumor_fill_holes,
        keep_largest_component=tumor_keep_largest_component,
        minimum_component_voxels=tumor_min_component_voxels,
    )
    lung, removed_lung_components = remove_small_lung_components(lung, 1_000)
    image_final = resize_full_ct_xy(normalize_ct_image(ct, hu_low, hu_high), False, output_z_spacing_mm)
    lung_final = resize_full_ct_xy(lung, True, output_z_spacing_mm)
    tumor_final = resize_full_ct_xy(tumor, True, output_z_spacing_mm)
    result.update(geometry_metadata(ct, image_final))
    result.update(
        seg_path=tumor_seg, lung_source="TotalSegmentator",
        lung_labels=f"TotalSegmentator:Task{task_id}:lung_lobes",
        removed_lung_components=str(removed_lung_components), tumor_source=tumor_source,
        tumor_labels=";".join(tumor_names), crop_index="(0, 0, 0)",
        crop_size=str(ct.GetSize()), crop_z_to_lung_bbox="False",
        totalsegmentator_input=str(input_path), totalsegmentator_output=str(total_output),
        totalsegmentator_force_split="False", **{key: str(value) for key, value in tumor_cleanup.items()},
    )
    write_case_outputs(processed_root, nifti_root, case_id, image_final, lung_final, tumor_final, overwrite=overwrite)
    return result


def process_case(
    case_dir: Path,
    processed_root: Path,
    nifti_root: Path,
    staging_root: Path,
    *,
    radiogenomics: bool,
    overwrite: bool,
    dry_run: bool,
    device: str | None,
    hu_low: int = HU_WINDOW_LOW,
    hu_high: int = HU_WINDOW_HIGH,
    output_z_spacing_mm: float | None = None,
    totalsegmentator_resample_mm: float = 3.0,
) -> dict[str, str]:
    """Process a full-FOV case; `radiogenomics` selects label and lung policy."""
    case_id = output_case_id(case_dir.name)
    result: dict[str, str] = {
        "raw_case_id": case_dir.name,
        "case_id": case_id,
        "status": "ok",
        "message": "",
    }
    if (
        (processed_root / case_id).exists() or (nifti_root / case_id).exists()
    ) and not overwrite:
        result.update(status="skipped", message="output already exists")
        return result
    ct, ct_paths = _load_ct(case_dir)
    seg_path = ""
    if radiogenomics:
        needs_totalseg, tumor_labels = True, RADIOGENOMICS_TUMOR_SEGMENT_LABELS
    else:
        tumor_labels = RADIOMICS_TUMOR_SEGMENT_LABELS
        try:
            candidate = _find_dicom(case_dir, "SEG")
            lung, lung_names = _decode_seg(candidate, LUNG_SEGMENT_LABELS, ct)
            needs_totalseg = False
            seg_path = str(candidate)
        except ValueError:
            needs_totalseg = True
    if needs_totalseg:
        if dry_run:
            raise RuntimeError(
                "dry run cannot create a required TotalSegmentator lung mask"
            )
        lung, total_input, total_output, total_split = segment_lung_with_totalsegmentator(
            clip_hu_image(ct, hu_low, hu_high),
            case_id,
            staging_root,
            device=device,
            overwrite=overwrite,
            resample_mm=totalsegmentator_resample_mm,
        )
        lung_names, lung_source = ["TotalSegmentator lung lobes"], "TotalSegmentator"
    else:
        total_input, total_output, total_split, lung_source = "", "", False, "DICOM SEG"
    tumor, tumor_names, tumor_source, tumor_seg = _tumor_mask(
        case_dir, ct, tumor_labels
    )
    image_final = resize_full_ct_xy(
        normalize_ct_image(ct, hu_low, hu_high), False, output_z_spacing_mm
    )
    lung_final = resize_full_ct_xy(lung, True, output_z_spacing_mm)
    tumor_final = resize_full_ct_xy(tumor, True, output_z_spacing_mm)
    result.update(geometry_metadata(ct, image_final))
    result.update(
        seg_path=seg_path or tumor_seg,
        lung_source=lung_source,
        lung_labels=";".join(lung_names),
        removed_lung_components="0",
        tumor_source=tumor_source,
        tumor_labels=";".join(tumor_names),
        crop_index="(0, 0, 0)",
        crop_size=str(ct.GetSize()),
        crop_z_to_lung_bbox="False",
        totalsegmentator_input=str(total_input),
        totalsegmentator_output=str(total_output),
        totalsegmentator_force_split=str(total_split),
    )
    if not dry_run:
        write_case_outputs(
            processed_root,
            nifti_root,
            case_id,
            image_final,
            lung_final,
            tumor_final,
            overwrite=overwrite,
        )
    return result


def write_report(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted({field for row in rows for field in row} - set(REPORT_FIELDS))
    fields = [*REPORT_FIELDS, *extra_fields]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )


def sync_split_config(source_root: Path, destination_root: Path) -> None:
    """Copy the established train/val/test split files from the previous project."""
    config = destination_root / "config"
    config.mkdir(parents=True, exist_ok=True)
    for name in ("train.txt", "val.txt", "test.txt"):
        source = source_root / "config" / name
        if not source.is_file():
            raise FileNotFoundError(f"missing prior split file: {source}")
        shutil.copy2(source, config / name)
