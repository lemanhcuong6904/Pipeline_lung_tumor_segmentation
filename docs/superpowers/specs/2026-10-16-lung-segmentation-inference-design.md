# Lung segmentation inference and evaluation

## Goal

Evaluate a trained `UNet` on a configurable split of NSCLC-Radiomics lung
segmentations.  The inference run must report 2D and spacing-aware 3D metrics
without persisting model prediction images or volumes.

## Inputs and configuration

`inferences/infer_lung_segmentation.py` will declare a `CFG` dictionary rather
than command-line options.  It will include the checkpoint, dataset root,
split file, NIfTI root, output directory, batch size, workers, device and
probability threshold.

The split file contains case IDs.  PNG images and lung labels live under
`processed/<case_id>/images` and `processed/<case_id>/lung_labels`.
`nifti/<case_id>/lung_mask.nii.gz` is the reference for spacing and validates
the final reconstructed volume shape.

## Components

### Inference dataset

Add a dataset dedicated to evaluation that yields image tensor, binary target,
case ID and zero-based slice index.  It reads the exact configured split and
keeps images in sorted filename order.  The existing training dataset API is
not changed.

### Metrics utilities

Add NumPy-based functions under `utils` for:

- binary Dice and IoU for a slice or a volume;
- stacking sorted case slices to a ZYX volume, rejecting missing or duplicate
  indices;
- surface extraction and symmetric nearest-neighbour distances in physical
  millimetres using the NIfTI spacing `(x, y, z)`;
- ASSD as the mean symmetric surface distance and HD95 as its 95th percentile.

Both-empty masks receive Dice and IoU of `1.0`.  When exactly one 3D mask is
empty, ASSD and HD95 are undefined and represented internally by `NaN` and as
JSON `null` in reports.  Surface-metric means use finite cases only and report
their valid-case counts.

### Inference loop

The script loads `UNet` from the training checkpoint's `model_state`, runs
`eval()` and `torch.inference_mode()` over batched slices, then thresholds
`sigmoid(logits)`.  It accumulates only thresholded Boolean masks in memory,
organised by case and slice index.  No PNG, NIfTI, logits or prediction volume
is written.

For every case, the script computes mean per-slice 2D Dice/IoU and 3D
Dice/IoU/ASSD/HD95 over its reconstructed volume.  NIfTI spacing is read from
`lung_mask.nii.gz` before calculating physical surface distances.

## Reports

Only two artifacts are created under `OUTPUT_DIR`:

- `per_case_metrics.csv`: one row per case with case ID, slice count, spacing,
  foreground voxel diagnostics, mean 2D Dice/IoU and 3D Dice/IoU/ASSD/HD95.
- `mean_metrics.json`: split/case counts, threshold, global mean 2D Dice/IoU
  across all slices, macro mean 3D metrics across cases, and valid surface
  metric case counts.

The JSON encoder converts non-finite floating values to `null` so the output
is valid standard JSON.

## Verification

Unit tests will cover overlap edge cases, anisotropic spacing in surface
distances, ordered volume aggregation and JSON-safe report values.  A small
synthetic inference fixture will verify that the script produces exactly the
CSV and JSON report files and no prediction directory.
