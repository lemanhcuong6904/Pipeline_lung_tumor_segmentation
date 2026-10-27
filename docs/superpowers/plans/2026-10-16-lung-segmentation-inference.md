# Lung Segmentation Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a configurable UNet inference script that evaluates lung segmentation in 2D and spacing-aware 3D without writing predictions.

**Architecture:** A metadata-aware inference dataset exposes case and slice identity alongside each PNG pair. NumPy metric utilities aggregate thresholded slices into ZYX volumes and evaluate them with spacing from the reference NIfTI. The inference script writes only one per-case CSV and one summary JSON.

**Tech Stack:** Python 3.11, PyTorch, NumPy, SciPy, SimpleITK, Pillow, tqdm, pytest.

**Spec:** `docs/superpowers/specs/2026-10-16-lung-segmentation-inference-design.md`

## Global Constraints

- Use configuration variables in the script; do not add command-line arguments.
- Load `UNet` weights from a checkpoint's `model_state`.
- Use `nifti/<case_id>/lung_mask.nii.gz` for spacing and volume-shape validation.
- Do not write predictions, logits, PNGs, NIfTI files, or a prediction directory.
- Write only `per_case_metrics.csv` and `mean_metrics.json` under the configured output directory.
- Encode undefined ASSD/HD95 as JSON `null`; calculate their means on finite cases only.

## Review Focus

- Both prediction and target empty: Dice and IoU must be `1.0`; owned by Task 1.
- Exactly one volume empty: ASSD and HD95 must be undefined rather than zero; owned by Task 1.
- Anisotropic NIfTI spacing: surface distances must be measured in millimetres using XYZ-to-ZYX conversion; owned by Task 1.
- Missing, duplicate, or out-of-order slice indices: volume aggregation must reject incomplete cases; owned by Task 1.
- Output directory contents: inference must create only the two report files; owned by Task 3.

---

### Task 1: Binary and volume metrics utilities

**Files:**
- Create: `utils/segmentation_metrics.py`
- Test: `tests/test_segmentation_metrics.py`

**Interfaces:**
- Produces: `binary_overlap_metrics(prediction, target) -> dict[str, float]`, `binary_volume_metrics(prediction_zyx, target_zyx, spacing_xyz) -> dict[str, float]`, `aggregate_case_slices(records) -> dict[str, dict[str, np.ndarray]]`, and `finite_mean(values) -> float`.
- Consumed by: Task 3.

- [ ] **Step 1: Write failing metric tests**

```python
def test_binary_overlap_metrics_gives_perfect_scores_for_two_empty_masks():
    assert binary_overlap_metrics(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == {
        "dice": 1.0, "iou": 1.0
    }

def test_binary_volume_metrics_uses_anisotropic_spacing_for_surface_distance():
    metrics = binary_volume_metrics(prediction, target, spacing_xyz=(1.0, 1.0, 3.0))
    assert metrics["assd"] == pytest.approx(3.0)

def test_aggregate_case_slices_rejects_missing_slice_indices():
    with pytest.raises(ValueError, match="non-contiguous"):
        aggregate_case_slices(records_with_indices_zero_and_two)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_segmentation_metrics.py -q`

Expected: FAIL because `utils.segmentation_metrics` does not exist.

- [ ] **Step 3: Implement the metric interfaces in `utils/segmentation_metrics.py`**

Implement overlap scores using Boolean arrays. Extract surfaces with `scipy.ndimage.binary_erosion`, transform ZYX voxel coordinates to physical space using reversed XYZ spacing, and use `scipy.spatial.cKDTree` for symmetric distances. Return `NaN` ASSD/HD95 whenever either 3D mask is empty.

- [ ] **Step 4: Run metric tests to verify they pass**

Run: `python -m pytest tests/test_segmentation_metrics.py -q`

Expected: PASS.

### Task 2: Metadata-aware evaluation dataset

**Files:**
- Modify: `data/lung_segmentation_dataset.py`
- Modify: `tests/test_lung_segmentation_dataset.py`

**Interfaces:**
- Consumes: dataset root and a split-file path.
- Produces: `LungSegmentationInferenceDataset`, where `__getitem__` returns a mapping with `image`, `mask`, `case_id`, and integer `slice_index`.
- Consumed by: Task 3.

- [ ] **Step 1: Write a failing dataset test**

```python
def test_inference_dataset_returns_case_and_slice_metadata(tmp_path):
    dataset = LungSegmentationInferenceDataset(root, split_file)
    sample = dataset[0]
    assert sample["case_id"] == "CASE-001"
    assert sample["slice_index"] == 0
    assert sample["image"].shape == sample["mask"].shape == (1, 2, 2)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_lung_segmentation_dataset.py::test_inference_dataset_returns_case_and_slice_metadata -q`

Expected: FAIL because `LungSegmentationInferenceDataset` does not exist.

- [ ] **Step 3: Implement `LungSegmentationInferenceDataset`**

Reuse the established PNG normalization and label binarization. Build sorted `(case_id, slice_index, image_path, mask_path)` records from the split file; raise the same clear missing-file errors as the training dataset.

- [ ] **Step 4: Run dataset tests to verify they pass**

Run: `python -m pytest tests/test_lung_segmentation_dataset.py -q`

Expected: PASS.

### Task 3: Config-only UNet inference reports

**Files:**
- Create: `inferences/infer_lung_segmentation.py`
- Create: `tests/test_infer_lung_segmentation.py`

**Interfaces:**
- Consumes: `LungSegmentationInferenceDataset`, Task 1 metric functions, `models.unet.UNet`, and `utils.training.load_checkpoint`.
- Produces: `run_inference(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]`, `per_case_metrics.csv`, and `mean_metrics.json`.

- [ ] **Step 1: Write a failing synthetic inference test**

```python
def test_run_inference_writes_only_per_case_csv_and_mean_json(tmp_path):
    rows, summary = run_inference(synthetic_cfg)
    assert (output_dir / "per_case_metrics.csv").is_file()
    assert (output_dir / "mean_metrics.json").is_file()
    assert {path.name for path in output_dir.iterdir()} == {
        "per_case_metrics.csv", "mean_metrics.json"
    }
    assert rows[0]["dice_3d"] == pytest.approx(1.0)
    assert summary["dice_2d"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_infer_lung_segmentation.py -q`

Expected: FAIL because the inference module does not exist.

- [ ] **Step 3: Implement `run_inference` and report writers**

Use a config dictionary modeled on existing inference scripts. Load the checkpoint and model, use `torch.inference_mode()` with `tqdm`, threshold logits, collect Boolean slice predictions, validate each reconstructed volume against `lung_mask.nii.gz`, and compute per-case metrics. Write CSV rows and JSON after converting non-finite floats to `None`; do not create prediction paths.

- [ ] **Step 4: Run inference tests to verify they pass**

Run: `python -m pytest tests/test_infer_lung_segmentation.py -q`

Expected: PASS.

### Task 4: Regression verification

**Files:**
- Verify: `tests/test_unet.py`, `tests/test_lung_segmentation_dataset.py`, `tests/test_segmentation_metrics.py`, `tests/test_infer_lung_segmentation.py`, `tests/test_train_lung_segmentation.py`

**Interfaces:**
- Consumes: all completed components.
- Produces: a verified inference workflow without regressions.

- [ ] **Step 1: Run the complete test suite**

Run: `python -m pytest -q`

Expected: PASS with no failures.

- [ ] **Step 2: Compile the new Python modules**

Run: `python -m py_compile utils/segmentation_metrics.py data/lung_segmentation_dataset.py inferences/infer_lung_segmentation.py`

Expected: exit code 0.

- [ ] **Step 3: Commit the implementation when Git metadata is available**

The current workspace has no `.git` directory, so report the changed files instead of attempting a commit.
