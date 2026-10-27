from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.lung_segmentation_dataset import (
    LungSegmentationInferenceDataset,
    LungSegmentationDataset,
    create_dataloader,
    visualize_batch,
)


def _write_slice(path: Path, values: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)


def test_dataset_uses_requested_split_and_returns_normalized_binary_pair(tmp_path: Path):
    """A wrong split or non-binary label would train the segmenter on invalid targets."""
    root = tmp_path / "nsclc_radiomics"
    (root / "config").mkdir(parents=True)
    (root / "config" / "train.txt").write_text("CASE-TRAIN\n", encoding="utf-8")
    (root / "config" / "val.txt").write_text("CASE-VAL\n", encoding="utf-8")

    _write_slice(root / "processed" / "CASE-TRAIN" / "images" / "0000.png", [[0, 255], [128, 64]])
    _write_slice(root / "processed" / "CASE-TRAIN" / "lung_labels" / "0000.png", [[0, 17], [255, 0]])
    _write_slice(root / "processed" / "CASE-VAL" / "images" / "0000.png", [[3, 4], [5, 6]])
    _write_slice(root / "processed" / "CASE-VAL" / "lung_labels" / "0000.png", [[1, 1], [1, 1]])

    dataset = LungSegmentationDataset(root, root / "config" / "train.txt")
    image, mask = dataset[0]

    assert len(dataset) == 1
    assert image.shape == mask.shape == (1, 2, 2)
    assert image.dtype == mask.dtype == torch.float32
    assert torch.equal(image, torch.tensor([[[0.0, 1.0], [128 / 255, 64 / 255]]]))
    assert torch.equal(mask, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))


def test_create_dataloader_uses_the_explicit_split_file(tmp_path: Path):
    """A hidden train/val/test default would prevent configurable experiment splits."""
    root = tmp_path / "nsclc_radiomics"
    custom_split = tmp_path / "custom_splits" / "experiment_train.txt"
    custom_split.parent.mkdir(parents=True)
    custom_split.write_text("CASE-CUSTOM\n", encoding="utf-8")
    _write_slice(root / "processed" / "CASE-CUSTOM" / "images" / "0000.png", [[42]])
    _write_slice(root / "processed" / "CASE-CUSTOM" / "lung_labels" / "0000.png", [[255]])

    loader = create_dataloader(root, custom_split, batch_size=1, shuffle=True)

    assert next(iter(loader))[0].shape == (1, 1, 1, 1)


def test_inference_dataset_returns_case_and_slice_metadata(tmp_path: Path):
    root = tmp_path / "dataset"
    split_file = tmp_path / "test.txt"
    split_file.write_text("CASE-001\n", encoding="utf-8")
    _write_slice(root / "processed" / "CASE-001" / "images" / "0000.png", [[1]])
    _write_slice(root / "processed" / "CASE-001" / "lung_labels" / "0000.png", [[255]])

    sample = LungSegmentationInferenceDataset(root, split_file)[0]

    assert sample["case_id"] == "CASE-001"
    assert sample["slice_index"] == 0
    assert sample["image"].shape == sample["mask"].shape == (1, 1, 1)


def test_visualize_batch_draws_slice_and_lung_mask_columns():
    """A swapped or missing panel makes visual quality checks unreliable."""
    images = torch.zeros(2, 1, 2, 2)
    masks = torch.ones(2, 1, 2, 2)

    figure = visualize_batch(images, masks)

    assert len(figure.axes) == 4
    assert [axis.get_title() for axis in figure.axes] == [
        r"$\text{Slice}$",
        r"$\text{Lung mask}$",
        r"$\text{Slice}$",
        r"$\text{Lung mask}$",
    ]
