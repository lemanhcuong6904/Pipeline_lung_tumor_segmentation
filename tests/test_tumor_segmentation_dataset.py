import albumentations as A
import numpy as np
import pytest
import torch
from PIL import Image


def write_slice(root, name, lung, tumor):
    arrays = {
        "images": [[0, 255], [64, 128]],
        "images_lung": [[1, 2], [3, 4]],
        "images_mediastinal": [[10, 20], [30, 40]],
        "lung_labels": lung,
        "tumor_labels": tumor,
    }
    for directory, values in arrays.items():
        path = root / "lung_cropped" / "processed" / "LUNG-001" / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.array(values, dtype=np.uint8)).save(path)


def test_lung_filter_window_order_and_binary_masks(tmp_path):
    from data.tumor_segmentation_dataset import TumorSegmentationDataset

    split = tmp_path / "train.txt"
    split.write_text("LUNG-001\n", encoding="utf-8")
    write_slice(tmp_path, "0000.png", [[1, 0], [0, 0]], [[0, 0], [0, 7]])
    write_slice(tmp_path, "0001.png", [[0, 0], [0, 0]], [[0, 0], [0, 0]])
    write_slice(tmp_path, "0002.png", [[1, 0], [0, 0]], [[0, 0], [0, 0]])
    dataset = TumorSegmentationDataset(tmp_path, split)
    assert len(dataset) == 2
    image, mask = dataset[0]
    expected = (
        torch.tensor(
            [[[0, 255], [64, 128]], [[1, 2], [3, 4]], [[10, 20], [30, 40]]],
            dtype=torch.float32,
        )
        / 255
    )
    torch.testing.assert_close(image, expected)
    assert torch.equal(mask, torch.tensor([[[0.0, 0.0], [0.0, 1.0]]]))
    assert dataset[1][1].sum() == 0


def test_geometric_transform_is_shared_by_all_windows_and_tumor(tmp_path):
    from data.tumor_segmentation_dataset import TumorSegmentationDataset

    split = tmp_path / "train.txt"
    split.write_text("LUNG-001\n", encoding="utf-8")
    write_slice(tmp_path, "0000.png", [[1, 0], [0, 0]], [[0, 0], [0, 7]])
    dataset = TumorSegmentationDataset(
        tmp_path, split, transform=A.Compose([A.HorizontalFlip(p=1)])
    )
    image, mask = dataset[0]
    torch.testing.assert_close(
        image[:, 0, :], torch.tensor([[255.0, 0.0], [2.0, 1.0], [20.0, 10.0]]) / 255
    )
    assert torch.equal(mask, torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]))


def test_missing_window_is_reported(tmp_path):
    from data.tumor_segmentation_dataset import TumorSegmentationDataset

    split = tmp_path / "train.txt"
    split.write_text("LUNG-001\n", encoding="utf-8")
    write_slice(tmp_path, "0000.png", [[1, 0], [0, 0]], [[0, 0], [0, 0]])
    (tmp_path / "lung_cropped/processed/LUNG-001/images_lung/0000.png").unlink()
    with pytest.raises(FileNotFoundError, match="images_lung"):
        TumorSegmentationDataset(tmp_path, split)


@pytest.mark.parametrize("mode,expected_count", [("baseline", 3), ("lung_roi", 2)])
def test_single_window_modes_use_correct_slice_set_without_extra_windows(
    tmp_path, mode, expected_count
):
    from data.tumor_segmentation_dataset import (
        TumorSegmentationDataset,
        create_dataloader,
    )

    split = tmp_path / "split.txt"
    split.write_text("LUNG-001\n", encoding="utf-8")
    for i, has_lung in enumerate((True, False, True)):
        write_slice(
            tmp_path,
            f"{i:04d}.png",
            [[int(has_lung), 0], [0, 0]],
            [[0, 0], [0, i == 0]],
        )
    cropped = tmp_path / "lung_cropped/processed/LUNG-001"
    import shutil

    shutil.copytree(cropped, tmp_path / "processed/LUNG-001")
    for folder in (cropped, tmp_path / "processed/LUNG-001"):
        shutil.rmtree(folder / "images_lung")
        shutil.rmtree(folder / "images_mediastinal")
    if mode == "baseline":
        shutil.rmtree(tmp_path / "processed/LUNG-001/lung_labels")
    dataset = TumorSegmentationDataset(
        tmp_path, split, mode=mode, transform=A.Compose([A.HorizontalFlip(p=1)])
    )
    assert len(dataset) == expected_count
    image, mask = dataset[0]
    assert image.shape == (1, 2, 2) and mask.shape == (1, 2, 2)
    torch.testing.assert_close(image[0, 0], torch.tensor([1.0, 0.0]))
    loader = create_dataloader(
        tmp_path,
        split,
        mode=mode,
        training=True,
        balanced_sampling=True,
        batch_size=2,
        batches_per_epoch=1,
        return_metadata=True,
        hard_negative_radius=0,
        positive_fraction=0.5,
        hard_negative_fraction=0,
        easy_negative_fraction=0.5,
    )
    assert next(iter(loader))["image"].shape == (2, 1, 2, 2)


def test_loader_train_transforms_and_visualization(tmp_path):
    import matplotlib.pyplot as plt

    from data.tumor_segmentation_dataset import create_dataloader, visualize_batch

    split = tmp_path / "train.txt"
    split.write_text("LUNG-001\n", encoding="utf-8")
    write_slice(tmp_path, "0000.png", [[1, 0], [0, 0]], [[0, 0], [0, 7]])
    train = create_dataloader(tmp_path, split, batch_size=1, training=True)
    validation = create_dataloader(tmp_path, split, batch_size=1)
    transforms = train.dataset.transform.transforms
    assert [type(t).__name__ for t in transforms] == [
        "HorizontalFlip",
        "Affine",
        "ElasticTransform",
    ]
    assert [t.p for t in transforms] == [0.3, 0.7, 0.2]
    assert validation.dataset.transform is None
    images, masks = next(iter(validation))
    fig = visualize_batch(images, masks)
    assert len(fig.axes) == 4
    plt.close(fig)
