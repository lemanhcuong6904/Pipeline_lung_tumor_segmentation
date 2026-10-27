import csv
from pathlib import Path

import albumentations as A
import numpy as np
import pytest
import torch
from PIL import Image


def write_slice(root: Path, case: str, name: str, image, lung, tumor):
    for directory, values in (("images", image), ("lung_labels", lung), ("tumor_labels", tumor)):
        path = root / "lung_cropped" / "processed" / case / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)


def test_filters_lung_slices_and_returns_labels_and_csv(tmp_path):
    from data.binary_classification_dataset import BinaryClassificationDataset

    root = tmp_path / "data"
    split = tmp_path / "train.txt"
    split.write_text("A\n", encoding="utf-8")
    write_slice(root, "A", "0000.png", [[0, 255], [64, 128]], [[0, 1], [0, 0]], [[0, 0], [0, 1]])
    write_slice(root, "A", "0001.png", [[1, 2], [3, 4]], [[0, 0], [0, 0]], [[0, 0], [0, 0]])
    write_slice(root, "A", "0002.png", [[0, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 0]])
    write_slice(root, "B", "0000.png", [[1]], [[1]], [[1]])

    dataset = BinaryClassificationDataset(root, split)
    image, label = dataset[0]
    assert len(dataset) == 2
    torch.testing.assert_close(image, torch.tensor([[[0, 1], [64 / 255, 128 / 255]]]))
    assert label.shape == (1,) and label.dtype == torch.float32 and label.item() == 1
    assert dataset[1][1].item() == 0
    path = tmp_path / "labels.csv"
    dataset.save_csv(path)
    with path.open(newline="", encoding="utf-8") as file:
        assert list(csv.DictReader(file)) == [
            {"case_id": "A", "slice": "0000.png", "label": "1"},
            {"case_id": "A", "slice": "0002.png", "label": "0"},
        ]


def test_augmentation_updates_image_and_label_together(tmp_path):
    from data.binary_classification_dataset import BinaryClassificationDataset

    root = tmp_path / "data"
    split = tmp_path / "train.txt"
    split.write_text("A\n", encoding="utf-8")
    write_slice(root, "A", "0000.png", [[255, 0], [0, 0]], [[1, 0], [0, 0]], [[1, 0], [0, 0]])
    transform = A.Compose([A.Crop(x_min=1, y_min=0, x_max=2, y_max=2)])
    dataset = BinaryClassificationDataset(root, split, transform=transform)
    image, label = dataset[0]
    assert image.shape == (1, 2, 1)
    assert image.sum().item() == 0 and label.item() == 0


def test_missing_tumor_mask_fails_instead_of_becoming_negative(tmp_path):
    from data.binary_classification_dataset import BinaryClassificationDataset

    root = tmp_path / "data"
    split = tmp_path / "train.txt"
    split.write_text("A\n", encoding="utf-8")
    write_slice(root, "A", "0000.png", [[1]], [[1]], [[1]])
    (root / "lung_cropped" / "processed" / "A" / "tumor_labels" / "0000.png").unlink()
    with pytest.raises(FileNotFoundError, match="tumor mask"):
        BinaryClassificationDataset(root, split)


def test_loader_train_augmentation_and_validation_no_augmentation(tmp_path):
    from data.binary_classification_dataset import create_dataloader

    root = tmp_path / "data"
    split = tmp_path / "val.txt"
    split.write_text("A\n", encoding="utf-8")
    write_slice(root, "A", "0000.png", [[255, 0], [0, 0]], [[1, 0], [0, 0]], [[1, 0], [0, 0]])
    train = create_dataloader(root, split, batch_size=1, training=True)
    val = create_dataloader(root, split, batch_size=1)
    assert [type(t).__name__ for t in train.dataset.transform.transforms] == ["HorizontalFlip", "Affine", "ElasticTransform"]
    assert val.dataset.transform is None
    image, label = next(iter(val))
    assert image.shape == (1, 1, 2, 2) and label.shape == (1, 1)


def test_csv_loader_does_not_scan_masks_and_rejects_wrong_split(tmp_path, monkeypatch):
    import data.binary_classification_dataset as module

    root = tmp_path / "data"
    split = tmp_path / "train.txt"
    split.write_text("A\n", encoding="utf-8")
    write_slice(root, "A", "0000.png", [[255]], [[1]], [[1]])
    csv_path = tmp_path / "train_labels.csv"
    module.BinaryClassificationDataset(root, split).save_csv(csv_path)
    original_read = module._read_grayscale
    def no_mask_scan(path):
        assert path.parent.name == "images"
        return original_read(path)
    monkeypatch.setattr(module, "_read_grayscale", no_mask_scan)
    loader = module.create_dataloader(root, split, labels_csv=csv_path)
    assert next(iter(loader))[1].item() == 1
    split.write_text("B\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        module.BinaryClassificationDataset(root, split, labels_csv=csv_path)


def test_main_generates_csv_for_all_three_splits(tmp_path, monkeypatch):
    import data.binary_classification_dataset as module
    import matplotlib.pyplot as plt

    monkeypatch.setattr(module, "train_augmentation", lambda: A.Compose([]))
    main = module.main

    root = tmp_path / "data"
    (root / "config").mkdir(parents=True)
    for split, case, label in (("train", "A", 1), ("val", "B", 0), ("test", "C", 1)):
        (root / "config" / f"{split}.txt").write_text(case + "\n", encoding="utf-8")
        write_slice(root, case, "0000.png", [[1]], [[1]], [[label]])
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    main(root)
    assert len(shown) == 1
    assert shown[0].axes[0].get_title() == r"$\text{Label: 1}$"
    plt.close(shown[0])
    for split, case, label in (("train", "A", 1), ("val", "B", 0), ("test", "C", 1)):
        with (root / "lung_cropped" / "config" / f"{split}_labels.csv").open(newline="", encoding="utf-8") as file:
            assert list(csv.DictReader(file)) == [{"case_id": case, "slice": "0000.png", "label": str(label)}]


def test_balanced_sampler_groups_by_original_slice_number_and_is_reproducible(tmp_path):
    from data.binary_classification_dataset import BinaryClassificationDataset, PatientAwareBalancedBatchSampler

    root = tmp_path / "data"
    split = tmp_path / "train.txt"
    split.write_text("A\nB\n", encoding="utf-8")
    for case in ("A", "B"):
        for index, label in ((0, 0), (4, 0), (5, 1), (6, 0), (20, 0)):
            write_slice(root, case, f"{index:04}.png", [[1]], [[1]], [[label]])
    dataset = BinaryClassificationDataset(root, split)
    sampler = PatientAwareBalancedBatchSampler(dataset, batch_size=20, batches_per_epoch=3,
                                               hard_negative_radius=1, seed=42,
                                               positive_fraction=.35, hard_negative_fraction=.2,
                                               easy_negative_fraction=.45)
    sampler.set_epoch(5)
    batches = list(sampler)
    assert len(batches) == 3
    for batch in batches:
        names = [dataset.samples[i][1].stem for i in batch]
        assert sum(name == "0005" for name in names) == 7
        assert sum(name in ("0004", "0006") for name in names) == 4
        assert sum(name in ("0000", "0020") for name in names) == 9
    sampler.set_epoch(5)
    assert list(sampler) == batches
    sampler.set_epoch(6)
    assert list(sampler) != batches


def test_balanced_loader_and_natural_loader_respect_switch(tmp_path):
    from data.binary_classification_dataset import create_dataloader, PatientAwareBalancedBatchSampler

    root = tmp_path / "data"
    split = tmp_path / "train.txt"
    split.write_text("A\n", encoding="utf-8")
    for index, label in ((0, 0), (5, 1), (6, 0)):
        write_slice(root, "A", f"{index:04}.png", [[1]], [[1]], [[label]])
    balanced = create_dataloader(root, split, batch_size=4, training=True, balanced_sampling=True,
                                 hard_negative_radius=1, batches_per_epoch=2)
    natural = create_dataloader(root, split, batch_size=4, training=True, balanced_sampling=False)
    assert isinstance(balanced.batch_sampler, PatientAwareBalancedBatchSampler)
    assert len(balanced) == 2
    assert len(natural) == 1 and not isinstance(natural.batch_sampler, PatientAwareBalancedBatchSampler)
    with pytest.raises(ValueError, match="training"):
        create_dataloader(root, split, balanced_sampling=True)


def test_limited_unbalanced_sampler_covers_all_positive_slices(tmp_path):
    from data.binary_classification_dataset import BinaryClassificationDataset, TumorCoveringBatchSampler

    root = tmp_path / "data"
    split = tmp_path / "train.txt"
    split.write_text("A\n", encoding="utf-8")
    for index, label in enumerate([0, 1, 0, 1, 0, 0, 0, 0]):
        write_slice(root, "A", f"{index:04}.png", [[1]], [[1]], [[label]])
    dataset = BinaryClassificationDataset(root, split)
    sampler = TumorCoveringBatchSampler(dataset, batch_size=2, batches_per_epoch=2)
    indices = [index for batch in sampler for index in batch]
    assert len(indices) == len(set(indices)) == 4
    assert sum(dataset.samples[i][3] for i in indices) == 2
    with pytest.raises(ValueError, match="positive"):
        TumorCoveringBatchSampler(dataset, batch_size=1, batches_per_epoch=1)


def test_balanced_sampling_selects_patients_uniformly_despite_unequal_slice_counts():
    from data.binary_classification_dataset import PatientAwareBalancedBatchSampler

    class Dataset:
        samples = [("A", Path(f"{i:04}.png"), Path("unused"), 1) for i in range(100)]
        samples += [("B", Path("0000.png"), Path("unused"), 1)]
        def __len__(self):
            return len(self.samples)

    dataset = Dataset()
    sampler = PatientAwareBalancedBatchSampler(dataset, 20, 50, positive_fraction=1,
                                               hard_negative_fraction=0, easy_negative_fraction=0)
    draws = [dataset.samples[i][0] for batch in sampler for i in batch]
    assert 400 < draws.count("B") < 600


def test_balanced_sampler_reports_missing_requested_group(tmp_path):
    from data.binary_classification_dataset import BinaryClassificationDataset, PatientAwareBalancedBatchSampler

    root = tmp_path / "data"
    split = tmp_path / "train.txt"
    split.write_text("A\n", encoding="utf-8")
    write_slice(root, "A", "0000.png", [[1]], [[1]], [[1]])
    with pytest.raises(ValueError, match="missing groups"):
        PatientAwareBalancedBatchSampler(BinaryClassificationDataset(root, split), 16)
