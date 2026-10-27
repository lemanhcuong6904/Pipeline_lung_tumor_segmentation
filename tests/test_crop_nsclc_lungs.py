from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from PIL import Image


def test_crop_case_writes_aligned_pngs_and_volumes_with_original_slice_numbers(tmp_path):
    from scripts.crop_nsclc_lungs import crop_case
    from data.binary_classification_dataset import BinaryClassificationDataset

    root = tmp_path / "data"
    source = root / "nifti" / "A"
    source.mkdir(parents=True)
    lung = np.zeros((6, 64, 64), dtype=np.uint8)
    lung[1, 25:28, 20:24] = 1
    lung[3, 20:30, 30:36] = 1
    tumor = np.zeros_like(lung)
    tumor[3, 22:24, 32:34] = 1
    image = np.broadcast_to(np.arange(64, dtype=np.uint8), lung.shape).copy()
    for name, array in (("image", image), ("lung_mask", lung), ("tumor_mask", tumor)):
        volume = sitk.GetImageFromArray(array)
        volume.SetSpacing((2, 2, 3))
        volume.SetOrigin((10, 20, 30))
        sitk.WriteImage(volume, str(source / f"{name}.nii.gz"))

    crop_case(root, "A")
    out = root / "lung_cropped"
    files = sorted((out / "processed" / "A" / "images").glob("*.png"))
    assert [f.name for f in files] == ["0001.png", "0002.png", "0003.png"]
    volumes = [sitk.ReadImage(str(out / "nifti" / "A" / f"{name}.nii.gz"))
               for name in ("image", "lung_mask", "tumor_mask")]
    assert all(v.GetSize() == (256, 256, 3) for v in volumes)
    assert volumes[0].GetOrigin() == pytest.approx((40, 44, 33))
    assert volumes[0].GetSpacing() == pytest.approx((52 / 256, 52 / 256, 3))
    for directory, volume in zip(("images", "lung_labels", "tumor_labels"), volumes):
        array = sitk.GetArrayFromImage(volume)
        for i, file in enumerate(files):
            png = np.asarray(Image.open(out / "processed" / "A" / directory / file.name))
            np.testing.assert_array_equal(png > 0, array[i] > 0)
    split = tmp_path / "train.txt"
    split.write_text("A\n", encoding="utf-8")
    dataset = BinaryClassificationDataset(root, split)
    assert [s[1].name for s in dataset.samples] == ["0001.png", "0003.png"]
    assert [s[3] for s in dataset.samples] == [0, 1]
    assert dataset[0][0].shape == (1, 256, 256)
    with pytest.raises(FileExistsError):
        crop_case(root, "A")
    crop_case(root, "A", overwrite=True)
    assert (out / "nifti" / "A" / "crop.json").is_file()


def test_square_crop_at_image_edge():
    from scripts.crop_nsclc_lungs import lung_crop_region
    mask = np.zeros((5, 64, 64), dtype=np.uint8)
    mask[2:4, :4, :8] = 1
    image = sitk.GetImageFromArray(mask)
    assert lung_crop_region(image) == ((0, 0, 2), (28, 28, 2))


def test_global_bbox_ignores_tiny_components_but_keeps_both_lungs():
    from scripts.crop_nsclc_lungs import lung_crop_region

    mask = np.zeros((10, 64, 64), dtype=np.uint8)
    mask[2:8, 20:30, 10:20] = 1
    mask[3:7, 20:30, 40:50] = 1
    mask[0, 0, 63] = 1
    mask[9, 63, 0] = 1
    index, size = lung_crop_region(sitk.GetImageFromArray(mask), margin_mm=0)
    assert (index, size) == ((10, 5, 2), (40, 40, 6))


def test_filter_preserves_components_over_relative_threshold_and_rejects_noise_only():
    from scripts.crop_nsclc_lungs import clean_lung_mask

    mask = np.zeros((4, 20, 20), dtype=np.uint8)
    mask[1:3, 2:7, 2:7] = 1
    mask[1:3, 12:15, 12:15] = 1
    mask[0, 0, 19] = 1
    result = sitk.GetArrayFromImage(clean_lung_mask(sitk.GetImageFromArray(mask)))
    assert result[1:3, 2:7, 2:7].all()
    assert result[1:3, 12:15, 12:15].all()
    assert not result[0, 0, 19]
    with pytest.raises(ValueError, match="empty"):
        clean_lung_mask(sitk.GetImageFromArray(np.zeros_like(mask)))


def test_rejects_crop_that_would_exclude_tumor(tmp_path):
    from scripts.crop_nsclc_lungs import crop_case

    source = tmp_path / "nifti" / "A"
    source.mkdir(parents=True)
    lung = np.zeros((6, 64, 64), dtype=np.uint8)
    lung[2:5, 20:40, 20:40] = 1
    lung[0, 0, 0] = 1
    tumor = np.zeros_like(lung)
    tumor[0, 0, 0] = 1
    for name, data in (("image", lung), ("lung_mask", lung), ("tumor_mask", tumor)):
        sitk.WriteImage(sitk.GetImageFromArray(data), str(source / f"{name}.nii.gz"))
    with pytest.raises(ValueError, match="exclude tumor"):
        crop_case(tmp_path, "A")


def test_main_uses_configuration_without_cli_arguments(tmp_path, monkeypatch):
    import scripts.crop_nsclc_lungs as module

    (tmp_path / "nifti").mkdir()
    calls = []
    monkeypatch.setattr(module, "DATASET_ROOT", tmp_path)
    monkeypatch.setattr(module, "CASE_IDS", ["A"])
    monkeypatch.setattr(module, "MARGIN_MM", 7.0)
    monkeypatch.setattr(module, "MIN_COMPONENT_FRACTION", 0.05)
    monkeypatch.setattr(module, "OVERWRITE", True)
    monkeypatch.setattr(module, "SKIP_EXISTING", False)
    monkeypatch.setattr(module, "MIN_FREE_GB", 0.0)
    monkeypatch.setattr(module, "crop_case", lambda *a, **kw: calls.append((a, kw)))
    assert module.main() == 0
    assert calls == [((tmp_path, "A", 7.0), {"min_component_fraction": 0.05, "overwrite": True})]


def test_main_rejects_conflicting_existing_output_settings(tmp_path, monkeypatch):
    import scripts.crop_nsclc_lungs as module

    monkeypatch.setattr(module, "DATASET_ROOT", tmp_path)
    monkeypatch.setattr(module, "SKIP_EXISTING", True)
    monkeypatch.setattr(module, "OVERWRITE", True)
    with pytest.raises(ValueError, match="SKIP_EXISTING.*OVERWRITE"):
        module.main()


def test_crop_retries_transient_windows_rename_error(tmp_path, monkeypatch):
    import scripts.crop_nsclc_lungs as module

    source = tmp_path / "nifti" / "R01-008"
    source.mkdir(parents=True)
    lung = np.zeros((3, 16, 16), dtype=np.uint8)
    lung[1, 3:13, 3:13] = 1
    for name, data in (("image", lung), ("lung_mask", lung), ("tumor_mask", np.zeros_like(lung))):
        sitk.WriteImage(sitk.GetImageFromArray(data), str(source / f"{name}.nii.gz"))
    original_rename = Path.rename
    attempts = []

    def transient_rename(path, target):
        if path.name.startswith(".R01-008.") and path.name.endswith(".tmp") and path.parent.name == "processed":
            attempts.append(1)
            if len(attempts) == 1:
                error = PermissionError("temporary access denied")
                error.winerror = 5
                raise error
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", transient_rename)
    module.crop_case(tmp_path, "R01-008")
    assert len(attempts) == 2
    assert (tmp_path / "lung_cropped" / "processed" / "R01-008" / "images" / "0001.png").is_file()
    assert (tmp_path / "lung_cropped" / "nifti" / "R01-008" / "crop.json").is_file()


def test_crop_restores_old_outputs_when_rename_remains_denied(tmp_path, monkeypatch):
    import scripts.crop_nsclc_lungs as module

    source = tmp_path / "nifti" / "R01-008"
    source.mkdir(parents=True)
    lung = np.zeros((3, 16, 16), dtype=np.uint8)
    lung[1, 3:13, 3:13] = 1
    for name, data in (("image", lung), ("lung_mask", lung), ("tumor_mask", np.zeros_like(lung))):
        sitk.WriteImage(sitk.GetImageFromArray(data), str(source / f"{name}.nii.gz"))
    for kind in ("processed", "nifti"):
        old = tmp_path / "lung_cropped" / kind / "R01-008"
        old.mkdir(parents=True)
        (old / "old.txt").write_text(kind)
    original_rename = Path.rename

    def denied_rename(path, target):
        if path.name.startswith(".R01-008.") and path.name.endswith(".tmp") and path.parent.name == "processed":
            error = PermissionError("access denied")
            error.winerror = 5
            raise error
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", denied_rename)
    with pytest.raises(PermissionError):
        module.crop_case(tmp_path, "R01-008", overwrite=True)
    for kind in ("processed", "nifti"):
        assert (tmp_path / "lung_cropped" / kind / "R01-008" / "old.txt").read_text() == kind
