import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from PIL import Image


def prepare_case(root, ct, case_id="LUNG-001", z_spacing=None):
    from scripts.preprocess_nsclc import normalize_ct_image, resize_full_ct_xy
    from scripts.crop_nsclc_lungs import crop_and_resize

    full = resize_full_ct_xy(normalize_ct_image(ct, -700, 500), False, z_spacing)
    index, size = (20, 20, 1), (100, 100, 2)
    cropped = crop_and_resize(full, index, size, is_label=False)
    for prefix, volume, start in ((root, full, 0), (root / 'lung_cropped', cropped, 1)):
        image_dir = prefix / 'processed' / case_id / 'images'
        nifti_dir = prefix / 'nifti' / case_id
        image_dir.mkdir(parents=True)
        nifti_dir.mkdir(parents=True)
        sitk.WriteImage(volume, str(nifti_dir / 'image.nii.gz'))
        for z, image in enumerate(sitk.GetArrayFromImage(volume), start=start):
            Image.fromarray(image).save(image_dir / f'{z:04d}.png')
    (root / 'lung_cropped' / 'nifti' / case_id / 'crop.json').write_text(
        json.dumps({'source_index_xyz': index, 'source_size_xyz': size}), encoding='utf-8')


def test_windows_preserve_slice_mapping_and_unclipped_hu(tmp_path):
    from scripts.extract_nsclc_windows import extract_case

    ct = sitk.GetImageFromArray(np.stack([np.full((256, 256), value, np.int16)
                                         for value in (-1350, -600, 40, 500)]))
    prepare_case(tmp_path, ct)
    extract_case(ct, tmp_path, 'LUNG-001')
    full = tmp_path / 'processed' / 'LUNG-001'
    cropped = tmp_path / 'lung_cropped' / 'processed' / 'LUNG-001'
    assert [p.name for p in sorted((cropped / 'images_lung').glob('*.png'))] == ['0001.png', '0002.png']
    assert [int(np.asarray(Image.open(full / 'images_lung' / f'{z:04d}.png'))[100, 100])
            for z in range(4)] == [0, 128, 236, 255]
    assert [int(np.asarray(Image.open(cropped / 'images_mediastinal' / f'{z:04d}.png'))[100, 100])
            for z in (1, 2)] == [0, 128]
    with pytest.raises(FileExistsError):
        extract_case(ct, tmp_path, 'LUNG-001')


def test_rejects_geometry_mismatch_before_writing(tmp_path):
    from scripts.extract_nsclc_windows import extract_case

    ct = sitk.GetImageFromArray(np.zeros((4, 256, 256), dtype=np.int16))
    prepare_case(tmp_path, ct)
    ct.SetOrigin((10, 0, 0))
    with pytest.raises(ValueError, match='geometry'):
        extract_case(ct, tmp_path, 'LUNG-001')
    assert not (tmp_path / 'processed' / 'LUNG-001' / 'images_lung').exists()


def test_radiogenomics_windows_match_resampled_z_grid(tmp_path):
    from scripts.extract_nsclc_windows import extract_case

    ct = sitk.GetImageFromArray(np.stack([np.full((256, 256), v, np.int16)
                                         for v in (-600, 40, 500, 500)]))
    ct.SetSpacing((1, 1, 4))
    prepare_case(tmp_path, ct, "R01-001", 2.0)
    extract_case(ct, tmp_path, "R01-001")
    full = tmp_path / "processed" / "R01-001" / "images_lung"
    assert len(list(full.glob("*.png"))) == 7
    assert int(np.asarray(Image.open(full / "0002.png"))[100, 100]) == 236


def test_main_discovers_radiogenomics_case(tmp_path, monkeypatch):
    import scripts.extract_nsclc_windows as module

    raw = tmp_path / "raw"
    data = tmp_path / "data"
    (raw / "R01-001").mkdir(parents=True)
    (data / "processed" / "R01-001").mkdir(parents=True)
    monkeypatch.setattr(module, "RAW_ROOT", raw)
    monkeypatch.setattr(module, "DATASET_ROOT", data)
    monkeypatch.setattr(module, "CASE_IDS", None)
    monkeypatch.setattr(module, "_load_ct", lambda path: (path, []))
    found = []
    monkeypatch.setattr(module, "extract_case", lambda ct, root, case_id, **kw: found.append(case_id))
    assert module.main() == 0
    assert found == ["R01-001"]


def test_extract_retries_temporary_windows_access_denial(tmp_path, monkeypatch):
    import scripts.extract_nsclc_windows as module
    import scripts.crop_nsclc_lungs as crop

    ct = sitk.GetImageFromArray(np.zeros((4, 256, 256), dtype=np.int16))
    prepare_case(tmp_path, ct, "R01-005")
    original = Path.rename
    attempts = []

    def rename(path, target):
        if path.name.startswith(".images_mediastinal.") and path.name.endswith(".tmp") and path.parent == tmp_path / "processed" / "R01-005":
            attempts.append(1)
            if len(attempts) == 1:
                error = PermissionError("temporary access denied")
                error.winerror = 5
                raise error
        return original(path, target)

    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr(crop.time, "sleep", lambda seconds: None)
    module.extract_case(ct, tmp_path, "R01-005")
    assert len(attempts) == 2
    for prefix in (tmp_path, tmp_path / "lung_cropped"):
        for name in ("images_lung", "images_mediastinal"):
            assert list((prefix / "processed" / "R01-005" / name).glob("*.png"))


def test_extract_restores_existing_windows_after_persistent_denial(tmp_path, monkeypatch):
    import scripts.extract_nsclc_windows as module
    import scripts.crop_nsclc_lungs as crop

    ct = sitk.GetImageFromArray(np.zeros((4, 256, 256), dtype=np.int16))
    prepare_case(tmp_path, ct, "R01-005")
    for prefix in (tmp_path, tmp_path / "lung_cropped"):
        for name in ("images_lung", "images_mediastinal"):
            folder = prefix / "processed" / "R01-005" / name
            folder.mkdir()
            (folder / "old.txt").write_text(name)
    original = Path.rename

    def rename(path, target):
        if path.name.startswith(".images_mediastinal.") and path.name.endswith(".tmp") and path.parent == tmp_path / "processed" / "R01-005":
            error = PermissionError("access denied")
            error.winerror = 5
            raise error
        return original(path, target)

    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr(crop.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        module.extract_case(ct, tmp_path, "R01-005", overwrite=True)
    for prefix in (tmp_path, tmp_path / "lung_cropped"):
        for name in ("images_lung", "images_mediastinal"):
            assert (prefix / "processed" / "R01-005" / name / "old.txt").read_text() == name
