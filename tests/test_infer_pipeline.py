import csv
import json

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from PIL import Image


def test_neighbor_expansion_uses_original_indices_and_stays_in_case():
    from inferences.infer_pipeline import select_slices

    candidates = [10, 12, 18, 25]
    assert select_slices(candidates, {12}, 0) == {12}
    assert select_slices(candidates, {12}, 6) == {10, 12, 18}
    assert select_slices(candidates, set(), 12) == set()
    with pytest.raises(ValueError):
        select_slices(candidates, {12}, -1)


def test_pipeline_reconstructs_full_volume_and_evaluates_missed_tumor(tmp_path, monkeypatch):
    import inferences.infer_pipeline as module

    root = tmp_path / 'data'
    case = 'LUNG-001'
    nifti = root / 'nifti' / case
    nifti.mkdir(parents=True)
    image = np.zeros((3, 16, 16), dtype=np.uint8)
    image[:, 4:12, 4:12] = 255
    tumor = np.zeros_like(image)
    tumor[0, 0, 0] = 1  # outside the predicted lung ROI
    tumor[1, 6, 6] = 1
    for name, array in [('image', image), ('tumor_mask', tumor)]:
        sitk.WriteImage(sitk.GetImageFromArray(array), str(nifti / f'{name}.nii.gz'))
    for folder in ('images', 'images_lung', 'images_mediastinal'):
        path = root / 'processed' / case / folder
        path.mkdir(parents=True)
        for z, img in enumerate(image):
            Image.fromarray(img).save(path / f'{z:04d}.png')
    split = tmp_path / 'test.txt'
    split.write_text(case + '\n')

    forward_batches = {'lung': [], 'classifier': [], 'tumor': []}
    class Lung(torch.nn.Module):
        def forward(self, x):
            forward_batches['lung'].append(len(x))
            return (x - 0.5) * 20
    class Classifier(torch.nn.Module):
        def forward(self, x):
            forward_batches['classifier'].append(len(x))
            result = torch.full((len(x), 1), -10.0, device=x.device)
            result[0] = 10.0
            return result
    class Tumor(torch.nn.Module):
        def forward(self, x):
            forward_batches['tumor'].append(len(x))
            return (x[:, :1] - 0.5) * 20

    monkeypatch.setattr(module, 'load_models', lambda cfg, device: (Lung(), Classifier(), Tumor()))
    cfg = dict(module.CFG, DATASET_ROOT=str(root), SPLIT_PATH=str(split), OUTPUT_ROOT=str(tmp_path / 'out'),
               EXP_NAME='', USE_CLASSIFIER=True, USE_MULTI_WINDOW=True, USE_NEIGHBOR_EXPANSION=True,
               NEIGHBOR_RADII=[0, 1], DEVICE='cpu', LUNG_MARGIN_MM=0,
               MIN_COMPONENT_FRACTION=0.0, BATCH_SIZE=8, AMP_DTYPE='none', MEASURE_INFERENCE=False)
    results = module.run_pipeline(cfg)
    assert set(results) == {'lung_roi_multi_classifier_k0', 'lung_roi_multi_classifier_k1'}
    assert results['lung_roi_multi_classifier_k0']['selected_slices'] == 1
    assert results['lung_roi_multi_classifier_k1']['selected_slices'] == 2
    assert forward_batches == {'lung': [3], 'classifier': [3], 'tumor': [2]}
    for name in results:
        rows = list(csv.DictReader((tmp_path / 'out' / name / 'per_case_metrics.csv').open()))
        assert int(rows[0]['target_foreground_voxels']) == 2
        assert results[name]['case_count'] == 1
        assert (tmp_path / 'out' / name / 'per_slice_metrics.csv').is_file()
        assert json.loads((tmp_path / 'out' / name / 'mean_metrics.json').read_text())['neighbor_expansion_enabled'] is True

    for batches in forward_batches.values():
        batches.clear()
    custom_results = module.run_pipeline(dict(cfg, EXP_NAME='my_experiment', MEASURE_INFERENCE=True, WARMUP_BATCHES=1))
    assert set(custom_results) == {'my_experiment_k0', 'my_experiment_k1'}
    assert (tmp_path / 'out' / 'my_experiment_k1' / 'mean_metrics.json').is_file()
    profile = custom_results['my_experiment_k0']['inference_profile']
    assert set(profile['stages_seconds']) == {'lung', 'classifier', 'tumor'}
    assert forward_batches == {'lung': [3, 3, 3, 3], 'classifier': [3, 3, 3, 3], 'tumor': [1, 1, 2, 2]}
    assert profile['stages_samples'] == {'lung': 3, 'classifier': 3, 'tumor': 1}
    assert custom_results['my_experiment_k1']['inference_profile']['stages_samples'] == {'lung': 3, 'classifier': 3, 'tumor': 2}
    assert 'neighbor_radius_scope' not in profile
    assert 'dice_3d_std' in custom_results['my_experiment_k0']

    monkeypatch.setattr(module, 'load_models', lambda cfg, device: (Lung(), Classifier() if cfg['USE_CLASSIFIER'] else None, Tumor()))
    for classifier, multi, name, selected in [
        (False, False, 'lung_roi_wide', 3),
        (True, False, 'lung_roi_wide_classifier', 1),
        (True, True, 'lung_roi_multi_classifier', 1),
        (False, True, 'lung_roi_multi', 3),
    ]:
        summary = module.run_pipeline(dict(cfg, USE_CLASSIFIER=classifier, USE_MULTI_WINDOW=multi, USE_NEIGHBOR_EXPANSION=False))[name]
        assert summary['selected_slices'] == selected

    class EmptyLung(torch.nn.Module):
        def forward(self, x): return torch.full_like(x, -10.0)

    monkeypatch.setattr(module, 'load_models', lambda cfg, device: (EmptyLung(), None, Tumor()))
    summary = module.run_pipeline(dict(cfg, USE_CLASSIFIER=False, USE_MULTI_WINDOW=False, USE_NEIGHBOR_EXPANSION=False))['lung_roi_wide']
    assert summary['selected_slices'] == 0
    assert summary['dice_3d'] == 0


def test_neighbor_expansion_requires_classifier():
    from inferences.infer_pipeline import CFG, run_pipeline

    with pytest.raises(ValueError, match='USE_CLASSIFIER'):
        run_pipeline(dict(CFG, USE_CLASSIFIER=False, USE_NEIGHBOR_EXPANSION=True))


@pytest.mark.parametrize('name', ['..', '../outside', r'..\outside'])
def test_experiment_name_rejects_paths(name):
    from inferences.infer_pipeline import CFG, run_pipeline

    with pytest.raises(ValueError, match='EXP_NAME'):
        run_pipeline(dict(CFG, EXP_NAME=name))
