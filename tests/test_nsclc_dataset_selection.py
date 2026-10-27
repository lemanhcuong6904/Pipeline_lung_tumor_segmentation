import ast
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def load_with_dataset(filename, dataset):
    """Execute the module with its editable DATASET value changed."""
    path = Path(filename).resolve()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "DATASET" for target in node.targets)]
    assert len(assignments) == 1, filename
    assignments[0].value = ast.copy_location(ast.Constant(dataset), assignments[0].value)
    namespace = {"__file__": str(path), "__name__": "dataset_selection_test"}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def test_dataset_variable_selects_matching_paths_in_all_scripts(monkeypatch):
    monkeypatch.setenv("NSCLC_DATASET", "invalid_environment_value")
    files = [
        "data/binary_classification_dataset.py",
        "data/lung_segmentation_dataset.py",
        "data/tumor_segmentation_dataset.py",
        "trainers/train_binary_classification_pytorch.py",
        "trainers/train_lung_segmentation.py",
        "trainers/train_tumor_segmentation.py",
        "inferences/infer_binary_classification.py",
        "inferences/infer_lung_segmentation.py",
        "inferences/infer_tumor_segmentation.py",
        "inferences/infer_pipeline.py",
    ]
    for dataset in ("nsclc_radiomics", "nsclc_radiogenomics"):
        for name in files:
            module = load_with_dataset(name, dataset)
            assert Path(module["DATASET_ROOT"]).name == dataset, name
            if "CFG" not in module:
                continue
            cfg = module["CFG"]
            for key in ("DATASET_ROOT", "TRAIN_SPLIT", "VAL_SPLIT", "SPLIT_PATH", "TRAIN_LABELS_CSV", "VAL_LABELS_CSV", "LABELS_CSV", "NIFTI_ROOT", "FULL_NIFTI_ROOT", "CROPPED_NIFTI_ROOT"):
                if key in cfg:
                    assert dataset in str(cfg[key]), (name, key)
            if dataset == "nsclc_radiogenomics":
                output = cfg.get("OUTPUT_DIR", cfg.get("OUTPUT_ROOT", cfg.get("SAVE_ROOT")))
                assert dataset in str(output), name
                for key in ("CHECKPOINT_PATH", "LUNG_CHECKPOINT", "CLASSIFIER_CHECKPOINT", "SINGLE_WINDOW_CHECKPOINT", "MULTI_WINDOW_CHECKPOINT"):
                    if key in cfg:
                        assert dataset in str(cfg[key]), (name, key)
                if name.endswith("train_lung_segmentation.py"):
                    assert not cfg["RESUME_PATH"]


def test_generate_labels_for_radiogenomics_splits(tmp_path):
    from data.binary_classification_dataset import generate_split_csvs

    root = tmp_path / "nsclc_radiogenomics"
    (root / "config").mkdir(parents=True)
    case = root / "lung_cropped" / "processed" / "R01-005"
    for folder in ("images", "lung_labels", "tumor_labels"):
        (case / folder).mkdir(parents=True)
        value = 0 if folder == "tumor_labels" else 255
        Image.fromarray(np.full((4, 4), value, dtype=np.uint8)).save(case / folder / "0001.png")
    for split in ("train", "val", "test"):
        (root / "config" / f"{split}.txt").write_text("R01-005\n")
    generate_split_csvs(root)
    for split in ("train", "val", "test"):
        with (root / "lung_cropped" / "config" / f"{split}_labels.csv").open(newline="") as f:
            assert list(csv.DictReader(f)) == [{"case_id": "R01-005", "slice": "0001.png", "label": "0"}]


def test_radiogenomics_missing_case_is_not_silently_skipped(tmp_path):
    import pytest
    from data.lung_segmentation_dataset import LungSegmentationDataset, LungSegmentationInferenceDataset

    case = tmp_path / "processed" / "R01-001"
    for folder in ("images", "lung_labels"):
        (case / folder).mkdir(parents=True)
        Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(case / folder / "0000.png")
    split = tmp_path / "test.txt"
    split.write_text("R01-001\nR01-005\n")
    for dataset in (LungSegmentationDataset, LungSegmentationInferenceDataset):
        with pytest.raises(FileNotFoundError, match="R01-005"):
            dataset(tmp_path, split)
