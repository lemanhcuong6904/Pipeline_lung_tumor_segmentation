import csv
import json

import numpy as np
import pytest
import torch
from PIL import Image


class MeanLogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(-0.5))

    def forward(self, images):
        return (images.mean(dim=(1, 2, 3)) + self.offset).unsqueeze(1) * 10


def test_inference_writes_slice_and_case_reports_with_global_metrics(tmp_path, monkeypatch):
    import inferences.infer_binary_classification as inference

    root = tmp_path / "dataset"
    split = tmp_path / "test.txt"
    split.write_text("A\nB\n", encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text("case_id,slice,label\nA,0000.png,1\nA,0001.png,0\nB,0000.png,1\nB,0001.png,0\n", encoding="utf-8")
    for case, name, pixel in (("A", "0000.png", 255), ("A", "0001.png", 255),
                               ("B", "0000.png", 0), ("B", "0001.png", 0)):
        path = root / "lung_cropped" / "processed" / case / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((4, 4), pixel, dtype=np.uint8)).save(path)
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state": MeanLogitModel().state_dict()}, checkpoint)
    monkeypatch.setattr(inference, "ResNet50", MeanLogitModel)
    output = tmp_path / "output"
    cfg = inference.CFG | {"CHECKPOINT_PATH": str(checkpoint), "DATASET_ROOT": str(root),
                           "SPLIT_PATH": str(split), "LABELS_CSV": str(labels),
                           "OUTPUT_DIR": str(output), "DEVICE": "cpu", "BATCH_SIZE": 3,
                           "NUM_WORKERS": 0, "PIN_MEMORY": False, "THRESHOLD": 0.5,
                           "USE_K_EXPANSION": True, "K_EXPANSION": [2],
                           "PLOT_THRESHOLD_SWEEP": True}
    rows, summary = inference.run_inference(cfg)
    assert [(r["case_id"], r["slice"], r["target"], r["prediction"]) for r in rows] == [
        ("A", "0000.png", 1, 1), ("A", "0001.png", 0, 1),
        ("B", "0000.png", 1, 0), ("B", "0001.png", 0, 0),
    ]
    assert rows[0]["probability"] == pytest.approx(torch.sigmoid(torch.tensor(5.)).item())
    assert {k: summary[k] for k in ("tp", "fp", "fn", "tn")} == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert {k: summary[k] for k in ("acc", "precision", "recall", "f1", "specificity")} == pytest.approx(
        {"acc": .5, "precision": .5, "recall": .5, "f1": .5, "specificity": .5})
    assert summary["slice_count"] == 4 and summary["case_count"] == 2
    with (output / "slice_predictions.csv").open(newline="", encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == 4
    with (output / "per_case_metrics.csv").open(newline="", encoding="utf-8") as file:
        case_rows = list(csv.DictReader(file))
    assert [(row["case_id"], row["slice_count"]) for row in case_rows] == [("A", "2"), ("B", "2")]
    from utils.inference_reporting import rounded_output
    assert json.loads((output / "mean_metrics.json").read_text()) == rounded_output(summary)
    expanded_summary = json.loads((output / 'k2' / 'mean_metrics.json').read_text())
    assert expanded_summary['neighbor_radius'] == 2
    assert (output / 'k2' / 'slice_predictions.csv').is_file()
    assert (output / 'threshold_sweep.png').is_file()


def test_threshold_sweep_uses_inclusive_probability_and_global_counts():
    from inferences.infer_binary_classification import _threshold_sweep

    rows = [{'probability': p, 'target': t} for p, t in [(0.0, 1), (0.5, 1), (0.5, 0), (1.0, 0)]]
    sweep = _threshold_sweep(rows)
    assert len(sweep) == 101
    assert sweep[0]['recall'] == 1.0
    assert sweep[50] == pytest.approx({'threshold': 0.5, 'acc': .25, 'precision': 1/3, 'recall': .5, 'f1': .4})
    assert sweep[-1] == pytest.approx({'threshold': 1.0, 'acc': .25, 'precision': 0, 'recall': 0, 'f1': 0})


def test_inference_rejects_invalid_threshold_and_missing_checkpoint(tmp_path):
    from inferences.infer_binary_classification import CFG, run_inference

    config = CFG | {"CHECKPOINT_PATH": str(tmp_path / "missing.pt"), "DEVICE": "cpu"}
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        run_inference(config)
    with pytest.raises(ValueError, match="threshold"):
        run_inference(config | {"THRESHOLD": 1.1})


def test_model_is_selected_from_checkpoint(monkeypatch):
    import inferences.infer_binary_classification as inference

    class Custom(torch.nn.Module):
        pass

    class Torchvision(torch.nn.Module):
        def __init__(self, pretrained=True):
            super().__init__()
            assert pretrained is False

    monkeypatch.setattr(inference, 'ResNet50', Custom)
    monkeypatch.setattr(inference, 'ResNet50PyTorch', Torchvision)
    assert isinstance(inference._model_from_state({'stem.0.weight': torch.zeros(1)}), Custom)
    assert isinstance(inference._model_from_state({'backbone.conv1.weight': torch.zeros(1)}), Torchvision)


def test_main_rounds_console_output(monkeypatch, capsys):
    import inferences.infer_binary_classification as inference

    monkeypatch.setattr(inference, 'run_inference', lambda cfg: ([], {'acc': 0.1234567}))
    inference.main()
    assert json.loads(capsys.readouterr().out)['acc'] == 0.1235


def test_expansion_uses_slice_numbers_within_each_case():
    from inferences.infer_binary_classification import expand_predictions, _expanded_metrics

    rows = [
        {'case_id': 'A', 'slice': '0010.png', 'prediction': 1, 'target': 1},
        {'case_id': 'A', 'slice': '0012.png', 'prediction': 0, 'target': 1},
        {'case_id': 'A', 'slice': '0015.png', 'prediction': 0, 'target': 0},
        {'case_id': 'B', 'slice': '0010.png', 'prediction': 0, 'target': 1},
    ]
    expanded = expand_predictions(rows, 2)
    assert [r['prediction'] for r in expanded] == [1, 1, 0, 0]
    assert rows[1]['prediction'] == 0
    _, summary = _expanded_metrics(expanded, 0.5, 2)
    assert (summary['tp'], summary['fp'], summary['fn'], summary['tn']) == (2, 0, 1, 1)
    with pytest.raises(ValueError):
        expand_predictions(rows, -1)
