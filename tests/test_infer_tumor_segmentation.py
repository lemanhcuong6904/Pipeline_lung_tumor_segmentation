import csv
import json

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class FakeModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        assert kwargs["backbone_pretrained"] is False

    def forward(self, images):
        return images[:, :1] * 20 - 10


class FakeDataset(Dataset):
    def __init__(self, channels=3):
        self.channels = channels

    def __len__(self):
        return 2

    def __getitem__(self, index):
        image = torch.zeros(self.channels, 4, 4)
        mask = torch.zeros(1, 4, 4)
        if index == 0:
            image[0, 1, 1] = 1
            mask[0, 1, 1] = 1
        return {
            "image": image,
            "mask": mask,
            "case_id": "A",
            "slice_index": (10, 12)[index],
        }


def test_legacy_checkpoint_infers_attention_options_from_weights():
    from inferences.infer_tumor_segmentation import checkpoint_model_config

    base = {"model_config": {"backbone_pretrained": True}, "model_state": {}}
    cfg = checkpoint_model_config(base)
    assert cfg["backbone_pretrained"] is False
    assert cfg["cross_attention_enabled"] is False
    assert cfg["channel_attention_enabled"] is False
    cfg = checkpoint_model_config(
        base
        | {
            "model_state": {
                "cross_window_attention.position": torch.zeros(1),
                "channel_attention.excitation.0.weight": torch.zeros(1),
            }
        }
    )
    assert cfg["cross_attention_enabled"] is True
    assert cfg["channel_attention_enabled"] is True
    cfg = checkpoint_model_config(
        {
            "model_config": {"cross_attention_enabled": False},
            "model_state": {"cross_window_attention.position": torch.zeros(1)},
        }
    )
    assert cfg["cross_attention_enabled"] is False

    cfg = checkpoint_model_config(
        {
            "model_config": {
                "window_attention_enabled": False,
                "window_attention_reduction": 16,
            },
            "model_state": {},
        }
    )
    assert (
        "window_attention_enabled" not in cfg
        and "window_attention_reduction" not in cfg
    )
    with pytest.raises(ValueError, match="window attention"):
        checkpoint_model_config(
            {"model_config": {"window_attention_enabled": True}, "model_state": {}}
        )


@pytest.mark.parametrize("mode,channels", [("multi_window", 3), ("lung_roi", 1)])
def test_inference_reconstructs_z_and_writes_2d_3d_reports(
    tmp_path, monkeypatch, mode, channels
):
    import inferences.infer_tumor_segmentation as module

    root = tmp_path / "nifti" / "A"
    root.mkdir(parents=True)
    truth = np.zeros((3, 4, 4), dtype=np.uint8)
    truth[0:2, 1, 1] = 1  # The omitted middle slice must remain in the 3D target.
    reference = sitk.GetImageFromArray(truth)
    reference.SetSpacing((1.0, 1.5, 3.0))
    sitk.WriteImage(reference, str(root / "tumor_mask.nii.gz"))
    (root / "crop.json").write_text(
        json.dumps({"source_index_xyz": [0, 0, 10], "source_size_xyz": [4, 4, 3]})
    )
    split = tmp_path / "test.txt"
    split.write_text("A\n", encoding="utf-8")
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": {},
            "model_config": {"backbone_pretrained": True, "in_channels": channels},
            "config": {"ABLATION": mode},
        },
        checkpoint,
    )
    monkeypatch.setattr(module, "TransUNet", FakeModel)
    monkeypatch.setattr(module, "SingleWindowTransUNet", FakeModel)

    def build_loader(*args, **kwargs):
        assert kwargs["mode"] == mode
        return torch.utils.data.DataLoader(FakeDataset(channels), batch_size=2)

    monkeypatch.setattr(module, "create_dataloader", build_loader)
    cfg = module.CFG | {
        "CHECKPOINT_PATH": str(checkpoint),
        "DATASET_ROOT": str(tmp_path),
        "SPLIT_PATH": str(split),
        "CROPPED_NIFTI_ROOT": str(root.parent),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "DEVICE": "cpu",
        "PLOT_THRESHOLD_SWEEP": True,
    }
    rows, summary = module.run_inference(cfg)
    assert rows[0]["dice_2d"] == 1 and rows[0]["iou_2d"] == 1
    assert rows[0]["dice_3d"] == pytest.approx(2 / 3)
    assert rows[0]["iou_3d"] == pytest.approx(1 / 2)
    assert rows[0]["assd_3d"] == pytest.approx(1)
    assert rows[0]["hd95_3d"] == pytest.approx(2.7)
    assert summary["slice_count"] == 2 and summary["case_count"] == 1
    assert summary["hd95_3d_valid_cases"] == 1
    assert (tmp_path / "output" / "threshold_sweep.png").is_file()
    with (tmp_path / "output" / "threshold_sweep.csv").open(newline="") as f:
        sweep = list(csv.DictReader(f))
    assert len(sweep) == 101
    assert float(sweep[0]["dice_3d"]) == pytest.approx(2 / 34, abs=0.0001)
    assert float(sweep[0]["iou_3d"]) == pytest.approx(1 / 33, abs=0.0001)
    assert float(sweep[50]["dice_3d"]) == pytest.approx(2 / 3, abs=0.0001)
    assert float(sweep[50]["iou_3d"]) == pytest.approx(1 / 2, abs=0.0001)
    assert float(sweep[100]["dice_3d"]) == 0
    assert json.loads((tmp_path / "output" / "mean_metrics.json").read_text())[
        "dice_3d"
    ] == 0.6667
    with (tmp_path / "output" / "per_slice_metrics.csv").open(newline="") as f:
        assert [row["slice_index"] for row in csv.DictReader(f)] == ["10", "12"]


def test_inference_rejects_invalid_threshold_before_loading(tmp_path):
    from inferences.infer_tumor_segmentation import CFG, run_inference

    with pytest.raises(ValueError, match="threshold"):
        run_inference(
            CFG | {"THRESHOLD": 1.1, "CHECKPOINT_PATH": str(tmp_path / "missing")}
        )


def test_baseline_inference_uses_full_nifti_and_single_channel(tmp_path, monkeypatch):
    import inferences.infer_tumor_segmentation as module

    root = tmp_path / "nifti" / "A"
    root.mkdir(parents=True)
    target = np.zeros((2, 4, 4), dtype=np.uint8)
    target[0, 1, 1] = 1
    sitk.WriteImage(sitk.GetImageFromArray(target), str(root / "tumor_mask.nii.gz"))
    split = tmp_path / "test.txt"
    split.write_text("A\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": {},
            "model_config": {"in_channels": 1},
            "config": {"ABLATION": "baseline"},
        },
        checkpoint,
    )

    class OneChannelDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            image = torch.from_numpy(target[index : index + 1].astype(np.float32))
            return {"image": image, "mask": image, "case_id": "A", "slice_index": index}

    captured = []
    monkeypatch.setattr(module, "SingleWindowTransUNet", FakeModel)

    def build_loader(*args, **kwargs):
        captured.append(kwargs["mode"])
        return torch.utils.data.DataLoader(OneChannelDataset(), batch_size=2)

    monkeypatch.setattr(module, "create_dataloader", build_loader)
    cfg = module.CFG | {
        "CHECKPOINT_PATH": str(checkpoint),
        "DATASET_ROOT": str(tmp_path),
        "SPLIT_PATH": str(split),
        "FULL_NIFTI_ROOT": str(root.parent),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "PLOT_THRESHOLD_SWEEP": False,
        "DEVICE": "cpu",
    }
    rows, summary = module.run_inference(cfg)
    assert captured == ["baseline"]
    assert not (tmp_path / "output" / "threshold_sweep.png").exists()
    assert not (tmp_path / "output" / "threshold_sweep.csv").exists()
    assert summary["case_count"] == 1 and rows[0]["volume_slice_count"] == 2
    assert rows[0]["dice_3d"] == 1




def test_threshold_sweep_averages_cases_and_includes_threshold_boundary():
    from inferences.infer_tumor_segmentation import _threshold_sweep

    # Bins 51 and 1 represent probabilities 0.5 and 0.0 respectively.
    a_pred = np.zeros(102, dtype=np.int64)
    a_true = np.zeros(102, dtype=np.int64)
    a_pred[51] = a_true[51] = 1
    b_pred = np.zeros(102, dtype=np.int64)
    b_true = np.zeros(102, dtype=np.int64)
    b_pred[1] = 1
    cases = {"A": {"target": np.array([1]), "sweep_predictions": a_pred, "sweep_intersections": a_true},
             "B": {"target": np.array([0]), "sweep_predictions": b_pred, "sweep_intersections": b_true}}
    sweep = _threshold_sweep(cases)
    assert sweep[0] == pytest.approx({"threshold": 0, "dice_3d": .5, "iou_3d": .5})
    assert sweep[50] == pytest.approx({"threshold": .5, "dice_3d": 1, "iou_3d": 1})
    assert sweep[51] == pytest.approx({"threshold": .51, "dice_3d": .5, "iou_3d": .5})
