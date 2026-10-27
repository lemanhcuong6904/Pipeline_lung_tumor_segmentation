import json

import numpy as np
import pytest
import torch
from PIL import Image


def make_case(root, case, labels):
    folder = root / "lung_cropped" / "processed" / case
    for index, positive in enumerate(labels):
        mask = np.zeros((32, 32), dtype=np.uint8)
        if positive:
            mask[10:15, 12:17] = 255
        for directory, array in (
            ("images", np.full((32, 32), 90, np.uint8)),
            ("images_lung", np.full((32, 32), 100, np.uint8)),
            ("images_mediastinal", np.full((32, 32), 110, np.uint8)),
            ("lung_labels", np.full((32, 32), 255, np.uint8)),
            ("tumor_labels", mask),
        ):
            path = folder / directory / f"{index:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(path)


def test_loss_matches_reference_weighted_batch_dice_and_empty_target():
    from utils.tumor_segmentation_loss import BCEWeightedDiceLoss

    criterion = BCEWeightedDiceLoss(1.2, 1.0, 1.0, 1.2, 1e-6, torch.device("cpu"))
    logits = torch.zeros((1, 1, 1, 2), requires_grad=True)
    for target, expected_dice_loss in (
        (torch.tensor([[[[1.0, 0.0]]]]), 0.5),
        (torch.zeros_like(logits), 1 - (2 + 1e-6) / (3 + 1e-6)),
    ):
        total, bce, dice = criterion(logits, target)
        assert dice.item() == pytest.approx(expected_dice_loss)
        assert total.item() == pytest.approx(bce.item() + dice.item())
        total.backward(retain_graph=True)
    assert logits.grad is not None


def test_dice_loss_stays_finite_for_large_float16_batch():
    from utils.tumor_segmentation_loss import BCEWeightedDiceLoss

    criterion = BCEWeightedDiceLoss(1.2, 1.0, 1.0, 1.2, 1e-6, torch.device("cpu"))
    logits = torch.zeros((2, 1, 256, 256), dtype=torch.float16, requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, :128] = 1
    loss, _, dice = criterion(logits, target)
    assert torch.isfinite(loss) and torch.isfinite(dice)
    assert dice.item() == pytest.approx(0.5, abs=1e-4)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_signed_distance_map_handles_empty_full_and_clipped_masks():
    from utils.tumor_segmentation_loss import signed_distance_maps

    mask = torch.zeros(3, 1, 5, 5)
    mask[0, 0, 2, 2] = 1
    mask[2] = 1
    distances = signed_distance_maps(mask, max_distance=2.0)
    assert distances.shape == mask.shape
    assert distances[0, 0, 2, 2] < 0
    assert distances[0, 0, 2, 3] > 0
    assert distances[0, 0, 0, 0] == 1
    assert torch.count_nonzero(distances[1]) == 0
    assert torch.all(distances[2] < 0)
    assert torch.isfinite(distances).all()
    with pytest.raises(ValueError, match="max_distance"):
        signed_distance_maps(mask, max_distance=0)


def test_boundary_loss_has_correct_gradient_and_ignores_negative_slices():
    from utils.tumor_segmentation_loss import BoundaryLoss, signed_distance_maps

    target = torch.zeros(2, 1, 5, 5)
    target[0, 0, 2, 2] = 1
    maps = signed_distance_maps(target, max_distance=2)
    logits = torch.zeros_like(target, requires_grad=True)
    loss = BoundaryLoss()(logits, maps)
    loss.backward()
    assert logits.grad[0, 0, 2, 2] < 0
    assert logits.grad[0, 0, 0, 0] > 0
    assert torch.count_nonzero(logits.grad[1]) == 0
    zero_logits = torch.zeros_like(target, requires_grad=True)
    zero_loss = BoundaryLoss()(zero_logits, torch.zeros_like(target))
    zero_loss.backward()
    assert zero_loss.item() == 0
    assert torch.count_nonzero(zero_logits.grad) == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_run_epoch_boundary_ablation_and_logging(enabled, tmp_path, monkeypatch):
    import trainers.train_tumor_segmentation as trainer
    from data.tumor_segmentation_dataset import create_dataloader

    make_case(tmp_path, "A", [True, False])
    split = tmp_path / "split.txt"
    split.write_text("A\n", encoding="utf-8")
    loader = create_dataloader(tmp_path, split, batch_size=2, return_metadata=True)
    model = torch.nn.Conv2d(3, 1, 1)
    criterion = trainer.BCEWeightedDiceLoss(1.2, 1, 1, 1.2, 1e-6, torch.device("cpu"))
    if not enabled:

        def fail(*args, **kwargs):
            raise AssertionError("distance maps must not be computed when disabled")

        monkeypatch.setattr(trainer, "signed_distance_maps", fail)
    steps = []
    scores = trainer.run_epoch(
        model,
        loader,
        criterion,
        torch.device("cpu"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        scheduler=None,
        scaler=None,
        amp_dtype=None,
        threshold=0.5,
        progress_desc="test",
        step_logger=steps.append,
        boundary_enabled=enabled,
        boundary_weight=0.1,
        boundary_max_distance=8.0,
    )
    assert "boundary_loss" in scores and "boundary_loss" in steps[0]
    assert scores["loss"] == pytest.approx(
        scores["bce_loss"] + scores["dice_loss"] + 0.1 * scores["boundary_loss"]
    )
    if enabled:
        assert scores["boundary_loss"] != 0
    else:
        assert scores["boundary_loss"] == 0


def test_case_dice_uses_all_slices_per_case_and_excludes_empty_cases():
    from utils.tumor_segmentation_metrics import TumorSegmentationMetrics

    metrics = TumorSegmentationMetrics()
    pred = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 0.0]]], [[[1.0, 0.0]]]])
    target = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 0.0]]], [[[0.0, 0.0]]]])
    metrics.update(pred.bool(), target.bool(), ["A", "A", "B"], [5, 6, 8])
    result = metrics.compute(include_cases=True)
    assert result["case_dice_3d_tumor"] == pytest.approx(2 / 3)
    assert result["dice_micro"] == pytest.approx(0.5)
    assert result["negative_slice_specificity"] == 0


def test_balanced_sampler_uses_positive_hard_and_easy_groups(tmp_path):
    from data.tumor_segmentation_dataset import create_dataloader

    make_case(tmp_path, "A", [False, False, True, False, False, False, False])
    split = tmp_path / "train.txt"
    split.write_text("A\n", encoding="utf-8")
    loader = create_dataloader(
        tmp_path,
        split,
        training=True,
        balanced_sampling=True,
        batch_size=4,
        hard_negative_radius=1,
        positive_fraction=0.5,
        hard_negative_fraction=0.25,
        easy_negative_fraction=0.25,
        batches_per_epoch=1,
        return_metadata=True,
    )
    batch = next(iter(loader.batch_sampler))
    assert sum(index == 2 for index in batch) == 2
    assert sum(index in (1, 3) for index in batch) == 1
    assert sum(index in (0, 4, 5, 6) for index in batch) == 1
    actual = next(iter(loader))
    assert actual["image"].shape == (4, 3, 32, 32)
    assert actual["mask"].shape == (4, 1, 32, 32)
    assert actual["case_id"] == ["A"] * 4


def test_dataset_loader_balanced_disabled_and_limited_modes(tmp_path):
    from data.binary_classification_dataset import TumorCoveringBatchSampler
    from data.tumor_segmentation_dataset import create_dataloader

    make_case(tmp_path, "A", [False, False, True, False, False, False, False])
    split = tmp_path / "train.txt"
    split.write_text("A\n", encoding="utf-8")
    natural = create_dataloader(
        tmp_path, split, training=True, batch_size=2, return_metadata=True
    )
    assert len(natural.dataset) == 7
    assert sum(len(batch) for batch in natural.batch_sampler) == 7
    limited = create_dataloader(
        tmp_path, split, training=True, batch_size=2, batches_per_epoch=2
    )
    assert isinstance(limited.batch_sampler, TumorCoveringBatchSampler)
    assert 2 in [index for batch in limited.batch_sampler for index in batch]
    with pytest.raises(ValueError, match="training"):
        create_dataloader(tmp_path, split, balanced_sampling=True)


def test_train_saves_history_and_best_and_resumes(tmp_path, monkeypatch):
    import trainers.train_tumor_segmentation as trainer

    make_case(tmp_path, "TRAIN", [True, False])
    make_case(tmp_path, "VAL", [True, False])
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    train.write_text("TRAIN\n", encoding="utf-8")
    val.write_text("VAL\n", encoding="utf-8")
    monkeypatch.setattr(trainer, "TransUNet", lambda **kwargs: torch.nn.Conv2d(3, 1, 1))
    cfg = dict(
        trainer.CFG,
        ABLATION="multi_window",
        DATASET_ROOT=str(tmp_path),
        TRAIN_SPLIT=str(train),
        VAL_SPLIT=str(val),
        SAVE_ROOT=str(tmp_path),
        EXP_NAME="run",
        DEVICE="cpu",
        AMP_DTYPE="none",
        WANDB_ENABLED=False,
        BATCH_SIZE=2,
        NUM_WORKERS=0,
        BALANCED_TRAIN_SAMPLING=False,
        TRAIN_BATCHES_PER_EPOCH=None,
        EPOCHS=1,
        BACKBONE_PRETRAINED=False,
        CHANNEL_ATTENTION_ENABLED=True,
        BOUNDARY_LOSS_ENABLED=True,
    )
    trainer.main(cfg)
    directory = tmp_path / "run"
    history = json.loads((directory / "history_metrics.json").read_text())
    assert len(history) == 1 and "val_case_dice_3d_tumor" in history[0]
    assert "train_boundary_loss" in history[0] and "val_boundary_loss" in history[0]
    assert (directory / "best.pt").is_file() and (directory / "last.pt").is_file()
    checkpoint = torch.load(
        directory / "last.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["config"]["BOUNDARY_LOSS_ENABLED"] is True
    assert (
        checkpoint["model_config"]["cross_attention_enabled"]
        == cfg["CROSS_ATTENTION_ENABLED"]
    )
    assert checkpoint["model_config"]["channel_attention_enabled"] is True
    assert (
        checkpoint["model_config"]["channel_attention_reduction"]
        == cfg["CHANNEL_ATTENTION_REDUCTION"]
    )
    assert (
        checkpoint["model_config"]["cross_attention_dim"] == cfg["CROSS_ATTENTION_DIM"]
    )
    cfg.update(EPOCHS=2, RESUME_PATH=str(directory / "last.pt"))
    trainer.main(cfg)
    history = json.loads((directory / "history_metrics.json").read_text())
    assert [row["epoch"] for row in history] == [1, 2]


def test_trainer_uses_dataset_balanced_loader(tmp_path, monkeypatch):
    import trainers.train_tumor_segmentation as trainer

    make_case(tmp_path, "TRAIN", [False, False, True, False, False, False, False])
    make_case(tmp_path, "VAL", [True, False])
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    train.write_text("TRAIN\n", encoding="utf-8")
    val.write_text("VAL\n", encoding="utf-8")
    monkeypatch.setattr(trainer, "TransUNet", lambda **kwargs: torch.nn.Conv2d(3, 1, 1))
    cfg = dict(
        trainer.CFG,
        ABLATION="multi_window",
        DATASET_ROOT=str(tmp_path),
        TRAIN_SPLIT=str(train),
        VAL_SPLIT=str(val),
        SAVE_ROOT=str(tmp_path),
        EXP_NAME="balanced",
        DEVICE="cpu",
        AMP_DTYPE="none",
        WANDB_ENABLED=False,
        BATCH_SIZE=4,
        BATCH_VAL=2,
        NUM_WORKERS=0,
        BALANCED_TRAIN_SAMPLING=True,
        TRAIN_BATCHES_PER_EPOCH=1,
        HARD_NEGATIVE_RADIUS=1,
        POSITIVE_FRACTION=0.5,
        HARD_NEGATIVE_FRACTION=0.25,
        EASY_NEGATIVE_FRACTION=0.25,
        EPOCHS=1,
        BACKBONE_PRETRAINED=False,
    )
    trainer.main(cfg)
    history = json.loads((tmp_path / "balanced" / "history_metrics.json").read_text())
    assert len(history) == 1 and "val_case_dice_3d_tumor" in history[0]


@pytest.mark.parametrize("mode", ["baseline", "lung_roi"])
def test_single_window_training_presets(mode, tmp_path, monkeypatch):
    import shutil

    import trainers.train_tumor_segmentation as trainer

    for case in ("TRAIN", "VAL"):
        make_case(tmp_path, case, [True, False])
        shutil.copytree(
            tmp_path / "lung_cropped/processed" / case, tmp_path / "processed" / case
        )
    split_train, split_val = tmp_path / "train.txt", tmp_path / "val.txt"
    split_train.write_text("TRAIN\n", encoding="utf-8")
    split_val.write_text("VAL\n", encoding="utf-8")
    seen = []

    def build(**kwargs):
        seen.append(kwargs)
        return torch.nn.Conv2d(kwargs["in_channels"], 1, 1)

    monkeypatch.setattr(trainer, "SingleWindowTransUNet", build)
    cfg = dict(
        trainer.CFG,
        ABLATION=mode,
        DATASET_ROOT=str(tmp_path),
        TRAIN_SPLIT=str(split_train),
        VAL_SPLIT=str(split_val),
        SAVE_ROOT=str(tmp_path),
        EXP_NAME=mode,
        DEVICE="cpu",
        AMP_DTYPE="none",
        WANDB_ENABLED=False,
        BATCH_SIZE=2,
        BATCH_VAL=2,
        NUM_WORKERS=0,
        BALANCED_TRAIN_SAMPLING=False,
        TRAIN_BATCHES_PER_EPOCH=None,
        EPOCHS=1,
        BACKBONE_PRETRAINED=False,
    )
    trainer.main(cfg)
    checkpoint = torch.load(
        tmp_path / mode / "last.pt", map_location="cpu", weights_only=False
    )
    assert seen[0]["in_channels"] == 1
    assert "cross_attention_enabled" not in seen[0]
    assert checkpoint["config"]["ABLATION"] == mode
    assert checkpoint["model_config"]["in_channels"] == 1


def test_wandb_step_logging_registers_metrics_and_drops_undefined_values():
    from trainers.train_tumor_segmentation import _step_logger

    class FakeRun:
        def __init__(self):
            self.metrics, self.logs = [], []

        def define_metric(self, name, **kwargs):
            self.metrics.append((name, kwargs))

        def log(self, values):
            self.logs.append(values)

    run = FakeRun()
    log = _step_logger(run, "val")
    log({"loss": 0.5, "dice_tumor": float("nan")})
    log({"loss": 0.4})
    assert ("val_step_loss", {"step_metric": "val_step"}) in run.metrics
    assert run.logs == [
        {"val_step": 1, "val_step_loss": 0.5},
        {"val_step": 2, "val_step_loss": 0.4},
    ]
