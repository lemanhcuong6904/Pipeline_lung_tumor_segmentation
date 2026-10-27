import json
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def test_focal_loss_matches_definition_and_gamma_zero_is_weighted_bce():
    from utils.classification_loss import BinaryFocalLoss
    import torch.nn.functional as F

    logits = torch.tensor([[-2.0], [0.0], [3.0], [-3.0]], requires_grad=True)
    targets = torch.tensor([[0.0], [1.0], [1.0], [1.0]])
    weighted = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor([4.0]), reduction="none")
    wrong_probability = torch.where(targets == 1, torch.sigmoid(-logits), torch.sigmoid(logits))
    result = BinaryFocalLoss(pos_weight=4.0, gamma=2.0)(logits, targets)
    torch.testing.assert_close(result, (weighted * wrong_probability.square()).mean())
    torch.testing.assert_close(BinaryFocalLoss(pos_weight=4.0, gamma=0.0)(logits, targets), weighted.mean())
    result.backward()
    assert torch.isfinite(logits.grad).all()


def test_loss_config_validation():
    from utils.classification_loss import create_classification_loss

    for name, weight, gamma in (("unknown", 1, 2), ("focal", 1, -1), ("focal", 1, math.nan), ("weighted_bce", 0, 2)):
        with pytest.raises(ValueError):
            create_classification_loss(name, weight, gamma, torch.device("cpu"))


@pytest.mark.parametrize("trainer_name", ["train_binary_classification", "train_binary_classification_pytorch"])
def test_focal_loss_is_named_correctly_in_step_and_epoch_logs(trainer_name):
    import importlib
    from utils.classification_loss import create_classification_loss

    trainer = importlib.import_module(f"trainers.{trainer_name}")
    logits = torch.tensor([[0.0], [1.0]])
    labels = torch.tensor([[1.0], [0.0]])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=1)
    criterion = create_classification_loss("focal", 2.0, 2.0, torch.device("cpu"))
    logged = []
    result = trainer.run_epoch(torch.nn.Identity(), loader, criterion, torch.device("cpu"),
                               optimizer=None, scaler=None, max_batches=None, step_logger=logged.append)
    assert result["focal_loss"] == pytest.approx(criterion(logits, labels).item())
    assert "weighted_bce_loss" not in result
    assert all("focal_loss" in row and "weighted_bce_loss" not in row for row in logged)


@pytest.mark.parametrize("trainer_name,model_name", [
    ("train_binary_classification", "ResNet50"),
    ("train_binary_classification_pytorch", "ResNet50PyTorch"),
])
def test_focal_config_is_used_and_saved_in_epoch_history(tmp_path, monkeypatch, trainer_name, model_name):
    import importlib

    trainer = importlib.import_module(f"trainers.{trainer_name}")
    loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.tensor([[1.0], [0.0]])), batch_size=2)
    monkeypatch.setattr(trainer, "create_dataloader", lambda *args, **kwargs: loader)
    monkeypatch.setattr(trainer, model_name, lambda **kwargs: torch.nn.Linear(1, 1))
    cfg = trainer.CFG | {"SAVE_ROOT": str(tmp_path), "EXP_NAME": "focal_test", "DEVICE": "cpu",
                         "AMP_DTYPE": "none", "WANDB_ENABLED": False, "EPOCHS": 1,
                         "TRAIN_BATCHES_PER_EPOCH": None, "RESUME_PATH": "", "LOSS": "focal",
                         "FOCAL_GAMMA": 2.0}
    trainer.main(cfg)
    history = json.loads((tmp_path / "focal_test" / "metrics_history.json").read_text())
    assert "train_focal_loss" in history[0] and "val_focal_loss" in history[0]
    assert "train_weighted_bce_loss" not in history[0]


def test_classification_metrics_count_samples_and_handle_empty_positive_class():
    from utils.classification_metrics import BinaryClassificationMetrics

    metrics = BinaryClassificationMetrics()
    metrics.update(torch.tensor([[2.0], [2.0], [-2.0]]), torch.tensor([[1.0], [0.0], [1.0]]))
    metrics.update(torch.tensor([[-2.0]]), torch.tensor([[0.0]]))
    assert metrics.compute() == pytest.approx({"acc": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5})
    empty = BinaryClassificationMetrics()
    empty.update(torch.tensor([[-1.0]]), torch.tensor([[0.0]]))
    assert empty.compute() == {"acc": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_run_epoch_weights_loss_and_aggregates_over_samples():
    from trainers.train_binary_classification import run_epoch

    logits = torch.tensor([[0.0], [0.0], [-2.0]])
    labels = torch.tensor([[1.0], [0.0], [1.0]])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]))
    logged = []
    result = run_epoch(torch.nn.Identity(), loader, criterion, torch.device("cpu"),
                       optimizer=None, scaler=None, max_batches=None, step_logger=logged.append)
    assert result["weighted_bce_loss"] == pytest.approx(criterion(logits, labels).item())
    assert result == pytest.approx({"weighted_bce_loss": criterion(logits, labels).item(),
                                    "acc": 1 / 3, "precision": 0.5, "recall": 0.5, "f1": 0.5})
    assert len(logged) == 2 and set(logged[0]) == set(result)


def test_main_saves_best_by_val_f1_and_stops_after_five_stale_epochs(tmp_path, monkeypatch):
    import trainers.train_binary_classification as trainer

    loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.ones(2, 1)), batch_size=1)
    calls = []
    def make_loader(*args, **kwargs):
        calls.append(kwargs)
        return loader
    monkeypatch.setattr(trainer, "create_dataloader", make_loader)
    def make_model():
        return torch.nn.Linear(1, 1)
    monkeypatch.setattr(trainer, "ResNet50", make_model)
    val_f1 = iter([0.5, 0.7, 0.6, 0.7, 0.2, 0.1, 0.0])
    def fake_epoch(*args, **kwargs):
        f1 = 0.9 if kwargs["optimizer"] is not None else next(val_f1)
        return {"weighted_bce_loss": 1.0, "acc": 0.5, "precision": 0.5, "recall": 0.5, "f1": f1}
    monkeypatch.setattr(trainer, "run_epoch", fake_epoch)
    cfg = trainer.CFG | {"SAVE_ROOT": str(tmp_path), "EXP_NAME": "test", "DEVICE": "cpu",
                         "AMP_DTYPE": "none", "WANDB_ENABLED": False, "EPOCHS": 20,
                         "TRAIN_BATCHES_PER_EPOCH": None, "RESUME_PATH": "", "EARLY_STOPPING_PATIENCE": 5}
    trainer.main(cfg)
    history = json.loads((tmp_path / "test" / "metrics_history.json").read_text())
    assert len(history) == 7 and history[-1]["val_f1"] == 0.0
    assert torch.load(tmp_path / "test" / "best.pt", weights_only=False)["epoch"] == 2
    assert torch.load(tmp_path / "test" / "last.pt", weights_only=False)["epoch"] == 7
    assert calls[0]["training"] is True and calls[1]["training"] is False
    assert calls[0]["labels_csv"] == cfg["TRAIN_LABELS_CSV"]
    assert calls[1]["labels_csv"] == cfg["VAL_LABELS_CSV"]
    assert calls[0]["balanced_sampling"] == cfg["BALANCED_TRAIN_SAMPLING"]
    assert calls[0]["hard_negative_radius"] == cfg["HARD_NEGATIVE_RADIUS"]
    assert not calls[1].get("balanced_sampling", False)
