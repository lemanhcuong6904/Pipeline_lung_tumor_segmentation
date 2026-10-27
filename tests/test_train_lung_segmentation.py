import pytest
import subprocess
import sys
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader, TensorDataset

from trainers.train_lung_segmentation import amp_dtype_from_config, run_epoch
from utils.metrics import dice_score
from utils.training import scheduler_step_counts
from utils.training import append_metrics_history, load_metrics_history


def test_amp_dtype_config_supports_bf16_without_a_grad_scaler():
    """BF16 needs autocast but must not use FP16 gradient scaling."""
    assert amp_dtype_from_config("bf16") is torch.bfloat16
    assert amp_dtype_from_config("fp16") is torch.float16
    assert amp_dtype_from_config("none") is None


def test_dice_score_penalizes_false_positive_pixels():
    """Turning a background pixel positive must lower segmentation Dice."""
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])

    assert dice_score(logits, targets) == pytest.approx(2 / 3)


def test_scheduler_step_counts_use_actual_limited_train_batches():
    """Using loader length instead of the cap produces an incorrectly long warm-up."""
    total_steps, warmup_steps = scheduler_step_counts(
        loader_batches=10,
        train_batches_per_epoch=3,
        epochs=4,
        warmup_fraction=0.25,
    )

    assert (total_steps, warmup_steps) == (12, 3)


def test_run_epoch_limits_train_batches_and_logs_only_bce_and_dice():
    """Ignoring the batch cap breaks the configured optimizer-step schedule."""
    loader = DataLoader(
        TensorDataset(torch.zeros(3, 1, 2, 2), torch.ones(3, 1, 2, 2)), batch_size=1
    )
    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    logged_steps: list[dict[str, float]] = []

    metrics = run_epoch(
        model,
        loader,
        torch.nn.BCEWithLogitsLoss(),
        torch.device("cpu"),
        optimizer=optimizer,
        scaler=None,
        max_batches=2,
        step_logger=logged_steps.append,
    )

    assert set(metrics) == {"bce_loss", "dice"}
    assert len(logged_steps) == 2
    assert all(set(values) == {"bce_loss", "dice"} for values in logged_steps)


def test_run_epoch_passes_epoch_label_to_tqdm(monkeypatch):
    """Without the epoch label, users cannot identify progress across epochs."""
    captured: dict[str, object] = {}

    class Progress:
        def __init__(self, iterable):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **_values):
            pass

    def fake_tqdm(iterable, **kwargs):
        captured.update(kwargs)
        return Progress(iterable)

    monkeypatch.setattr("trainers.train_lung_segmentation.tqdm", fake_tqdm)
    loader = DataLoader(TensorDataset(torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2)))
    run_epoch(
        torch.nn.Conv2d(1, 1, kernel_size=1),
        loader,
        torch.nn.BCEWithLogitsLoss(),
        torch.device("cpu"),
        optimizer=None,
        scaler=None,
        max_batches=None,
        step_logger=None,
        progress_desc="Epoch 3/10 | Val",
    )

    assert captured["desc"] == "Epoch 3/10 | Val"


def test_run_epoch_sets_tqdm_total_to_the_training_batch_limit(monkeypatch):
    """A full-loader total falsely suggests that limited batches will be processed."""
    captured: dict[str, object] = {}

    class Progress:
        def __init__(self, iterable):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **_values):
            pass

    def fake_tqdm(iterable, **kwargs):
        captured.update(kwargs)
        return Progress(iterable)

    monkeypatch.setattr("trainers.train_lung_segmentation.tqdm", fake_tqdm)
    loader = DataLoader(TensorDataset(torch.zeros(3, 1, 2, 2), torch.ones(3, 1, 2, 2)))
    run_epoch(
        torch.nn.Conv2d(1, 1, kernel_size=1), loader, torch.nn.BCEWithLogitsLoss(),
        torch.device("cpu"), optimizer=None, scaler=None, max_batches=2, step_logger=None,
    )

    assert captured["total"] == 2


def test_metrics_history_appends_new_epochs_without_duplicates(tmp_path: Path):
    """Resuming must keep prior curve points and replace, not duplicate, an epoch."""
    path = tmp_path / "metrics_history.json"
    first = {"epoch": 1, "train_bce_loss": 0.4, "train_dice": 0.5, "val_bce_loss": 0.3, "val_dice": 0.6}
    updated = first | {"val_dice": 0.7}
    second = {"epoch": 2, "train_bce_loss": 0.2, "train_dice": 0.8, "val_bce_loss": 0.1, "val_dice": 0.9}

    append_metrics_history(path, [], first)
    history = append_metrics_history(path, load_metrics_history(path), updated)
    history = append_metrics_history(path, history, second)

    assert history == [updated, second]
    assert json.loads(path.read_text(encoding="utf-8")) == [updated, second]


def test_trainer_file_imports_when_run_directly_from_the_project_root():
    """Direct execution must resolve sibling packages such as ``data``."""
    script = Path(__file__).parents[1] / "trainers" / "train_lung_segmentation.py"
    command = (
        "import runpy, sys; "
        f"root = {str(script.parent.parent)!r}; "
        f"sys.path = [{str(script.parent)!r}] + [path for path in sys.path if path not in ('', root)]; "
        f"runpy.run_path({str(script)!r}, run_name='import_check')"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
