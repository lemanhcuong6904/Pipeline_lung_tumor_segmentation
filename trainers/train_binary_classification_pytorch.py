"""Config-only TorchVision ResNet-50 trainer for slice-level tumor classification."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from data.binary_classification_dataset import create_dataloader
from models.resnet50_pytorch import ResNet50PyTorch
from trainers.train_lung_segmentation import (
    amp_dtype_from_config,
    _step_logger,
    _wandb_run,
)
from utils.classification_metrics import BinaryClassificationMetrics
from utils.classification_loss import BinaryFocalLoss, create_classification_loss
from utils.training import (
    append_metrics_history,
    load_checkpoint,
    load_metrics_history,
    save_checkpoint,
    scheduler_step_counts,
    seed_everything,
    warmup_cosine_step_lambda,
)

DATASET = "nsclc_radiogenomics"  # "nsclc_radiomics" or "nsclc_radiogenomics".
if DATASET not in {"nsclc_radiomics", "nsclc_radiogenomics"}:
    raise ValueError(f"unsupported DATASET: {DATASET}")
DATASET_ROOT = ROOT_DIR / "data" / DATASET

CFG: dict[str, Any] = {
    "EXP_NAME": "nsclc_radiogenomics_resnet50_pytorch_binary_classification",
    "PRETRAINED": True,
    "DATASET_ROOT": str(DATASET_ROOT),
    "TRAIN_SPLIT": str(DATASET_ROOT / "config" / "train.txt"),
    "VAL_SPLIT": str(DATASET_ROOT / "config" / "test.txt"),
    "TRAIN_LABELS_CSV": str(
        DATASET_ROOT / "lung_cropped" / "config" / "train_labels.csv"
    ),
    "VAL_LABELS_CSV": str(
        DATASET_ROOT / "lung_cropped" / "config" / "test_labels.csv"
    ),
    "SAVE_ROOT": str((ROOT_DIR / "experiments" if DATASET == "nsclc_radiomics" else ROOT_DIR / "experiments" / DATASET)),
    "BATCH_SIZE": 32,
    "NUM_WORKERS": 0,
    "TRAIN_BATCHES_PER_EPOCH": None,
    "BALANCED_TRAIN_SAMPLING": True,
    "HARD_NEGATIVE_RADIUS": 6,
    "POSITIVE_FRACTION": 0.35,
    "HARD_NEGATIVE_FRACTION": 0.20,
    "EASY_NEGATIVE_FRACTION": 0.45,
    "SAMPLER_SEED": 42,
    "EPOCHS": 30,
    "LR": 5e-4,
    "LR_MIN": 1e-6,
    "WEIGHT_DECAY": 1e-4,
    "WARMUP_FRACTION": 0.1,
    "AMP_DTYPE": "bf16",  # bf16, fp16 or none
    "THRESHOLD": 0.5,
    "LOSS": "focal",  # weighted_bce or focal
    "FOCAL_GAMMA": 2.0,
    "POS_WEIGHT": 2.0,
    "EARLY_STOPPING_PATIENCE": 7,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "WANDB_ENABLED": False,
    "WANDB_PROJECT": "lung-tumor-classification",
    "RESUME_PATH": r"experiments\resnet50_pytorch_binary_classification\best.pt",
    "RESUME_WEIGHTS_ONLY": True,
    "SEED": 42,
}


def run_epoch(
    model: torch.nn.Module,
    loader: Any,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    max_batches: int | None,
    step_logger: Callable[[dict[str, float]], None] | None,
    amp_dtype: torch.dtype | None = None,
    scheduler: LambdaLR | None = None,
    threshold: float = 0.5,
    progress_desc: str | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    use_amp = amp_dtype is not None and device.type == "cuda"
    autocast_args: dict[str, Any] = {"device_type": device.type, "enabled": use_amp}
    if use_amp:
        autocast_args["dtype"] = amp_dtype
    metrics = BinaryClassificationMetrics(threshold)
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(
        loader,
        leave=False,
        dynamic_ncols=True,
        total=len(loader) if max_batches is None else min(len(loader), max_batches),
        desc=progress_desc or ("Train" if training else "Val"),
    )
    for batch_index, (images, labels) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(**autocast_args):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            if scaler is not None:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= scale_before
            else:
                loss.backward()
                optimizer.step()
                optimizer_stepped = True
            if scheduler is not None and optimizer_stepped:
                scheduler.step()
        batch_metrics = BinaryClassificationMetrics(threshold)
        batch_metrics.update(logits, labels)
        metrics.update(logits, labels)
        loss_value = float(loss.detach())
        total_loss += loss_value * images.shape[0]
        total_samples += images.shape[0]
        loss_name = "focal_loss" if isinstance(criterion, BinaryFocalLoss) else "weighted_bce_loss"
        values = {loss_name: loss_value, **batch_metrics.compute()}
        progress.set_postfix(**{key: f"{value:.4f}" for key, value in values.items()})
        if step_logger:
            step_logger(values)
    if not total_samples:
        raise ValueError("loader yielded no batches")
    return {loss_name: total_loss / total_samples, **metrics.compute()}


def main(config: dict[str, Any] | None = None) -> None:
    cfg = dict(CFG if config is None else config)
    seed_everything(int(cfg["SEED"]))
    device = torch.device(cfg["DEVICE"])
    directory = Path(cfg["SAVE_ROOT"]) / cfg["EXP_NAME"]
    directory.mkdir(parents=True, exist_ok=True)
    history_path = directory / "metrics_history.json"
    train_loader = create_dataloader(
        cfg["DATASET_ROOT"],
        cfg["TRAIN_SPLIT"],
        batch_size=int(cfg["BATCH_SIZE"]),
        num_workers=int(cfg["NUM_WORKERS"]),
        shuffle=True,
        training=True,
        labels_csv=cfg["TRAIN_LABELS_CSV"],
        balanced_sampling=bool(cfg["BALANCED_TRAIN_SAMPLING"]),
        batches_per_epoch=cfg["TRAIN_BATCHES_PER_EPOCH"],
        sampler_seed=int(cfg["SAMPLER_SEED"]),
        hard_negative_radius=int(cfg["HARD_NEGATIVE_RADIUS"]),
        positive_fraction=float(cfg["POSITIVE_FRACTION"]),
        hard_negative_fraction=float(cfg["HARD_NEGATIVE_FRACTION"]),
        easy_negative_fraction=float(cfg["EASY_NEGATIVE_FRACTION"]),
    )
    val_loader = create_dataloader(
        cfg["DATASET_ROOT"],
        cfg["VAL_SPLIT"],
        batch_size=int(cfg["BATCH_SIZE"]),
        num_workers=int(cfg["NUM_WORKERS"]),
        shuffle=False,
        training=False,
        labels_csv=cfg["VAL_LABELS_CSV"],
    )
    pretrained = bool(cfg["PRETRAINED"]) and not str(cfg["RESUME_PATH"]).strip()
    model = ResNet50PyTorch(pretrained=pretrained).to(device)
    criterion = create_classification_loss(str(cfg["LOSS"]), float(cfg["POS_WEIGHT"]),
                                           float(cfg["FOCAL_GAMMA"]), device)
    optimizer = AdamW(
        model.parameters(), lr=float(cfg["LR"]), weight_decay=float(cfg["WEIGHT_DECAY"])
    )
    total, warmup = scheduler_step_counts(
        loader_batches=len(train_loader),
        train_batches_per_epoch=cfg["TRAIN_BATCHES_PER_EPOCH"],
        epochs=int(cfg["EPOCHS"]),
        warmup_fraction=float(cfg["WARMUP_FRACTION"]),
    )
    scheduler = LambdaLR(
        optimizer,
        warmup_cosine_step_lambda(
            total, warmup, float(cfg["LR"]), float(cfg["LR_MIN"])
        ),
    )
    amp_dtype = amp_dtype_from_config(str(cfg["AMP_DTYPE"]))
    if amp_dtype is not None and device.type != "cuda":
        print("[WARN] AMP_DTYPE is ignored because training is not using CUDA.")
        amp_dtype = None
    if amp_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "AMP_DTYPE='bf16' requires a CUDA GPU with BF16 support. Use 'fp16' or 'none'."
        )
    scaler = (
        torch.amp.GradScaler("cuda")
        if amp_dtype is torch.float16 and device.type == "cuda"
        else None
    )
    start, best_f1, stale = 1, float("-inf"), 0
    history: list[dict[str, float | int]] = []
    resume = str(cfg["RESUME_PATH"]).strip()
    if resume:
        state = load_checkpoint(resume, device)
        model.load_state_dict(state["model_state"])
        if not cfg["RESUME_WEIGHTS_ONLY"]:
            optimizer.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
            if scaler and state.get("scaler_state"):
                scaler.load_state_dict(state["scaler_state"])
            start = int(state["epoch"]) + 1
            best_f1 = float(state.get("best_val_f1", float("-inf")))
            stale = int(state.get("stale_epochs", 0))
            history = load_metrics_history(history_path)
            for row in state.get("metrics_history", []):
                history = append_metrics_history(history_path, history, row)
    run = _wandb_run(cfg)
    train_log, val_log = _step_logger(run, "train"), _step_logger(run, "val")
    try:
        for epoch in range(start, int(cfg["EPOCHS"]) + 1):
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch - 1)
            train = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                max_batches=cfg["TRAIN_BATCHES_PER_EPOCH"],
                step_logger=train_log,
                amp_dtype=amp_dtype,
                scheduler=scheduler,
                threshold=float(cfg["THRESHOLD"]),
                progress_desc=f"Epoch {epoch}/{cfg['EPOCHS']} | Train",
            )
            val = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                scaler=None,
                max_batches=None,
                step_logger=val_log,
                amp_dtype=amp_dtype,
                threshold=float(cfg["THRESHOLD"]),
                progress_desc=f"Epoch {epoch}/{cfg['EPOCHS']} | Val",
            )
            values = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train.items()},
                **{f"val_{k}": v for k, v in val.items()},
            }
            print(values)
            if run:
                run.log(values)
            history = append_metrics_history(history_path, history, values)
            improved = val["f1"] > best_f1
            best_f1 = max(best_f1, val["f1"])
            stale = 0 if improved else stale + 1
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler else None,
                "best_val_f1": best_f1,
                "stale_epochs": stale,
                "metrics_history": history,
                "config": cfg,
            }
            save_checkpoint(directory / "last.pt", state)
            if improved:
                save_checkpoint(directory / "best.pt", state)
            if stale >= int(cfg["EARLY_STOPPING_PATIENCE"]):
                print("Early stopping: val_f1 did not improve.")
                break
    finally:
        if run:
            run.finish()


if __name__ == "__main__":
    main()
