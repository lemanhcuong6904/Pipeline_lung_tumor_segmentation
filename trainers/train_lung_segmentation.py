"""Config-only UNet trainer for lung segmentation."""

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

from data.lung_segmentation_dataset import create_dataloader
from models.unet import UNet
from utils.metrics import dice_score
from utils.training import (
    append_metrics_history,
    load_checkpoint,
    load_metrics_history,
    save_checkpoint,
    scheduler_step_counts,
    seed_everything,
    warmup_cosine_step_lambda,
)

DATASET = "nsclc_radiomics"  # "nsclc_radiomics" or "nsclc_radiogenomics".
if DATASET not in {"nsclc_radiomics", "nsclc_radiogenomics"}:
    raise ValueError(f"unsupported DATASET: {DATASET}")
DATASET_ROOT = ROOT_DIR / "data" / DATASET

CFG: dict[str, Any] = {
    "EXP_NAME": "unet_lung_segmentation",
    "DATASET_ROOT": str(DATASET_ROOT),
    "TRAIN_SPLIT": str(DATASET_ROOT / "config" / "train.txt"),
    "VAL_SPLIT": str(DATASET_ROOT / "config" / "val.txt"),
    "SAVE_ROOT": str((ROOT_DIR / "experiments" if DATASET == "nsclc_radiomics" else ROOT_DIR / "experiments" / DATASET)),
    "BATCH_SIZE": 16,
    "NUM_WORKERS": 0,
    "TRAIN_BATCHES_PER_EPOCH": 500,
    "EPOCHS": 20,
    "LR": 5e-4,
    "LR_MIN": 1e-6,
    "WEIGHT_DECAY": 1e-4,
    "WARMUP_FRACTION": 0.1,
    # Chọn "bf16" cho AMP ổn định hơn FP16 trên GPU hỗ trợ BF16.
    # Các giá trị hợp lệ: "bf16", "fp16", "none".
    "AMP_DTYPE": "bf16",
    "THRESHOLD": 0.5,
    "EARLY_STOPPING_PATIENCE": 5,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "WANDB_ENABLED": True,
    "WANDB_PROJECT": "lung-segmentation",
    "RESUME_PATH": r"experiments\unet_lung_segmentation\last.pt" if DATASET == "nsclc_radiomics" else "",
    "RESUME_WEIGHTS_ONLY": False,
    "SEED": 42,
}


def amp_dtype_from_config(mode: str) -> torch.dtype | None:
    """Convert the configured AMP mode to a PyTorch autocast dtype."""
    normalized_mode = mode.strip().lower()
    dtypes: dict[str, torch.dtype | None] = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "none": None,
    }
    if normalized_mode not in dtypes:
        choices = ", ".join(dtypes)
        raise ValueError(f"AMP_DTYPE must be one of: {choices}; got {mode!r}")
    return dtypes[normalized_mode]


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
    total_loss = total_tp = total_pred = total_target = 0.0
    total_images = 0
    progress_total = len(loader) if max_batches is None else min(len(loader), max_batches)
    progress = tqdm(
        loader,
        leave=False,
        dynamic_ncols=True,
        total=progress_total,
        desc=progress_desc or ("Train" if training else "Val"),
    )
    for batch_index, (images, masks) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, masks = images.to(device, non_blocking=True), masks.to(
            device, non_blocking=True
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast_args: dict[str, Any] = {
            "device_type": device.type,
            "enabled": use_amp,
        }
        if use_amp:
            autocast_args["dtype"] = amp_dtype
        with torch.set_grad_enabled(training), torch.autocast(**autocast_args):
            logits = model(images)
            loss = criterion(logits, masks)
        if training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler:
                scheduler.step()
        predictions, targets = torch.sigmoid(logits.detach()) >= threshold, masks >= 0.5
        total_tp += float((predictions & targets).sum())
        total_pred += float(predictions.sum())
        total_target += float(targets.sum())
        total_loss += float(loss.detach()) * images.shape[0]
        total_images += images.shape[0]
        values = {
            "bce_loss": float(loss.detach()),
            "dice": dice_score(logits.detach(), masks, threshold=threshold),
        }
        progress.set_postfix(
            bce_loss=f"{values['bce_loss']:.4f}", dice=f"{values['dice']:.4f}"
        )
        if step_logger:
            step_logger(values)
    if not total_images:
        raise ValueError("loader yielded no batches")
    denominator = total_pred + total_target
    return {
        "bce_loss": total_loss / total_images,
        "dice": 1.0 if not denominator else 2 * total_tp / denominator,
    }


def _wandb_run(cfg: dict[str, Any]) -> Any | None:
    if not cfg["WANDB_ENABLED"]:
        return None
    try:
        import wandb

        return wandb.init(
            project=cfg["WANDB_PROJECT"], name=cfg["EXP_NAME"], config=cfg
        )
    except Exception as error:
        print(f"[WARN] W&B disabled: {error}")
        return None


def _step_logger(
    run: Any | None, split: str
) -> Callable[[dict[str, float]], None] | None:
    if run is None:
        return None
    step = 0

    def log(values: dict[str, float]) -> None:
        nonlocal step
        step += 1
        run.log(
            {
                f"{split}_step": step,
                **{f"{split}_step_{key}": value for key, value in values.items()},
            }
        )

    return log


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
    )
    val_loader = create_dataloader(
        cfg["DATASET_ROOT"],
        cfg["VAL_SPLIT"],
        batch_size=int(cfg["BATCH_SIZE"]),
        num_workers=int(cfg["NUM_WORKERS"]),
        shuffle=False,
    )
    model = UNet().to(device)
    optimizer = AdamW(
        model.parameters(), lr=float(cfg["LR"]), weight_decay=float(cfg["WEIGHT_DECAY"])
    )
    criterion = torch.nn.BCEWithLogitsLoss()
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
            "AMP_DTYPE='bf16' requires a CUDA GPU with BF16 support. "
            "Use AMP_DTYPE='fp16' or 'none' instead."
        )
    # GradScaler prevents FP16 underflow, but is unnecessary for BF16.
    scaler = (
        torch.amp.GradScaler("cuda")
        if amp_dtype is torch.float16 and device.type == "cuda"
        else None
    )
    start, best_dice, stale = 1, float("-inf"), 0
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
            start, best_dice, stale = (
                int(state["epoch"]) + 1,
                float(state.get("best_val_dice", float("-inf"))),
                int(state.get("stale_epochs", 0)),
            )
            history = load_metrics_history(history_path)
            for row in state.get("metrics_history", []):
                history = append_metrics_history(history_path, history, row)
    run = _wandb_run(cfg)
    train_log, val_log = _step_logger(run, "train"), _step_logger(run, "val")
    try:
        for epoch in range(start, int(cfg["EPOCHS"]) + 1):
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
                threshold=float(cfg["THRESHOLD"]),
                progress_desc=f"Epoch {epoch}/{cfg['EPOCHS']} | Val",
            )
            values = {
                "epoch": epoch,
                "train_bce_loss": train["bce_loss"],
                "train_dice": train["dice"],
                "val_bce_loss": val["bce_loss"],
                "val_dice": val["dice"],
            }
            print(values)
            if run:
                run.log(values)
            history = append_metrics_history(history_path, history, values)
            improved = val["dice"] > best_dice
            best_dice = max(best_dice, val["dice"])
            stale = 0 if improved else stale + 1
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler else None,
                "best_val_dice": best_dice,
                "stale_epochs": stale,
                "metrics_history": history,
                "config": cfg,
            }
            save_checkpoint(directory / "last.pt", state)
            if improved:
                save_checkpoint(directory / "best.pt", state)
            if stale >= int(cfg["EARLY_STOPPING_PATIENCE"]):
                print("Early stopping: val_dice did not improve.")
                break
    finally:
        if run:
            run.finish()


if __name__ == "__main__":
    main()
