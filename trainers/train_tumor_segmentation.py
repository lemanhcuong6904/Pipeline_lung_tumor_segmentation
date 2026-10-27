"""Config-only training of the three-window multi-encoder tumor TransUNet."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.tumor_segmentation_dataset import ABLATION_MODES, create_dataloader
from models.transunet import SingleWindowTransUNet, TransUNet
from trainers.train_lung_segmentation import amp_dtype_from_config
from utils.training import (
    append_metrics_history,
    load_checkpoint,
    load_metrics_history,
    save_checkpoint,
    scheduler_step_counts,
    seed_everything,
    warmup_cosine_step_lambda,
)
from utils.tumor_segmentation_loss import (
    BCEWeightedDiceLoss,
    BoundaryLoss,
    signed_distance_maps,
)
from utils.tumor_segmentation_metrics import TumorSegmentationMetrics

ABLATION_EXPERIMENT_NAMES = {
    "baseline": "transunet_baseline_wide",
    "lung_roi": "transunet_lung_roi_wide",
    "multi_window": "transunet_multi_window_tumor_channel_attention_boundary_loss",
}

DATASET = "nsclc_radiogenomics"  # "nsclc_radiomics" or "nsclc_radiogenomics".
if DATASET not in {"nsclc_radiomics", "nsclc_radiogenomics"}:
    raise ValueError(f"unsupported DATASET: {DATASET}")
DATASET_ROOT = ROOT_DIR / "data" / DATASET

CFG: dict[str, Any] = {
    "ABLATION": "baseline",  # baseline | lung_roi | multi_window
    "EXP_NAME": "nsclc_radiogenomics_transunet_baseline_wide",  # empty: select the corresponding name above
    "DATASET_ROOT": str(DATASET_ROOT),
    "TRAIN_SPLIT": str(DATASET_ROOT / "config" / "train.txt"),
    "VAL_SPLIT": str(DATASET_ROOT / "config" / "test.txt"),
    "SAVE_ROOT": str((ROOT_DIR / "experiments" if DATASET == "nsclc_radiomics" else ROOT_DIR / "experiments" / DATASET)),
    "IMAGE_SIZE": 256,
    "PATCH_DIM": 16,
    "BASE_CHANNELS": 64,
    "EMBED_DIM": 512,
    "TRANSFORMER_DEPTH": 8,
    "TRANSFORMER_HEADS": 4,
    "MLP_DIM": 512,
    "MLP_RATIO": 0.5,
    "DROPOUT": 0.0,
    # Cross/channel attention only apply to the multi_window architecture.
    "CROSS_ATTENTION_ENABLED": False,
    "CROSS_ATTENTION_DIM": 128,
    "CROSS_ATTENTION_HEADS": 4,
    "CROSS_ATTENTION_DROPOUT": 0.0,
    "CHANNEL_ATTENTION_ENABLED": True,
    "CHANNEL_ATTENTION_REDUCTION": 16,
    "BACKBONE_PRETRAINED": True,
    "BATCH_SIZE": 8,
    "BATCH_VAL": 8,
    "NUM_WORKERS": 0,
    "PIN_MEMORY": True,
    "BALANCED_TRAIN_SAMPLING": True,
    "HARD_NEGATIVE_RADIUS": 4,
    "POSITIVE_FRACTION": 0.35,
    "HARD_NEGATIVE_FRACTION": 0.2,
    "EASY_NEGATIVE_FRACTION": 0.45,
    "TRAIN_BATCHES_PER_EPOCH": 1200,
    "SAMPLER_SEED": 42,
    "EPOCHS": 30,
    "LR": 1e-4,
    "LR_MIN": 1e-6,
    "WEIGHT_DECAY": 1e-4,
    "WARMUP_FRACTION": 0.05,
    "AMP_DTYPE": "bf16",
    "THRESHOLD": 0.5,
    "BCE_POS_WEIGHT": 1.2,
    "DICE_LOSS_WEIGHT": 1.0,
    "DICE_BACKGROUND_WEIGHT": 1.0,
    "DICE_FOREGROUND_WEIGHT": 1.2,
    "DICE_SMOOTH": 1e-6,
    "BOUNDARY_LOSS_ENABLED": False,
    "BOUNDARY_LOSS_WEIGHT": 0.1,
    "BOUNDARY_DISTANCE_CLIP": 32.0,  # pixels after augmentation
    "EARLY_STOPPING_PATIENCE": 10,
    "CHECKPOINT_METRIC": "val_case_dice_3d_tumor",
    "EARLY_STOPPING_METRIC": "val_case_dice_3d_tumor",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "WANDB_ENABLED": True,
    "WANDB_PROJECT": "tumor-segmentation",
    "RESUME_PATH": r"",
    "RESUME_WEIGHTS_ONLY": False,
    "SEED": 42,
}


def _finite(values: dict[str, float | int]) -> dict[str, float | int]:
    return {k: v for k, v in values.items() if math.isfinite(float(v))}


def _wandb_run(cfg: dict[str, Any]):
    if not cfg["WANDB_ENABLED"]:
        return None
    try:
        import wandb

        return wandb.init(
            project=cfg["WANDB_PROJECT"], name=cfg["EXP_NAME"], config=cfg
        )
    except Exception as error:  # noqa: BLE001 - match the existing W&B fallback
        print(f"[WARN] W&B disabled: {error}")
        return None


def _step_logger(run: Any, split: str) -> Callable[[dict[str, float]], None] | None:
    if run is None:
        return None
    step = 0
    step_name = f"{split}_step"
    run.define_metric(step_name)
    for name in (
        "loss",
        "bce_loss",
        "dice_loss",
        "boundary_loss",
        "dice",
        "dice_micro",
        "dice_tumor",
        "negative_slice_specificity",
    ):
        run.define_metric(f"{split}_step_{name}", step_metric=step_name)

    def log(values: dict[str, float]) -> None:
        nonlocal step
        step += 1
        run.log(
            {
                step_name: step,
                **{f"{split}_step_{k}": v for k, v in _finite(values).items()},
            }
        )

    return log


def _improved(key: str, value: float, best: float | None) -> bool:
    lower_is_better = "loss" in key or key in {"val_fp_2d", "val_fn_2d"}
    return math.isfinite(value) and (
        best is None or (value < best if lower_is_better else value > best)
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: BCEWeightedDiceLoss,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scheduler: LambdaLR | None,
    scaler: torch.amp.GradScaler | None,
    amp_dtype: torch.dtype | None,
    threshold: float,
    progress_desc: str,
    step_logger: Callable[[dict[str, float]], None] | None = None,
    boundary_enabled: bool = False,
    boundary_weight: float = 0.1,
    boundary_max_distance: float = 32.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics = TumorSegmentationMetrics()
    sums = {"loss": 0.0, "bce_loss": 0.0, "dice_loss": 0.0, "boundary_loss": 0.0}
    boundary_criterion = BoundaryLoss()
    samples = 0
    iterator = tqdm(loader, desc=progress_desc, leave=False, dynamic_ncols=True)
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        distance_maps = (
            signed_distance_maps(batch["mask"], boundary_max_distance).to(
                device, non_blocking=True
            )
            if boundary_enabled and boundary_weight > 0
            else None
        )
        masks = batch["mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=device.type,
                enabled=amp_dtype is not None and device.type == "cuda",
                dtype=amp_dtype,
            ),
        ):
            logits = model(images)
            loss, bce, dice_loss = criterion(logits, masks)
            boundary_loss = (
                boundary_criterion(logits, distance_maps)
                if distance_maps is not None
                else logits.new_zeros((), dtype=torch.float32)
            )
            if distance_maps is not None:
                loss = loss + boundary_weight * boundary_loss
        if training:
            if scaler is not None:
                old_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= old_scale
            else:
                loss.backward()
                optimizer.step()
                stepped = True
            if scheduler is not None and stepped:
                scheduler.step()
        prediction, target = logits.detach().sigmoid() >= threshold, masks >= 0.5
        ids, indices = list(batch["case_id"]), batch["slice_index"].tolist()
        metrics.update(prediction, target, ids, indices)
        batch_metrics = TumorSegmentationMetrics()
        batch_metrics.update(prediction, target, ids, indices)
        values = {
            "loss": float(loss.detach()),
            "bce_loss": float(bce.detach()),
            "dice_loss": float(dice_loss.detach()),
            "boundary_loss": float(boundary_loss.detach()),
        }
        count = len(images)
        samples += count
        for key in sums:
            sums[key] += values[key] * count
        score = batch_metrics.compute()
        step_values = {
            **values,
            **{
                key: score[key]
                for key in (
                    "dice",
                    "dice_micro",
                    "dice_tumor",
                    "negative_slice_specificity",
                )
            },
        }
        iterator.set_postfix(
            loss=f"{values['loss']:.4f}",
            bce=f"{values['bce_loss']:.4f}",
            dice_loss=f"{values['dice_loss']:.4f}",
            boundary=f"{values['boundary_loss']:.4f}",
        )
        if step_logger:
            step_logger(step_values)
    if not samples:
        raise ValueError("loader yielded no batches")
    return {key: value / samples for key, value in sums.items()} | metrics.compute(
        include_cases=not training
    )


def main(config: dict[str, Any] | None = None) -> None:
    cfg = dict(CFG if config is None else config)
    cfg.setdefault("ABLATION", "multi_window")
    mode = str(cfg["ABLATION"])
    if mode not in ABLATION_MODES:
        raise ValueError(f"ABLATION must be one of {ABLATION_MODES}, got {mode!r}")
    cfg["EXP_NAME"] = str(cfg.get("EXP_NAME") or ABLATION_EXPERIMENT_NAMES[mode])
    for key in (
        "BOUNDARY_LOSS_ENABLED",
        "BOUNDARY_LOSS_WEIGHT",
        "BOUNDARY_DISTANCE_CLIP",
    ):
        cfg.setdefault(key, CFG[key])
    seed_everything(int(cfg["SEED"]))
    device = torch.device(cfg["DEVICE"])
    if int(cfg["EARLY_STOPPING_PATIENCE"]) < 1 or not 0 <= float(cfg["THRESHOLD"]) <= 1:
        raise ValueError("invalid patience or threshold")
    boundary_weight = float(cfg["BOUNDARY_LOSS_WEIGHT"])
    boundary_max_distance = float(cfg["BOUNDARY_DISTANCE_CLIP"])
    if not math.isfinite(boundary_weight) or boundary_weight < 0:
        raise ValueError("BOUNDARY_LOSS_WEIGHT must be finite and nonnegative")
    if not math.isfinite(boundary_max_distance) or boundary_max_distance <= 0:
        raise ValueError("BOUNDARY_DISTANCE_CLIP must be finite and positive")
    valid_metrics = {
        f"val_{key}"
        for key in (
            "loss",
            "bce_loss",
            "dice_loss",
            "boundary_loss",
            "dice",
            "dice_micro",
            "iou_2d",
            "recall_2d",
            "precision_2d",
            "fp_2d",
            "fn_2d",
            "dice_tumor",
            "negative_slice_specificity",
            "dice_3d",
            "iou_3d",
            "case_dice_3d_tumor",
        )
    }
    if (
        cfg["CHECKPOINT_METRIC"] not in valid_metrics
        or cfg["EARLY_STOPPING_METRIC"] not in valid_metrics
    ):
        raise ValueError(
            "checkpoint and early-stopping metrics must be valid validation metrics"
        )
    directory = Path(cfg["SAVE_ROOT"]) / cfg["EXP_NAME"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf-8"
    )
    history_path = directory / "history_metrics.json"
    limit = cfg["TRAIN_BATCHES_PER_EPOCH"]
    train_loader = create_dataloader(
        cfg["DATASET_ROOT"],
        cfg["TRAIN_SPLIT"],
        mode=mode,
        batch_size=int(cfg["BATCH_SIZE"]),
        num_workers=int(cfg["NUM_WORKERS"]),
        pin_memory=bool(cfg["PIN_MEMORY"]),
        shuffle=True,
        training=True,
        balanced_sampling=bool(cfg["BALANCED_TRAIN_SAMPLING"]),
        batches_per_epoch=None if limit is None else int(limit),
        sampler_seed=int(cfg["SAMPLER_SEED"]),
        hard_negative_radius=int(cfg["HARD_NEGATIVE_RADIUS"]),
        positive_fraction=float(cfg["POSITIVE_FRACTION"]),
        hard_negative_fraction=float(cfg["HARD_NEGATIVE_FRACTION"]),
        easy_negative_fraction=float(cfg["EASY_NEGATIVE_FRACTION"]),
        return_metadata=True,
    )
    val_loader = create_dataloader(
        cfg["DATASET_ROOT"],
        cfg["VAL_SPLIT"],
        mode=mode,
        batch_size=int(cfg["BATCH_VAL"]),
        shuffle=False,
        num_workers=int(cfg["NUM_WORKERS"]),
        pin_memory=bool(cfg["PIN_MEMORY"]),
        return_metadata=True,
    )
    model_cfg = {
        "in_channels": 3 if mode == "multi_window" else 1,
        "base_channels": int(cfg["BASE_CHANNELS"]),
        "embed_dim": int(cfg["EMBED_DIM"]),
        "transformer_depth": int(cfg["TRANSFORMER_DEPTH"]),
        "transformer_heads": int(cfg["TRANSFORMER_HEADS"]),
        "mlp_dim": int(cfg["MLP_DIM"]),
        "mlp_ratio": float(cfg["MLP_RATIO"]),
        "dropout": float(cfg["DROPOUT"]),
        "img_dim": int(cfg["IMAGE_SIZE"]),
        "patch_dim": int(cfg["PATCH_DIM"]),
        "backbone_pretrained": bool(cfg["BACKBONE_PRETRAINED"])
        and not bool(str(cfg["RESUME_PATH"]).strip()),
    }
    if mode == "multi_window":
        model_cfg.update(
            cross_attention_enabled=bool(cfg["CROSS_ATTENTION_ENABLED"]),
            cross_attention_dim=int(cfg["CROSS_ATTENTION_DIM"]),
            cross_attention_heads=int(cfg["CROSS_ATTENTION_HEADS"]),
            cross_attention_dropout=float(cfg["CROSS_ATTENTION_DROPOUT"]),
            channel_attention_enabled=bool(cfg["CHANNEL_ATTENTION_ENABLED"]),
            channel_attention_reduction=int(cfg["CHANNEL_ATTENTION_REDUCTION"]),
        )
    model_class = TransUNet if mode == "multi_window" else SingleWindowTransUNet
    model = model_class(**model_cfg).to(device)
    criterion = BCEWeightedDiceLoss(
        float(cfg["BCE_POS_WEIGHT"]),
        float(cfg["DICE_LOSS_WEIGHT"]),
        float(cfg["DICE_BACKGROUND_WEIGHT"]),
        float(cfg["DICE_FOREGROUND_WEIGHT"]),
        float(cfg["DICE_SMOOTH"]),
        device,
    )
    optimizer = AdamW(
        model.parameters(), lr=float(cfg["LR"]), weight_decay=float(cfg["WEIGHT_DECAY"])
    )
    total, warmup = scheduler_step_counts(
        loader_batches=len(train_loader),
        train_batches_per_epoch=None,
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
    if device.type != "cuda":
        amp_dtype = None
    elif amp_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("AMP_DTYPE='bf16' requires CUDA BF16 support")
    scaler = torch.amp.GradScaler("cuda") if amp_dtype is torch.float16 else None
    start, best_checkpoint, best_early, stale = 1, None, None, 0
    history = []
    if str(cfg["RESUME_PATH"]).strip():
        state = load_checkpoint(cfg["RESUME_PATH"], device)
        model.load_state_dict(state["model_state"])
        if not cfg["RESUME_WEIGHTS_ONLY"]:
            if (
                state.get("checkpoint_metric") != cfg["CHECKPOINT_METRIC"]
                or state.get("early_stopping_metric") != cfg["EARLY_STOPPING_METRIC"]
            ):
                raise ValueError(
                    "resume metric settings differ from checkpoint; use weights-only resume"
                )
            optimizer.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
            if scaler is not None and state.get("scaler_state"):
                scaler.load_state_dict(state["scaler_state"])
            start, best_checkpoint, best_early, stale = (
                int(state["epoch"]) + 1,
                state["best_checkpoint"],
                state["best_early"],
                int(state["stale_epochs"]),
            )
            history = load_metrics_history(history_path)
            for row in state.get("metrics_history", []):
                history = append_metrics_history(history_path, history, row)
    run = _wandb_run(cfg)
    train_log, val_log = _step_logger(run, "train"), _step_logger(run, "val")
    try:
        for epoch in range(start, int(cfg["EPOCHS"]) + 1):
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch - 1)
            lr = optimizer.param_groups[0]["lr"]
            train = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                amp_dtype=amp_dtype,
                threshold=float(cfg["THRESHOLD"]),
                progress_desc=f"Epoch {epoch}/{cfg['EPOCHS']} | train",
                step_logger=train_log,
                boundary_enabled=bool(cfg["BOUNDARY_LOSS_ENABLED"]),
                boundary_weight=boundary_weight,
                boundary_max_distance=boundary_max_distance,
            )
            val = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                scheduler=None,
                scaler=None,
                amp_dtype=amp_dtype,
                threshold=float(cfg["THRESHOLD"]),
                progress_desc=f"Epoch {epoch}/{cfg['EPOCHS']} | val",
                step_logger=val_log,
                boundary_enabled=bool(cfg["BOUNDARY_LOSS_ENABLED"]),
                boundary_weight=boundary_weight,
                boundary_max_distance=boundary_max_distance,
            )
            values = {
                "epoch": epoch,
                "lr": lr,
                **{f"train_{k}": v for k, v in train.items()},
                **{f"val_{k}": v for k, v in val.items()},
            }
            print(values, flush=True)
            if run is not None:
                run.log(_finite(values))
            history = append_metrics_history(
                history_path,
                history,
                {
                    k: (v if math.isfinite(float(v)) else None)
                    for k, v in values.items()
                },
            )
            checkpoint_key, early_key = (
                cfg["CHECKPOINT_METRIC"],
                cfg["EARLY_STOPPING_METRIC"],
            )
            if checkpoint_key not in values or early_key not in values:
                raise ValueError("unknown checkpoint or early-stopping metric")

            checkpoint_improved, early_improved = (
                _improved(checkpoint_key, values[checkpoint_key], best_checkpoint),
                _improved(early_key, values[early_key], best_early),
            )
            if checkpoint_improved:
                best_checkpoint = values[checkpoint_key]
            if early_improved:
                best_early, stale = values[early_key], 0
            else:
                stale += 1
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler else None,
                "best_checkpoint": best_checkpoint,
                "best_early": best_early,
                "stale_epochs": stale,
                "checkpoint_metric": checkpoint_key,
                "early_stopping_metric": early_key,
                "metrics_history": history,
                "config": cfg,
                "model_config": model_cfg,
            }
            save_checkpoint(directory / "last.pt", state)
            if checkpoint_improved:
                save_checkpoint(directory / "best.pt", state)
            if stale >= int(cfg["EARLY_STOPPING_PATIENCE"]):
                print(
                    f"Early stopping: {early_key} did not improve for {stale} epochs."
                )
                break
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
