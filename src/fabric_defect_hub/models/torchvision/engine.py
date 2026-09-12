"""The actual training/evaluation loop for the torchvision detection
backend. `torchvision.models.detection` ships architectures and losses but,
unlike Ultralytics or Lightning-based anomalib, no `Trainer` — the loop
below is ours to write.

Structure follows torchvision's own official reference training script
(`torchvision/references/detection/{engine,train}.py`, the canonical
"how to actually train these models" source): per-batch forward returns a
loss dict in train mode (verified live — see `presets.py` docstring), which
we sum and backprop; a linear LR warmup covers the first `warmup_epochs`
(large early gradients from a freshly swapped head can otherwise diverge);
gradient norm clipping guards against the same instability on small
batches. `evaluate()` accumulates predictions into `torchmetrics`' COCO-
style `MeanAveragePrecision` rather than hand-rolling COCO-mAP matching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from fabric_defect_hub.core.progress import ProgressReporter, note


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    train_loss_components: dict[str, float]
    lr: float
    val_metrics: dict[str, float] = field(default_factory=dict)


def build_optimizer(model, name: str, lr: float, momentum: float, weight_decay: float):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}; expected 'sgd' or 'adamw'.")


def build_lr_scheduler(optimizer, name: str, epochs: int, step_size: int, gamma: float):
    import torch

    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    raise ValueError(f"unknown lr_scheduler {name!r}; expected 'none', 'step', or 'cosine'.")


def _build_warmup_scheduler(optimizer, warmup_iters: int):
    import torch

    if warmup_iters <= 0:
        return None
    warmup_factor = 1.0 / 1000
    return torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    )


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch: int,
    warmup_scheduler=None,
    grad_clip_norm: float | None = None,
    amp: bool = False,
    scaler=None,
    log_fn: Callable[[str], None] | None = None,
    task: str = "detect",
    progress: ProgressReporter | None = None,
) -> tuple[float, dict[str, float]]:
    """One training epoch. Returns (mean total loss, mean per-component losses).

    `amp`/`scaler`: mixed-precision forward (`torch.autocast`) and loss
    scaling (`torch.amp.GradScaler`) for CUDA training. `run_training`
    already resolves `amp` to `False` outside CUDA before calling this, and
    a `GradScaler(enabled=False)` is a documented no-op passthrough, so the
    scale/unscale/step calls below are unconditional and safe on CPU/MPS.
    """

    import torch

    model.train()
    loss_sums: dict[str, float] = {}
    total_loss_sum = 0.0
    num_batches = 0

    for images, targets in data_loader:
        if task == "segmentation":
            images_stacked = torch.stack(images, dim=0).to(device)
            masks_stacked = torch.stack(targets, dim=0).to(device)

            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images_stacked)
                loss = torch.nn.BCEWithLogitsLoss()(logits, masks_stacked)
                loss_dict = {"loss_seg": loss}
                total_loss = loss
        else:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()} for t in targets]

            with torch.autocast(device_type=device.type, enabled=amp):
                loss_dict = model(images, targets)
                total_loss = sum(loss_dict.values())

        loss_value = float(total_loss.detach())
        if not math.isfinite(loss_value):
            if log_fn:
                log_fn(f"[epoch {epoch}] non-finite loss ({loss_value}); skipping batch.")
            continue

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip_norm
            )
        scaler.step(optimizer)
        scaler.update()
        if warmup_scheduler is not None:
            warmup_scheduler.step()

        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach())
        total_loss_sum += loss_value
        num_batches += 1
        if progress is not None:
            progress.update(loss=loss_value, lr=optimizer.param_groups[0]["lr"])

    if num_batches == 0:
        return float("nan"), {}
    mean_components = {k: v / num_batches for k, v in loss_sums.items()}
    return total_loss_sum / num_batches, mean_components


def evaluate(
    model,
    data_loader,
    device,
    with_masks: bool = False,
    task: str = "detect",
) -> dict[str, float]:
    """Run inference over `data_loader` and compute COCO-style mAP via
    `torchmetrics.detection.MeanAveragePrecision` (pycocotools backend).
    Returns a flat metrics dict (`map`, `map_50`, `map_75`, `mar_100`, ...).
    """

    import torch

    if task == "segmentation":
        from fabric_defect_hub.evaluation.segmentation import _iou, _dice, _pixel_f1

        model.eval()
        total_iou = 0.0
        total_dice = 0.0
        total_f1 = 0.0
        num_eval = 0

        with torch.no_grad():
            for images, targets in data_loader:
                images_stacked = torch.stack(images, dim=0).to(device)
                logits = model(images_stacked)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).cpu().numpy()

                for pred_mask, gt_mask_tensor in zip(preds, targets):
                    gt_mask = gt_mask_tensor.cpu().numpy() > 0.5

                    val_iou = _iou(gt_mask[0], pred_mask[0])
                    val_dice = _dice(gt_mask[0], pred_mask[0])
                    val_f1 = _pixel_f1(gt_mask[0], pred_mask[0])

                    total_iou += val_iou
                    total_dice += val_dice
                    total_f1 += val_f1
                    num_eval += 1

        if num_eval == 0:
            return {}

        return {
            "miou": total_iou / num_eval,
            "dice": total_dice / num_eval,
            "pixel_f1": total_f1 / num_eval,
            "map": total_iou / num_eval,  # for checkpointing compatibility
        }

    from torchmetrics.detection import MeanAveragePrecision

    model.eval()
    iou_types = ("bbox", "segm") if with_masks else ("bbox",)
    metric = MeanAveragePrecision(iou_type=list(iou_types) if len(iou_types) > 1 else iou_types[0])

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            preds = []
            for out in outputs:
                entry = {
                    "boxes": out["boxes"].detach().cpu(),
                    "scores": out["scores"].detach().cpu(),
                    "labels": out["labels"].detach().cpu(),
                }
                if with_masks:
                    entry["masks"] = (out["masks"].detach().cpu() > 0.5).squeeze(1).to(torch.uint8)
                preds.append(entry)

            gts = []
            for t in targets:
                entry = {
                    "boxes": t["boxes"].detach().cpu() if hasattr(t["boxes"], "detach") else t["boxes"],
                    "labels": t["labels"].detach().cpu() if hasattr(t["labels"], "detach") else t["labels"],
                }
                if with_masks:
                    entry["masks"] = t["masks"].detach().cpu().to(torch.uint8)
                gts.append(entry)

            metric.update(preds, gts)

    result = metric.compute()
    flat: dict[str, float] = {}
    for key, value in result.items():
        if hasattr(value, "numel") and value.numel() == 1:
            flat[key] = float(value)
    return flat


def run_training(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    optimizer_name: str,
    lr: float,
    momentum: float,
    weight_decay: float,
    lr_scheduler_name: str,
    step_size: int,
    gamma: float,
    warmup_epochs: int,
    grad_clip_norm: float | None,
    patience: int,
    with_masks: bool,
    amp: bool = False,
    resume_state: dict[str, Any] | None = None,
    on_epoch_end: Callable[[EpochLog, Any, Any, float, bool], None] | None = None,
    task: str = "detect",
) -> tuple[list[EpochLog], float]:
    """Full multi-epoch loop with warmup, LR scheduling, and mAP-based early
    stopping. Returns `(per-epoch logs, final best val-mAP)`; the caller
    (`adapter.py`) is responsible for checkpointing on each `on_epoch_end`
    call, since only it knows where model artifacts should be written.

    `amp` requests mixed precision; it is only actually enabled on CUDA (no
    numerically-stable half-precision kernel path exists for these detection
    ops on CPU/MPS, so it is silently treated as disabled there rather than
    erroring — this is the one thing this function decides on the caller's
    behalf instead of trusting the flag literally).

    `resume_state`, if given (as produced by
    `TorchvisionAdapter._load_resume_checkpoint`), restores optimizer/
    scheduler state and continues from `resume_state['epoch'] + 1` instead
    of epoch 0, with `best_map` carried over. Early-stopping's own
    "epochs without improvement" counter always restarts at 0 on resume —
    tracking it across a resume would require persisting one more counter
    for a rarely-hit edge case (resuming right as patience was about to
    trigger), so it is intentionally not carried over.
    """

    import torch

    optimizer = build_optimizer(model, optimizer_name, lr, momentum, weight_decay)
    scheduler = build_lr_scheduler(optimizer, lr_scheduler_name, epochs, step_size, gamma)

    start_epoch = 0
    best_map = -1.0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        if scheduler is not None and resume_state.get("scheduler_state") is not None:
            scheduler.load_state_dict(resume_state["scheduler_state"])
        start_epoch = resume_state["epoch"] + 1
        best_map = resume_state.get("best_map", -1.0)

    effective_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=effective_amp)

    warmup_iters = 0
    if warmup_epochs > 0:
        warmup_iters = min(len(train_loader), len(train_loader) * warmup_epochs)

    logs: list[EpochLog] = []
    epochs_without_improvement = 0

    for epoch in range(start_epoch, epochs):
        warmup_scheduler = _build_warmup_scheduler(optimizer, warmup_iters) if epoch == 0 else None
        # One reporter per epoch rather than one for the whole run: the
        # epoch number is the thing a watcher actually wants in the line,
        # and `len(train_loader)` is only meaningful per epoch anyway.
        with ProgressReporter(
            f"torchvision train epoch {epoch + 1}/{epochs}",
            total=len(train_loader),
            unit="batch",
        ) as progress:
            train_loss, components = train_one_epoch(
                model, optimizer, train_loader, device, epoch,
                warmup_scheduler=warmup_scheduler, grad_clip_norm=grad_clip_norm,
                amp=effective_amp, scaler=scaler,
                task=task, log_fn=note, progress=progress,
            )
        if scheduler is not None and epoch > 0:
            scheduler.step()

        if val_loader is not None:
            note(f"torchvision epoch {epoch + 1}/{epochs}: validating ({len(val_loader)} batches)")
            val_metrics = evaluate(model, val_loader, device, with_masks=with_masks, task=task)
        else:
            val_metrics = {}
        current_lr = optimizer.param_groups[0]["lr"]
        log = EpochLog(epoch=epoch, train_loss=train_loss, train_loss_components=components, lr=current_lr, val_metrics=val_metrics)
        logs.append(log)

        current_map = val_metrics.get("map", None)
        improved = current_map is not None and current_map > best_map
        if improved:
            best_map = current_map
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if on_epoch_end is not None:
            on_epoch_end(log, optimizer, scheduler, best_map, improved)

        summary = f"torchvision epoch {epoch + 1}/{epochs}: train_loss {train_loss:.4f} | lr {current_lr:.6f}"
        if val_metrics:
            summary += " | " + " ".join(f"{k} {v:.4f}" for k, v in sorted(val_metrics.items()))
        note(summary)

        if patience > 0 and epochs_without_improvement >= patience:
            note(f"torchvision: early stop after {epochs_without_improvement} epoch(s) without improvement")
            break

    return logs, best_map
