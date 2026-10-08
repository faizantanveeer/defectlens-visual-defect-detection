"""Two-stage transfer-learning trainer.

Stage 1 (head warm-up): backbone frozen (BatchNorm statistics too), only the new head
    learns. Prevents large random-head gradients from wrecking pretrained features.
Stage 2 (fine-tune): everything unfrozen, discriminative learning rates
    (backbone 10x lower than head), cosine decay, early stopping on validation AP.

Class imbalance is handled by ``pos_weight`` in the BCE loss (default) or a weighted
sampler, and - more importantly for the operating point - by tuning the decision
threshold on validation data for a recall-weighted F-beta instead of using 0.5.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from defect_detection.config import Config, dump_config
from defect_detection.data.split import validate_split
from defect_detection.evaluation.metrics import ranking_metrics, select_threshold, threshold_metrics
from defect_detection.preprocessing import PreprocessSpec
from defect_detection.training.dataset import DefectDataset, build_augmentation, make_weighted_sampler
from defect_detection.training.models import build_model, count_parameters, head_module

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str) -> torch.device:
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def spec_from_config(cfg: Config) -> PreprocessSpec:
    return PreprocessSpec(cfg.preprocess.image_size, cfg.preprocess.mean, cfg.preprocess.std)


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    probs, losses, n = [], 0.0, 0
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    for x, y, _ in loader:
        logits = model(x.to(device)).squeeze(1)
        losses += float(loss_fn(logits, y.to(device)))
        n += len(y)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs), losses / max(n, 1)


def _set_train_mode(model: nn.Module, head: nn.Module, frozen_backbone: bool, freeze_bn_stats: bool) -> None:
    if frozen_backbone:
        model.eval()  # backbone is a fixed feature extractor (incl. BN statistics)
        head.train()
        return
    model.train()
    if freeze_bn_stats:
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()  # use running stats in forward and do not update them


def _train_epoch(model, loader, criterion, optimizer, device, frozen_backbone: bool, head,
                 freeze_bn_stats: bool = True, scheduler=None) -> float:
    _set_train_mode(model, head, frozen_backbone, freeze_bn_stats)
    total, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x).squeeze(1), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # per-iteration schedule
        total += float(loss) * len(y)
        n += len(y)
    return total / max(n, 1)


def run_training(cfg: Config) -> dict:
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    seed_everything(cfg.data.seed)
    device = resolve_device(cfg.train.device)
    spec = spec_from_config(cfg)

    split = pd.read_csv(cfg.data.split_path)
    validate_split(split)
    split.to_csv(run_dir / "split.csv", index=False)
    train_df, val_df = split[split.split == "train"], split[split.split == "val"]

    train_ds = DefectDataset(train_df, cfg.data.raw_dir, spec, build_augmentation(cfg.augment))
    val_ds = DefectDataset(val_df, cfg.data.raw_dir, spec)
    labels = train_df["label"].to_numpy().astype(int)
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())

    sampler = None
    pos_weight = None
    if cfg.train.imbalance_strategy == "weighted_sampler":
        sampler = make_weighted_sampler(labels, cfg.data.seed)
    elif cfg.train.imbalance_strategy == "pos_weight":
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    loader_kw = {
        "num_workers": cfg.train.num_workers,
        "persistent_workers": cfg.train.num_workers > 0,
        "pin_memory": device.type == "cuda",
    }
    gen = torch.Generator().manual_seed(cfg.data.seed)
    train_loader = DataLoader(
        train_ds, cfg.train.batch_size, shuffle=sampler is None, sampler=sampler,
        drop_last=len(train_ds) > cfg.train.batch_size, generator=gen, **loader_kw,
    )
    val_loader = DataLoader(val_ds, cfg.train.batch_size * 2, shuffle=False, **loader_kw)

    model = build_model(cfg.train.backbone, cfg.train.pretrained, cfg.train.dropout).to(device)
    head = head_module(model, cfg.train.backbone)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    log.info(
        "training start",
        extra={
            "backbone": cfg.train.backbone, "params": count_parameters(model), "device": str(device),
            "train": len(train_ds), "val": len(val_ds), "pos": n_pos, "neg": n_neg,
            "imbalance": cfg.train.imbalance_strategy,
            "pos_weight": None if pos_weight is None else round(float(pos_weight), 3),
        },
    )

    ckpt_path = run_dir / "checkpoint.pt"
    history: list[dict] = []
    best = {"ap": -1.0, "loss": float("inf"), "epoch": -1}
    stale = 0

    def evaluate_and_maybe_save(epoch: int, stage: str, train_loss: float, lr: float) -> bool:
        nonlocal stale
        probs, val_loss = predict_proba(model, val_loader, device)
        y = val_df["label"].to_numpy()
        rank = ranking_metrics(y, probs)
        at05 = threshold_metrics(y, probs, 0.5)
        rec = {
            "epoch": epoch, "stage": stage, "lr": lr, "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5), "val_ap": round(rank["average_precision"], 5),
            "val_auc": round(rank["roc_auc"], 5), "val_f1@0.5": round(at05["f1"], 4),
            "val_recall@0.5": round(at05["recall"], 4), "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        log.info("epoch", extra=rec)
        improved = rank["average_precision"] > best["ap"] + 1e-6 or (
            abs(rank["average_precision"] - best["ap"]) <= 1e-6 and val_loss < best["loss"]
        )
        if improved:
            best.update(ap=rank["average_precision"], loss=val_loss, epoch=epoch)
            torch.save(
                {"state_dict": model.state_dict(), "backbone": cfg.train.backbone, "epoch": epoch,
                 "val_ap": rank["average_precision"], "val_loss": val_loss,
                 "config": cfg.model_dump(mode="json")},
                ckpt_path,
            )
            stale = 0
        else:
            stale += 1
        return improved

    start = time.time()
    epoch = 0
    # ---- Stage 1: head warm-up -------------------------------------------------------
    for p in model.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.train.head_lr, weight_decay=cfg.train.weight_decay)
    for _ in range(cfg.train.head_epochs):
        epoch += 1
        t0 = time.time()
        loss = _train_epoch(model, train_loader, criterion, opt, device, True, head)
        evaluate_and_maybe_save(epoch, "head", loss, cfg.train.head_lr)

    # ---- Stage 2: full fine-tune -----------------------------------------------------
    for p in model.parameters():
        p.requires_grad = True
    head_ids = {id(p) for p in head.parameters()}
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": cfg.train.backbone_lr},
            {"params": list(head.parameters()), "lr": cfg.train.backbone_lr * 10},
        ],
        weight_decay=cfg.train.weight_decay,
    )
    # Linear warm-up then cosine decay, stepped per iteration. Warm-up matters: Adam's first
    # updates are ~lr-sized for *every* weight, which otherwise jolts pretrained features.
    steps_per_epoch = len(train_loader)
    total_steps = max(cfg.train.finetune_epochs * steps_per_epoch, 1)
    warm_steps = min(cfg.train.warmup_epochs * steps_per_epoch, total_steps - 1)

    def lr_lambda(step: int) -> float:
        if step < warm_steps:
            return (step + 1) / warm_steps
        progress = (step - warm_steps) / max(total_steps - warm_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    stale = 0
    for _ in range(cfg.train.finetune_epochs):
        epoch += 1
        t0 = time.time()
        loss = _train_epoch(model, train_loader, criterion, opt, device, False, head,
                            cfg.train.freeze_bn_stats, sched)
        lr = opt.param_groups[0]["lr"]
        evaluate_and_maybe_save(epoch, "finetune", loss, lr)
        if stale >= cfg.train.early_stopping_patience:
            log.info("early stopping", extra={"epoch": epoch, "best_epoch": best["epoch"]})
            break
    train_seconds = time.time() - start

    # ---- Threshold calibration on validation ----------------------------------------
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    val_probs, _ = predict_proba(model, val_loader, device)
    tcfg = cfg.threshold
    choice = select_threshold(
        val_df["label"].to_numpy(), val_probs, tcfg.strategy, tcfg.beta, tcfg.min_recall, tcfg.fixed,
        tcfg.search_bounds,
    )
    val_df.assign(defect_probability=val_probs).to_csv(run_dir / "val_predictions.csv", index=False)
    with open(run_dir / "history.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(h) + "\n" for h in history)
    _plot_history(history, run_dir / "training_curves.png")

    summary = {
        "backbone": cfg.train.backbone,
        "best_epoch": best["epoch"],
        "epochs_run": epoch,
        "best_val_ap": best["ap"],
        "best_val_loss": best["loss"],
        "train_seconds": round(train_seconds, 1),
        "threshold": choice.threshold,
        "threshold_strategy": choice.strategy,
        "val_metrics_at_threshold": choice.val_metrics,
        "parameters": count_parameters(model),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("training done", extra={k: summary[k] for k in ("best_epoch", "best_val_ap", "threshold")})
    return summary


def _plot_history(history: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    a.plot(ep, [h["train_loss"] for h in history], label="train loss")
    a.plot(ep, [h["val_loss"] for h in history], label="val loss")
    a.set_xlabel("epoch")
    a.legend()
    b.plot(ep, [h["val_ap"] for h in history], label="val AP")
    b.plot(ep, [h["val_auc"] for h in history], label="val ROC-AUC")
    b.plot(ep, [h["val_f1@0.5"] for h in history], label="val F1@0.5")
    b.set_ylim(0, 1.02)
    b.set_xlabel("epoch")
    b.legend()
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
