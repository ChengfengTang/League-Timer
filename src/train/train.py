"""Fine-tune the clip classifier and report per-class metrics.

Run locally or on a cloud GPU::

    python -m src.train.train --config configs/ezreal.yaml

Saves the best checkpoint (by macro-F1 over the ability classes, ignoring
``background``) to ``models/<champion>/best.pt`` along with everything the
recognizer needs to rebuild and normalize inputs identically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.config import Config
from src.common.torch_utils import pick_device
from src.dataset.clip_dataset import ClipTransform, VideoClipDataset, class_weights
from src.train.model import build_model


def set_requires_grad(model: nn.Module, value: bool) -> None:
    for p in model.parameters():
        p.requires_grad = value


@torch.no_grad()
def evaluate(model, loader, device, num_classes) -> Dict:
    model.eval()
    all_pred: List[int] = []
    all_true: List[int] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_pred.extend(pred.tolist())
        all_true.extend(y.numpy().tolist())
    labels = list(range(num_classes))
    p, r, f1, sup = precision_recall_fscore_support(
        all_true, all_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    return {"precision": p, "recall": r, "f1": f1, "support": sup, "cm": cm}


def print_metrics(metrics: Dict, classes: List[str]) -> None:
    print(f"  {'class':<12} {'prec':>6} {'rec':>6} {'f1':>6} {'n':>5}")
    for i, name in enumerate(classes):
        print(f"  {name:<12} {metrics['precision'][i]:6.3f} {metrics['recall'][i]:6.3f} "
              f"{metrics['f1'][i]:6.3f} {int(metrics['support'][i]):5d}")
    print("  confusion matrix (rows=true, cols=pred):")
    print("   " + "".join(f"{c[:6]:>7}" for c in classes))
    for i, name in enumerate(classes):
        print(f"  {name[:6]:>3}" + "".join(f"{int(v):7d}" for v in metrics["cm"][i]))


def ability_macro_f1(metrics: Dict, classes: List[str]) -> float:
    idxs = [i for i, c in enumerate(classes) if c != "background"]
    if not idxs:
        return 0.0
    return float(np.mean([metrics["f1"][i] for i in idxs]))


def train(config_path: str, clips_root: str, models_root: str, device_str: str) -> None:
    cfg = Config.load(config_path)
    tcfg = cfg.section("train")
    backbone = str(tcfg.get("backbone", "x3d_s"))
    epochs = int(tcfg.get("epochs", 25))
    batch_size = int(tcfg.get("batch_size", 8))
    lr = float(tcfg.get("lr", 5e-4))
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    num_workers = int(tcfg.get("num_workers", 4))
    freeze_epochs = int(tcfg.get("freeze_backbone_epochs", 3))

    manifest_path = Path(clips_root) / cfg.champion / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}. Run the dataset builder first.")

    device = pick_device(device_str)
    print(f"Device: {device} | backbone: {backbone}")

    model, spec, head_params = build_model(backbone, cfg.num_classes)
    model.to(device)

    train_tf = ClipTransform(cfg.crop_size, spec.mean, spec.std, train=True,
                             frame_mode=cfg.frame_mode, spatial_jitter=cfg.spatial_jitter)
    val_tf = ClipTransform(cfg.crop_size, spec.mean, spec.std, train=False,
                           frame_mode=cfg.frame_mode)
    train_ds = VideoClipDataset(manifest_path, "train", train_tf)
    try:
        val_ds = VideoClipDataset(manifest_path, "val", val_tf)
    except ValueError:
        val_ds = None
        print("No validation split present; training without held-out metrics.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers) if val_ds else None)

    weights = class_weights(train_ds.label_ids, cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    models_dir = Path(models_root) / cfg.champion
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "best.pt"

    best_score = -1.0
    frozen = None  # unknown; forces optimizer construction on first epoch
    optimizer = None
    can_freeze = backbone != "mobilenet_baseline" and freeze_epochs > 0

    for epoch in range(1, epochs + 1):
        # Head warm-up: freeze backbone for the first few epochs.
        want_frozen = can_freeze and epoch <= freeze_epochs
        if want_frozen != frozen:
            if want_frozen:
                set_requires_grad(model, False)
                for p in head_params:
                    p.requires_grad = True
            else:
                set_requires_grad(model, True)
            frozen = want_frozen
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}{' [head]' if frozen else ''}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{running / max(n, 1):.4f}")

        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, cfg.num_classes)
            print(f"[epoch {epoch}] train_loss={running / max(n, 1):.4f}")
            print_metrics(metrics, cfg.classes)
            score = ability_macro_f1(metrics, cfg.classes)
            print(f"  ability macro-F1 = {score:.4f} (best {max(best_score, 0):.4f})")
        else:
            score = -epoch  # without val, just keep the latest

        if score > best_score:
            best_score = score
            torch.save({
                "state_dict": model.state_dict(),
                "backbone": backbone,
                "classes": cfg.classes,
                "mean": spec.mean,
                "std": spec.std,
                "crop_size": cfg.crop_size,
                "num_frames": cfg.num_frames,
                "sample_fps": cfg.sample_fps,
                "frame_mode": cfg.frame_mode,
                "hud_mask": cfg.hud_mask,
                "localize_enabled": cfg.localize_enabled,
                "localize": cfg.section("localize") if cfg.localize_enabled else None,
                "champion": cfg.champion,
                "epoch": epoch,
                "ability_macro_f1": best_score,
            }, ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}")

    print(f"\nDone. Best ability macro-F1: {best_score:.4f}")
    print(f"Checkpoint: {ckpt_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the clip classifier.")
    p.add_argument("--config", required=True)
    p.add_argument("--clips-root", default="data/clips")
    p.add_argument("--models-root", default="models")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    args = p.parse_args()
    train(args.config, args.clips_root, args.models_root, args.device)


if __name__ == "__main__":
    main()
