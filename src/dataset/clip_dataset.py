"""Torch dataset + transforms for the extracted clips.

Reads the ``manifest.json`` produced by ``src.dataset.build`` and yields
normalized clip tensors of shape ``(C, T, H, W)`` plus an integer label id.
Stored clips already have their shorter side resized to ``crop_size``; the
transform only crops, optionally flips, and normalizes.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_tchw(clip: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (T, 3, H, W) float in [0, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(clip)).float().div_(255.0)
    return t.permute(0, 3, 1, 2).contiguous()


def _ensure_min_size(t: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = t.shape
    if min(h, w) >= size:
        return t
    return torch.nn.functional.interpolate(
        t, size=(max(h, size), max(w, size)), mode="bilinear", align_corners=False
    )


class ClipTransform:
    """Spatial fit + optional flip/jitter + per-channel normalization.

    Clips are expected to arrive already shaped by ``preprocess_clip`` for the
    matching ``frame_mode``:

    - ``letterbox``: frames are already ``crop_size`` x ``crop_size`` (whole frame
      kept). At eval we just normalize; in train we apply ``spatial_jitter``
      (random zoom-in + reposition) + optional hflip.
    - ``center_crop`` (legacy): frames have their short side at ``crop_size``; we
      center/random crop to a square as before.
    """

    def __init__(self, crop_size: int, mean: Sequence[float], std: Sequence[float],
                 train: bool, hflip: bool = True, frame_mode: str = "letterbox",
                 spatial_jitter: float = 0.0):
        self.crop_size = crop_size
        self.train = train
        self.hflip = hflip
        self.frame_mode = frame_mode
        self.spatial_jitter = spatial_jitter
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, clip: np.ndarray) -> torch.Tensor:
        t = _to_tchw(clip)               # (T, C, H, W)
        if self.frame_mode == "center_crop":
            t = self._center_crop(t)
        else:
            t = self._letterbox(t)
        if self.train and self.hflip and random.random() < 0.5:
            t = torch.flip(t, dims=[3])
        t = (t - self.mean) / self.std
        return t.permute(1, 0, 2, 3).contiguous()   # (C, T, H, W)

    def _letterbox(self, t: torch.Tensor) -> torch.Tensor:
        size = self.crop_size
        if self.train and self.spatial_jitter > 0:
            # Zoom in by up to (1 + jitter) and take a random-position crop so the
            # cast is seen at varied screen locations/scales.
            scale = 1.0 + random.uniform(0.0, self.spatial_jitter)
            _, _, h, w = t.shape
            nh, nw = max(size, int(round(h * scale))), max(size, int(round(w * scale)))
            t = torch.nn.functional.interpolate(
                t, size=(nh, nw), mode="bilinear", align_corners=False)
            top = random.randint(0, nh - size)
            left = random.randint(0, nw - size)
            t = t[:, :, top:top + size, left:left + size]
        else:
            t = _ensure_min_size(t, size)
            _, _, h, w = t.shape
            if (h, w) != (size, size):   # safety: center-fit unexpected sizes
                top, left = (h - size) // 2, (w - size) // 2
                t = t[:, :, top:top + size, left:left + size]
        return t

    def _center_crop(self, t: torch.Tensor) -> torch.Tensor:
        t = _ensure_min_size(t, self.crop_size)
        _, _, h, w = t.shape
        size = self.crop_size
        if self.train:
            top = random.randint(0, h - size)
            left = random.randint(0, w - size)
        else:
            top = (h - size) // 2
            left = (w - size) // 2
        return t[:, :, top:top + size, left:left + size]


class VideoClipDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str, transform: Callable):
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        self.root = manifest_path.parent
        self.classes: List[str] = self.manifest["classes"]
        self.transform = transform
        self.items = [it for it in self.manifest["items"] if it.get("split") == split]
        if not self.items:
            raise ValueError(f"No items for split='{split}' in {manifest_path}")

    def __len__(self) -> int:
        return len(self.items)

    @property
    def label_ids(self) -> List[int]:
        return [int(it["label_id"]) for it in self.items]

    def __getitem__(self, idx: int):
        it = self.items[idx]
        clip = np.load(self.root / it["clip"])
        x = self.transform(clip)
        y = int(it["label_id"])
        return x, y


def class_weights(label_ids: Sequence[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = torch.zeros(num_classes, dtype=torch.float)
    for y in label_ids:
        counts[y] += 1
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights
