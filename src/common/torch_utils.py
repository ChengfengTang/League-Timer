"""Small torch helpers shared by training and inference."""
from __future__ import annotations

import os

import torch

# X3D uses avg_pool3d, which MPS does not implement; fall back to CPU for those ops.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device(prefer: str = "auto") -> torch.device:
    """Choose the best available device.

    ``auto`` prefers CUDA (cloud training), then Apple MPS (Mac inference),
    then CPU. Pass an explicit string to override.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
