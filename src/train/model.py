"""Backbone factory for the clip classifier.

All backbones consume input of shape ``(B, C, T, H, W)`` and emit raw logits of
shape ``(B, num_classes)``. ``build_model`` returns the model, a normalization
spec, and the list of "head" parameters (the freshly initialised layer) so the
training loop can warm up the head before unfreezing the rest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class BackboneSpec:
    mean: List[float]
    std: List[float]


class MobileNetBaseline(nn.Module):
    """Per-frame 2D CNN features, mean-pooled over time. Fast sanity-check model."""

    def __init__(self, num_classes: int):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = m.classifier[0].in_features  # 576
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T, H, W)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.features(x)
        x = self.pool(x).flatten(1)            # (B*T, feat)
        x = x.view(b, t, -1).mean(dim=1)        # temporal average pool
        return self.head(x)


def _build_x3d_s(num_classes: int) -> Tuple[nn.Module, BackboneSpec, List[nn.Parameter]]:
    # Pulls the architecture + Kinetics weights via torch.hub (needs pytorchvideo).
    model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=True)
    head = model.blocks[-1]
    in_features = head.proj.in_features
    head.proj = nn.Linear(in_features, num_classes)
    # The pretrained head ends in a softmax activation; we want raw logits.
    if hasattr(head, "activation"):
        head.activation = nn.Identity()
    spec = BackboneSpec(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    return model, spec, list(head.proj.parameters())


def _build_r2plus1d(num_classes: int) -> Tuple[nn.Module, BackboneSpec, List[nn.Parameter]]:
    from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    spec = BackboneSpec(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989])
    return model, spec, list(model.fc.parameters())


def _build_mobilenet(num_classes: int) -> Tuple[nn.Module, BackboneSpec, List[nn.Parameter]]:
    model = MobileNetBaseline(num_classes)
    spec = BackboneSpec(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return model, spec, list(model.head.parameters())


_BUILDERS = {
    "x3d_s": _build_x3d_s,
    "r2plus1d_18": _build_r2plus1d,
    "mobilenet_baseline": _build_mobilenet,
}


def available_backbones() -> List[str]:
    return list(_BUILDERS)


def build_model(backbone: str, num_classes: int) -> Tuple[nn.Module, BackboneSpec, List[nn.Parameter]]:
    if backbone not in _BUILDERS:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose from: {', '.join(_BUILDERS)}"
        )
    return _BUILDERS[backbone](num_classes)
