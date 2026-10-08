"""Backbone factory. Every model outputs a single logit (P(defective) after sigmoid).

Candidates span the accuracy / latency spectrum for CPU edge deployment:

=====================  ========  ====================================================
backbone               params    rationale
=====================  ========  ====================================================
mobilenet_v3_large     ~4.2M     built for CPU/edge latency; default production pick
efficientnet_b0        ~4.0M     better accuracy per FLOP, slower on CPU (SiLU, SE)
resnet18               ~11.2M    simple, very well-understood baseline
resnet50               ~23.5M    higher-capacity reference point
=====================  ========  ====================================================
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_model(name: str, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    if name == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.mobilenet_v3_large(weights=w)
        m.classifier[2] = nn.Dropout(dropout)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
    elif name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, 1))
    elif name in ("resnet18", "resnet50"):
        ctor, w = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
            "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
        }[name]
        m = ctor(weights=w if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 1))
    else:
        raise ValueError(f"unknown backbone: {name}")
    return m


def head_module(model: nn.Module, name: str) -> nn.Module:
    """The newly initialised classification head (trained alone during warm-up)."""
    return model.fc if name.startswith("resnet") else model.classifier


def gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    """Last spatial feature map - target layer for Grad-CAM."""
    return model.layer4 if name.startswith("resnet") else model.features[-1]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class ProbabilityWrapper(nn.Module):
    """Export wrapper: image batch (N,3,H,W) -> P(defective) (N,)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x)).squeeze(1)
