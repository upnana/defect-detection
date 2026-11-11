"""TinyDETR-Pose：轻量 CNN 回归相机系平移 (mm) + 6D 旋转。"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyDETRPoseNet(nn.Module):
    def __init__(self, out_dim: int = 9):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            from torchvision.models import mobilenet_v3_small

            backbone = mobilenet_v3_small(weights=None)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)
