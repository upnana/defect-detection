"""MobileNetV3 / EfficientNet 图像分类头（品种 / 品类识别）。"""

from __future__ import annotations

import torch
import torch.nn as nn

BACKBONE_MOBILENET = "mobilenet_v3_small"
BACKBONE_EFFICIENTNET = "efficientnet_b0"


def build_variety_classifier(backbone: str, num_classes: int) -> nn.Module:
    name = (backbone or BACKBONE_MOBILENET).lower()
    if name in (BACKBONE_EFFICIENTNET, "efficientnet", "efficientnet-b0"):
        return _build_efficientnet_b0(num_classes)
    return _build_mobilenet_v3_small(num_classes)


def _build_mobilenet_v3_small(num_classes: int) -> nn.Module:
    try:
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        from torchvision.models import mobilenet_v3_small

        model = mobilenet_v3_small(weights=None)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_efficientnet_b0(num_classes: int) -> nn.Module:
    try:
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        from torchvision.models import efficientnet_b0

        model = efficientnet_b0(weights=None)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def classifier_feature_dim(backbone: str) -> int:
    name = (backbone or BACKBONE_MOBILENET).lower()
    if name in (BACKBONE_EFFICIENTNET, "efficientnet", "efficientnet-b0"):
        return 1280
    return 1024
