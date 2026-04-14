from __future__ import annotations

import torch.nn as nn
from torchvision import models

try:
    import timm
except Exception:  # pragma: no cover - optional dependency path
    timm = None


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """
    Build a classification backbone.

    Priority:
    1) Try timm for broad model support.
    2) Fallback to a small torchvision mapping for reliability.
    """
    lowered = model_name.lower().strip()

    if timm is not None:
        try:
            return timm.create_model(lowered, pretrained=pretrained, num_classes=num_classes)
        except Exception:
            # If timm cannot create this model, continue to fallback mapping.
            pass

    if lowered == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if lowered == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if lowered == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        "Install timm for wider support, or use one of: resnet18, resnet50, mobilenet_v3_small."
    )
