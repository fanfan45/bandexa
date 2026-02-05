# examples/_resnet_backbone.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Final

import torch
from torch import nn

# --- policy knobs (edit here, no CLI) ---
PRETRAINED: bool = True           # edit here
# FINE_TUNE is a set of layer names:
#   {"none"}                 -> freeze everything
#   {"all"}                  -> unfreeze everything
#   {"layer4"}               -> unfreeze only layer4 (valid singleton)
#   {"layer2", "layer4"}     -> unfreeze only those layers
FINE_TUNE: set[str] = {"layer4"}  # edit here
# Canonical ResNet layer names you allow for partial fine-tuning
_ADMISSIBLE: Final[set[str]] = {"layer1", "layer2", "layer3", "layer4"}
# ---

@dataclass(frozen=True)
class ResNetBackboneConfig:
    arch: str = "resnet18"          # "resnet18" | "resnet34" | "resnet50" | ...
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32


def build_resnet_backbone(cfg: ResNetBackboneConfig) -> nn.Module:
    """
    Build a torchvision ResNet backbone that outputs a pooled feature vector.

    - Uses torchvision's new-style `weights=...` API.
    - Only selects the constructor/weights for the requested `arch`.
    - Returns a module that outputs shape (B, C) where C depends on the arch.

    Trainability behavior
    ---------------------
    We intentionally make the backbone manage its own trainability, so callers (examples/policies)
    don't need to pass a "trainable" flag around.
    """
    import torchvision.models as models  # single import

    arch = cfg.arch.lower().strip()

    constructors = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }

    weights_default = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "resnet101": models.ResNet101_Weights.DEFAULT,
        "resnet152": models.ResNet152_Weights.DEFAULT,
    }

    if arch not in constructors:
        raise ValueError(f"Unsupported arch '{cfg.arch}'. Choose one of: {sorted(constructors.keys())}")

    weights = weights_default[arch] if PRETRAINED else None
    model = constructors[arch](weights=weights)

    # Safety: random init + mostly frozen is usually a bad combo
    if not PRETRAINED and FINE_TUNE != "all":
        raise ValueError(
            "PRETRAINED=False with FINE_TUNE != 'all' is usually ineffective. "
            "Set FINE_TUNE='all' or set PRETRAINED=True."
    )

    # -------------------------
    # Trainability policy
    # -------------------------

    # Freeze everything by default
    for p in model.parameters():
        p.requires_grad = False

    # Empty set is almost certainly a mistake (choose {"none"} instead).
    if not FINE_TUNE:
        raise ValueError("FINE_TUNE must be non-empty. Use {'none'}, {'all'}, or a subset of admissible layers.")

    # Special singletons first
    if FINE_TUNE == {"none"}:
        pass  # keep everything frozen
    elif FINE_TUNE == {"all"}:
        for p in model.parameters():
            p.requires_grad = True
    else:
        # Any other set (including singleton {"layer4"}) is treated as a subset selection.
        unknown = FINE_TUNE - _ADMISSIBLE
        if unknown:
            raise ValueError(f"Unknown layer(s) in FINE_TUNE: {sorted(unknown)}. Allowed: {sorted(_ADMISSIBLE)}")
        for name in FINE_TUNE:
            layer = getattr(model, name)  # e.g. model.layer2
            for p in layer.parameters():
                p.requires_grad = True

    # Drop the classification head; keep everything up to avgpool
    backbone = nn.Sequential(*list(model.children())[:-1])  # ends with (B, C, 1, 1)

    # Wrap to flatten output to (B, C)
    class _Backbone(nn.Module):
        def __init__(self, net: nn.Module) -> None:
            super().__init__()
            self.net = net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.net(x)
            return y.flatten(1)

    out = _Backbone(backbone)

    # Move to device/dtype
    device = cfg.device if cfg.device is not None else torch.device("cpu")
    out.to(device=device, dtype=cfg.dtype)

    return out
