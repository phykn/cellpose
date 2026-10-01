from importlib import import_module
from pathlib import Path

import torch
from torch import nn

from .config import MODEL_DEFAULTS, check_backbone


def build_dinov3(
    weights: str | Path | None = None,
    model_name: str = MODEL_DEFAULTS["backbone"],
) -> nn.Module:
    check_backbone(model_name)

    try:
        backbones = import_module("dinov3.hub.backbones")
    except ImportError as exc:
        raise ImportError(
            "DINOv3 is required. Install requirements.txt after accepting "
            "the DINOv3 license."
        ) from exc

    builder_name = f"dinov3_{model_name}"
    try:
        builder = getattr(backbones, builder_name)
    except AttributeError as exc:
        raise ImportError(f"installed DINOv3 does not expose {builder_name}.") from exc

    if weights is None:
        return builder(pretrained=False)

    path = Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DINOv3 backbone weights do not exist: {path}")
    # The hub loader caches by basename, so different local files with the same
    # name can otherwise silently load an earlier cached checkpoint.
    encoder = builder(pretrained=False)
    encoder.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=True
    )
    return encoder
