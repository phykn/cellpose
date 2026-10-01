import os
from pathlib import Path

import torch
from torch import nn

from .config import load_yaml
from .model.config import check_model_config


def save_weights(path: str | Path, model: nn.Module) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temp)
    os.replace(temp, path)


def load_weights(
    path: str | Path,
    model: nn.Module,
    device: torch.device | str = "cpu",
    initialize_classifier: bool = False,
    classes: list[str] | None = None,
    model_cfg: dict | None = None,
) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model weights do not exist: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"model weights must contain a state dict: {path}")
    has_classifier = any(key.startswith("class_output.") for key in state)
    if model_cfg is not None or (classes is not None and has_classifier):
        try:
            config_path = find_prediction_config(path)
        except FileNotFoundError:
            if has_classifier:
                raise
            if (
                model_cfg is not None
                and model_cfg.get("classes")
                and not initialize_classifier
            ):
                raise ValueError(
                    "classification weights require their saved model.yaml."
                ) from None
        else:
            saved = load_yaml(config_path).get("model", {})
            if saved.get("classes") and not has_classifier:
                raise ValueError(
                    "model weights are missing classification head parameters."
                )
            if model_cfg is not None:
                check_model_config(
                    model_cfg,
                    saved,
                    initialize_classifier=initialize_classifier and not has_classifier,
                )
            if classes is not None and has_classifier:
                check_model_config(
                    {"classes": classes}, {"classes": saved.get("classes", [])}
                )
    if (
        initialize_classifier
        and getattr(model, "num_classes", 0)
        and not has_classifier
    ):
        # Only the newly added classifier may be absent. Still load everything strictly.
        state = dict(state)
        state.update(
            {
                key: value
                for key, value in model.state_dict().items()
                if key.startswith("class_output.")
            }
        )
    model.load_state_dict(state, strict=True)


def find_prediction_config(weights: str | Path) -> Path:
    weights = Path(weights)
    for path in (weights.with_suffix(".yaml"), weights.parent / "train.yaml"):
        if path.is_file():
            return path
    raise FileNotFoundError("weights have no saved configuration; pass --config.")
