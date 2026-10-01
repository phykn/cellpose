from .. import MODEL_STRIDE

MODEL_DEFAULTS = {
    "backbone": "vitb16",
    "backbone_weights": None,
    "patch_stride": MODEL_STRIDE,
    "classes": [],
}


def check_model_config(
    model: dict, saved: dict, initialize_classifier: bool = False
) -> None:
    classes = model.get("classes", MODEL_DEFAULTS["classes"])
    saved_classes = saved.get("classes", MODEL_DEFAULTS["classes"])
    if classes != saved_classes and not (initialize_classifier and not saved_classes):
        raise ValueError("model.classes differs from the saved class names/order.")
    for key in ("backbone", "patch_stride"):
        if model.get(key, MODEL_DEFAULTS[key]) != saved.get(key, MODEL_DEFAULTS[key]):
            raise ValueError(f"model.{key} differs from the saved configuration.")


def prediction_config(cfg: dict) -> dict:
    return {
        "format_version": 2 if cfg["model"].get("classes") else 1,
        "model": {**cfg["model"], "backbone_weights": None},
        "data": {
            "channel_axis": cfg["data"]["channel_axis"],
            "crop_size": cfg["data"]["crop_size"],
        },
        "predict": cfg["predict"],
    }
