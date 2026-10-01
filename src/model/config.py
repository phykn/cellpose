from .. import MODEL_STRIDE

DINO_VIT_BACKBONES = (
    "vits16",
    "vits16plus",
    "vitb16",
    "vitl16",
    "vith16plus",
    "vit7b16",
)

MODEL_DEFAULTS = {
    "backbone": "vitb16",
    "backbone_weights": None,
    "patch_stride": MODEL_STRIDE,
    "classes": [],
}


def validate_model_config(model: dict) -> None:
    check_backbone(model.get("backbone", MODEL_DEFAULTS["backbone"]))
    check_patch_stride(model.get("patch_stride", MODEL_DEFAULTS["patch_stride"]))
    classes = model.get("classes", MODEL_DEFAULTS["classes"])
    if not isinstance(classes, list) or any(
        not isinstance(name, str) or not name.strip() for name in classes
    ):
        raise ValueError("model.classes must be a list of non-empty class names.")
    if len(set(classes)) != len(classes):
        raise ValueError("model.classes must not contain duplicate names.")


def check_backbone(backbone: str) -> None:
    if not isinstance(backbone, str):
        raise TypeError("backbone must be a string.")
    if backbone not in DINO_VIT_BACKBONES:
        choices = ", ".join(DINO_VIT_BACKBONES)
        raise ValueError(f"backbone must be one of: {choices}.")


def check_patch_stride(stride: int) -> None:
    if not isinstance(stride, int) or isinstance(stride, bool):
        raise TypeError("patch_stride must be an integer.")
    if stride <= 0:
        raise ValueError("patch_stride must be positive.")
    if stride > 16:
        raise ValueError("patch_stride must not exceed the patch size.")
    if stride % 2:
        raise ValueError("patch_stride must be even to restore the input resolution.")


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
