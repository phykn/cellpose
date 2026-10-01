from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_weights
from .data.dataset import CellposeDataset
from .model.cellpose import CellposeDINO
from .model.config import MODEL_DEFAULTS
from .model.dinov3 import build_dinov3
from .train.checkpoint import load_checkpoint
from .train.loss import CellposeLoss
from .train.trainer import BatchStream, Trainer


def build_model(
    cfg: dict,
    weights: str | Path | None = None,
    load_backbone: bool = True,
    initialize_classifier: bool = False,
) -> CellposeDINO:
    model_cfg = MODEL_DEFAULTS | cfg["model"]
    backbone_weights = model_cfg["backbone_weights"] if load_backbone else None
    encoder = build_dinov3(
        backbone_weights,
        model_name=model_cfg["backbone"],
    )
    model = CellposeDINO(
        encoder,
        patch_stride=model_cfg["patch_stride"],
        num_classes=len(model_cfg["classes"]),
    )
    if weights is not None:
        load_weights(
            weights,
            model,
            initialize_classifier=initialize_classifier,
            model_cfg=model_cfg,
        )
    return model


def build_dataset(cfg: dict) -> CellposeDataset:
    data = cfg["data"]
    for key in ("image_dir", "mask_dir"):
        if data.get(key) is None:
            raise ValueError(f"data.{key} is required for training.")
    return CellposeDataset(
        data["image_dir"],
        data["mask_dir"],
        crop_size=data.get("crop_size", 384),
        channel_axis=data.get("channel_axis"),
        augment=data.get("augment", True),
        label_dir=data.get("label_dir"),
        classes=tuple(cfg["model"].get("classes", [])),
    )


def build_loader(cfg: dict, device: torch.device) -> DataLoader:
    data = cfg["data"]
    num_workers = data.get("num_workers", 0)
    if not isinstance(num_workers, int) or isinstance(num_workers, bool):
        raise TypeError("data.num_workers must be an integer.")
    if num_workers < 0:
        raise ValueError("data.num_workers must be non-negative.")
    batch_size = data.get("batch_size", 1)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("data.batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("data.batch_size must be positive.")
    return DataLoader(
        build_dataset(cfg),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def build_trainer(cfg: dict, device: torch.device) -> Trainer:
    train = cfg["train"]
    resume = train.get("resume")
    weights = train.get("weights")
    if resume is not None and weights is not None:
        raise ValueError("use either resume or weights, not both.")
    loader = build_loader(cfg, device)
    model = build_model(
        cfg,
        weights=weights,
        load_backbone=resume is None and weights is None,
        initialize_classifier=weights is not None,
    ).to(device)
    optim = cfg["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim.get("learning_rate", 1e-5),
        weight_decay=optim.get("weight_decay", 0.1),
    )
    trainer = Trainer(
        model,
        BatchStream(loader),
        optimizer,
        CellposeLoss(train.get("class_loss_weight", 1.0)),
        device,
        mixed_precision=train.get("mixed_precision", True),
        cfg=cfg,
    )
    if resume is not None:
        trainer.step_idx = load_checkpoint(
            resume, model, optimizer, device=device, scaler=trainer.scaler, cfg=cfg
        )
        cfg["optim"]["learning_rate"] = optimizer.param_groups[0]["lr"]
        cfg["optim"]["weight_decay"] = optimizer.param_groups[0]["weight_decay"]
    return trainer
