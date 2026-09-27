import pytest
import torch
from torch import nn

from src import build
from src.checkpoint import load_weights, save_weights
from src.config import save_yaml


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_features = 4
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 4, 16, stride=16)

    def forward_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.patch_embed.proj(image)
        return {"x_norm_patchtokens": features.flatten(2).transpose(1, 2)}


def test_build_model_passes_backbone_options(monkeypatch, tmp_path) -> None:
    requested = []
    monkeypatch.setattr(
        build,
        "build_dinov3",
        lambda weights, model_name: (
            requested.append((weights, model_name)) or FakeEncoder()
        ),
    )
    cfg = {
        "model": {
            "backbone": "vitl16",
            "backbone_weights": str(tmp_path / "backbone.pt"),
            "patch_stride": 8,
        }
    }

    model = build.build_model(cfg)

    assert requested == [(str(tmp_path / "backbone.pt"), "vitl16")]
    assert model.patch_stride == 8


@pytest.mark.parametrize("saved", [{}, {"model": {}}])
def test_weights_accept_default_only_saved_configuration(monkeypatch, tmp_path, saved):
    monkeypatch.setattr(build, "build_dinov3", lambda *a, **kw: FakeEncoder())
    cfg = {"model": {"backbone": "vitb16", "patch_stride": 8, "classes": []}}
    original = build.build_model(cfg, load_backbone=False)
    path = tmp_path / "model.pt"
    save_weights(path, original)
    save_yaml(path.with_suffix(".yaml"), saved)

    restored = build.build_model(cfg, weights=path, load_backbone=False)
    image = torch.randn(1, 3, 24, 32)
    torch.testing.assert_close(restored(image), original(image), rtol=0, atol=0)


def test_weight_loading_checks_model_configuration_before_mutation(tmp_path):
    model = nn.Linear(2, 1)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / "model.pt"
    save_weights(path, nn.Linear(2, 1))
    save_yaml(path.with_suffix(".yaml"), {"model": {"backbone": "vitl16"}})
    with pytest.raises(ValueError, match="model.backbone"):
        load_weights(path, model, model_cfg={"backbone": "vitb16"})
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


@pytest.mark.parametrize("key", ["image_dir", "mask_dir"])
def test_training_reports_missing_data_paths(key):
    data = {"image_dir": "images", "mask_dir": "masks", key: None}
    with pytest.raises(ValueError, match=f"data.{key} is required for training"):
        build.build_dataset({"data": data, "model": {}})
