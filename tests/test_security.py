import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from src.checkpoint import load_weights
from src.config import load_yaml
from src.data.image import read_array
from src.model.dinov3 import build_dinov3
from src.train.checkpoint import load_checkpoint


def write_marker(path):
    path.write_text("unpickled", encoding="utf-8")
    return {}


class PicklePayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return write_marker, (self.marker,)


@pytest.mark.parametrize("loader", ["weights", "checkpoint", "backbone"])
def test_weight_loaders_reject_executable_pickle(tmp_path, monkeypatch, loader):
    marker = tmp_path / "executed.txt"
    path = tmp_path / "model.pt"
    torch.save(PicklePayload(marker), path)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    monkeypatch.setattr(
        "src.model.dinov3.import_module",
        lambda _: SimpleNamespace(dinov3_vitb16=lambda pretrained: nn.Linear(2, 1)),
    )

    with pytest.raises(pickle.UnpicklingError):
        if loader == "weights":
            load_weights(path, model)
        elif loader == "checkpoint":
            load_checkpoint(path, model, optimizer)
        else:
            build_dinov3(path)
    assert not marker.exists()


def test_numpy_images_reject_executable_pickle(tmp_path):
    marker = tmp_path / "executed.txt"
    path = tmp_path / "image.npy"
    np.save(path, np.array([PicklePayload(marker)], dtype=object))

    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        read_array(path)
    assert not marker.exists()


def test_yaml_configuration_rejects_python_objects(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("!!python/object/apply:builtins.eval ['1 + 1']", encoding="utf-8")

    with pytest.raises(yaml.constructor.ConstructorError):
        load_yaml(path)
