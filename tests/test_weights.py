import pytest

from src.checkpoint import find_prediction_config
from src.config import save_yaml


def test_prediction_uses_weights_sidecar_or_legacy_run_config(tmp_path):
    weights = tmp_path / "model.pt"
    with pytest.raises(FileNotFoundError, match="--config"):
        find_prediction_config(weights)
    legacy = tmp_path / "train.yaml"
    save_yaml(legacy, {})
    assert find_prediction_config(weights) == legacy
    sidecar = tmp_path / "model.yaml"
    save_yaml(sidecar, {})
    assert find_prediction_config(weights) == sidecar
