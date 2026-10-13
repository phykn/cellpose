import pytest

from src.config import PROJECT_ROOT, load_config, save_yaml


def test_config_resolves_paths_and_preserves_explicit_precedence(tmp_path, monkeypatch):
    path = tmp_path / "train.yaml"
    save_yaml(path, {"device": "cpu", "data": {"image_dir": "data/images"}})
    monkeypatch.chdir(tmp_path)
    cfg = load_config(path)
    assert cfg["device"] == "cpu"
    assert cfg["data"]["image_dir"] == str(PROJECT_ROOT / "data" / "images")
    assert cfg["data"]["crop_size"] == 384
    assert cfg["model"]["patch_stride"] == 8
    save_yaml(path, {"device": "cuda", "train": {"resume": "old.pt"}})
    cfg = load_config(path, device="cpu", resume="new.pt")
    assert cfg["device"] == "cpu"
    assert cfg["train"]["resume"] == str(tmp_path / "new.pt")


@pytest.mark.parametrize("key, other", [("resume", "weights"), ("weights", "resume")])
def test_cli_checkpoint_paths_use_working_directory(tmp_path, monkeypatch, key, other):
    path = tmp_path / "train.yaml"
    save_yaml(path, {"train": {other: "saved.pt"}})
    monkeypatch.chdir(tmp_path)

    cfg = load_config(path, device="cpu", **{key: "selected.pt"})

    assert cfg["train"][key] == str(tmp_path / "selected.pt")
    assert cfg["train"][other] is None


@pytest.mark.parametrize("key", ["resume", "weights"])
def test_yaml_checkpoint_paths_use_project_root(tmp_path, monkeypatch, key):
    path = tmp_path / "train.yaml"
    save_yaml(path, {"train": {key: "saved.pt"}})
    monkeypatch.chdir(tmp_path)

    cfg = load_config(path, device="cpu")

    assert cfg["train"][key] == str(PROJECT_ROOT / "saved.pt")


def test_cli_resume_and_weights_remain_mutually_exclusive(tmp_path):
    path = tmp_path / "train.yaml"
    save_yaml(path, {})
    with pytest.raises(ValueError, match="either train.resume or train.weights"):
        load_config(path, device="cpu", resume="resume.pt", weights="weights.pt")


@pytest.mark.parametrize(
    "cfg",
    [
        {"model": {"backbone": "convnext_tiny"}},
        {"model": {"patch_stride": 3}},
        {"model": {"patch_stride": 16}, "data": {"crop_size": 24}},
        {"train": {"mixed_precision": "false"}},
        {"data": {"batch_size": 0}},
    ],
)
def test_invalid_configuration_fails_before_building(cfg, tmp_path):
    path = tmp_path / "train.yaml"
    save_yaml(path, cfg)
    with pytest.raises((TypeError, ValueError)):
        load_config(path, device="cpu")
