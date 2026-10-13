import json
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.checkpoint import load_weights
from src.config import load_config, load_yaml, save_yaml
from src.train.trainer import Trainer


def make_trainer(loader=None) -> tuple[Trainer, nn.Module, torch.optim.Optimizer]:
    model = nn.Conv2d(3, 3, 1)
    if loader is None:
        images = torch.randn(2, 3, 8, 8)
        targets = torch.zeros_like(images)
        loader = DataLoader(TensorDataset(images, targets), batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    trainer = Trainer(
        model,
        loader,
        optimizer,
        nn.MSELoss(),
        torch.device("cpu"),
        mixed_precision=False,
    )
    return trainer, model, optimizer


def test_trainer_runs_and_round_trips_checkpoint(tmp_path) -> None:
    trainer, model, _ = make_trainer()

    trainer.fit(steps=2, save_every=1, run_dir=tmp_path)

    assert trainer.step_idx == 2
    assert (tmp_path / "model.pt").is_file()
    assert (tmp_path / "checkpoint.pt").is_file()

    restored_trainer, restored_model, _ = make_trainer()
    load_weights(tmp_path / "model.pt", restored_model)
    for actual, expected in zip(
        restored_model.parameters(),
        model.parameters(),
        strict=True,
    ):
        assert torch.equal(actual, expected)

    restored_trainer.restore(tmp_path / "checkpoint.pt")
    assert restored_trainer.step_idx == 2
    restored_trainer.fit(steps=3, save_every=1, run_dir=tmp_path / "resumed")
    assert restored_trainer.step_idx == 3


def test_restore_keeps_effective_optimizer_settings_in_saved_configuration(tmp_path):
    config = tmp_path / "source.yaml"
    save_yaml(config, {"device": "cpu"})
    trainer, _, optimizer = make_trainer()
    trainer.cfg = load_config(config)
    trainer.cfg["optim"] = {
        "learning_rate": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
    }
    first, second = tmp_path / "first", tmp_path / "second"
    trainer.fit(steps=1, save_every=1, run_dir=first)

    restored, _, restored_optimizer = make_trainer()
    restored.cfg = load_config(config)
    restored_optimizer.param_groups[0].update(lr=0.5, weight_decay=0.3)
    restored.restore(first / "checkpoint.pt")
    assert restored.cfg["optim"] == trainer.cfg["optim"]
    restored.fit(steps=2, save_every=1, run_dir=second)
    state = torch.load(second / "checkpoint.pt", weights_only=True)
    assert state["step"] == 2
    assert state["config"]["optim"] == trainer.cfg["optim"]
    assert load_yaml(second / "model.yaml") == load_yaml(first / "model.yaml")


def test_completed_resume_still_exports_weights(tmp_path):
    trainer, _, _ = make_trainer()
    trainer.step_idx = 2
    trainer.fit(steps=2, save_every=1, run_dir=tmp_path)
    assert (tmp_path / "model.pt").is_file()
    assert (tmp_path / "checkpoint.pt").is_file()


def test_batches_start_on_first_step_and_repeat_across_epochs(tmp_path, monkeypatch):
    starts = []
    original_iter = DataLoader.__iter__

    def record_iter(loader):
        starts.append(True)
        return original_iter(loader)

    monkeypatch.setattr(DataLoader, "__iter__", record_iter)
    trainer, _, _ = make_trainer()
    assert starts == []

    trainer.fit(steps=5, save_every=5, run_dir=tmp_path)

    assert len(starts) == 3
    assert trainer.step_idx == 5


def test_completed_resume_does_not_start_loader(tmp_path, monkeypatch):
    def unexpected_iter(loader):
        raise AssertionError("completed training must not start data workers")

    monkeypatch.setattr(DataLoader, "__iter__", unexpected_iter)
    trainer, _, _ = make_trainer()
    trainer.step_idx = 2
    trainer.fit(steps=2, save_every=1, run_dir=tmp_path)
    assert (tmp_path / "model.pt").is_file()


def test_restore_starts_a_fresh_data_pass(tmp_path):
    images = torch.stack((torch.zeros(3, 8, 8), torch.ones(3, 8, 8)))
    loader = DataLoader(TensorDataset(images, torch.zeros_like(images)), batch_size=1)
    trainer, model, _ = make_trainer(loader)
    seen = []
    model.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    trainer.fit(steps=1, save_every=1, run_dir=tmp_path)

    trainer.restore(tmp_path / "checkpoint.pt")
    trainer.fit(steps=2, save_every=1, run_dir=tmp_path / "resumed")

    assert len(seen) == 2
    torch.testing.assert_close(seen[0], images[:1])
    torch.testing.assert_close(seen[1], images[:1])


def test_training_matches_manual_updates_across_epochs(tmp_path):
    torch.manual_seed(23)
    images = torch.randn(2, 3, 8, 8)
    targets = torch.randn_like(images)
    loader = DataLoader(TensorDataset(images, targets), batch_size=1)
    trainer, model, _ = make_trainer(loader)
    expected = deepcopy(model)
    optimizer = torch.optim.AdamW(expected.parameters(), lr=1e-2)
    losses = []
    for step in range(5):
        idx = step % len(images)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(
            expected(images[idx : idx + 1]), targets[idx : idx + 1]
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    trainer.fit(steps=5, save_every=5, run_dir=tmp_path)

    for actual, reference in zip(
        model.parameters(), expected.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    metrics = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert metrics == [
        {"step": step, "loss": loss} for step, loss in enumerate(losses, 1)
    ]
