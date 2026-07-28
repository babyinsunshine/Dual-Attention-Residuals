from __future__ import annotations

import torch

import train


class _StateModule(torch.nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.events = events

    def state_dict(self, *args, **kwargs):
        self.events.append("state_dict")
        return super().state_dict(*args, **kwargs)


class _OverlapDDP:
    use_forward_hook = True

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def disable_forward_pre_hook(self, param_sync: bool = True) -> None:
        assert param_sync
        self.events.append("disable_and_sync")

    def enable_forward_pre_hook(self) -> None:
        self.events.append("enable")


class _Proxy:
    def __init__(self, module: torch.nn.Module, ddp: _OverlapDDP) -> None:
        self.module = module
        self.ddp = ddp


class _Optimizer:
    def state_dict(self):
        return {}


class _DistributedOptimizer(_Optimizer):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def save_parameter_state(self, filename: str) -> None:
        self.events.append(f"optimizer_parameters:{filename}")

    def load_state_dict(self, state) -> None:
        del state
        self.events.append("optimizer_state")

    def load_parameter_state(self, filename: str) -> None:
        self.events.append(f"load_optimizer_parameters:{filename}")


def test_checkpoint_materializes_overlap_parameters_before_reading_state(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []
    model = _Proxy(_StateModule(events), _OverlapDDP(events))

    def record_save(payload, path) -> None:
        del payload, path
        events.append("save")

    monkeypatch.setattr(torch, "save", record_save)

    train.maybe_save_checkpoint(tmp_path, model, _Optimizer(), None, step=10)

    assert events == ["disable_and_sync", "state_dict", "save", "enable"]


def test_checkpoint_saves_distributed_optimizer_parameter_state(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []
    model = _Proxy(_StateModule(events), _OverlapDDP(events))

    monkeypatch.setattr(torch, "save", lambda payload, path: None)

    train.maybe_save_checkpoint(
        tmp_path,
        model,
        _DistributedOptimizer(events),
        None,
        step=10,
    )

    expected = tmp_path / "iter_0000010" / "optimizer_parameter_state_rank_00000.pt"
    assert f"optimizer_parameters:{expected}" in events


def test_checkpoint_loads_distributed_optimizer_parameter_state(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []
    module = _StateModule(events)
    optimizer = _DistributedOptimizer(events)
    checkpoint_file = tmp_path / "iter_0000010" / "rank_00000.pt"
    checkpoint_file.parent.mkdir()
    checkpoint_file.touch()
    payload = {
        "model": module.state_dict(),
        "optimizer": {},
        "lr_scheduler": None,
        "step": 10,
        "rank": 0,
        "world_size": 1,
        "rng_state": None,
        "cuda_rng_state": None,
    }
    events.clear()
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(train, "cleanup_after_checkpoint_load", lambda: None)

    loaded_step = train.load_checkpoint(
        str(checkpoint_file),
        module,
        optimizer,
        None,
    )

    expected = checkpoint_file.parent / "optimizer_parameter_state_rank_00000.pt"
    assert loaded_step == 10
    assert f"load_optimizer_parameters:{expected}" in events
