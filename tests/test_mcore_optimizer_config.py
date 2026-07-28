from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import train


def test_megatron_optimizer_creates_gloo_groups_for_checkpoint_state() -> None:
    captured: dict[str, object] = {}

    class FakeOptimizerConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    def fake_get_megatron_optimizer(
        *, config, model_chunks, use_gloo_process_groups: bool = True
    ):
        captured.update(
            config=config,
            model_chunks=model_chunks,
            use_gloo_process_groups=use_gloo_process_groups,
        )
        return SimpleNamespace()

    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    optimizer_module = ModuleType("megatron.core.optimizer")
    optimizer_config_module = ModuleType("megatron.core.optimizer.optimizer_config")
    optimizer_module.get_megatron_optimizer = fake_get_megatron_optimizer
    optimizer_config_module.OptimizerConfig = FakeOptimizerConfig
    megatron.core = core
    core.optimizer = optimizer_module

    train_cfg = SimpleNamespace(
        bf16=True,
        fp16=False,
        lr=3.0e-4,
        min_lr=3.0e-5,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1.0e-8,
        clip_grad=1.0,
        use_distributed_optimizer=True,
    )
    cfg = SimpleNamespace(train=train_cfg)
    fake_modules = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.optimizer": optimizer_module,
        "megatron.core.optimizer.optimizer_config": optimizer_config_module,
    }

    with mock.patch.dict(sys.modules, fake_modules):
        train.build_megatron_optimizer(SimpleNamespace(), cfg)

    assert captured["use_gloo_process_groups"] is True
