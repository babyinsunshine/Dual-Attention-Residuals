from __future__ import annotations

from .config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "build_model", "load_config"]


def build_model(*args, **kwargs):
    from .model import build_model as _build_model

    return _build_model(*args, **kwargs)
