from __future__ import annotations

import inspect
from typing import Any, Callable


class MegatronCoreImportError(RuntimeError):
    pass


def require_megatron_core() -> None:
    try:
        import megatron.core  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on target cluster
        raise MegatronCoreImportError(
            "Megatron-Core is required for this copied project. Install NVIDIA Megatron-LM/"
            "Megatron-Core on the H800 training image before launching train.py."
        ) from exc


def call_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in sig.parameters and value is not None
    }
    return fn(**supported)


def first_tensor(value: Any):
    if hasattr(value, "shape"):
        return value
    if isinstance(value, tuple):
        for item in value:
            if hasattr(item, "shape"):
                return item
    if isinstance(value, dict):
        for item in value.values():
            if hasattr(item, "shape"):
                return item
    raise TypeError(f"Could not find a tensor in {type(value)!r}")
