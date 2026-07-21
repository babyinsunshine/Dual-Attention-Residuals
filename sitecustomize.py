from __future__ import annotations

import typing

if not hasattr(typing, "override"):
    def _override(func):
        return func

    typing.override = _override
