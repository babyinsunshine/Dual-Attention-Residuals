from __future__ import annotations

import atexit
import warnings


_SENTENCEPIECE_SWIG_DEPRECATIONS = (
    "builtin type SwigPyPacked has no __module__ attribute",
    "builtin type SwigPyObject has no __module__ attribute",
    "builtin type swigvarlink has no __module__ attribute",
)
_ORIGINAL_SHOWWARNING = warnings.showwarning


def _filter_sentencepiece_swig_deprecations() -> None:
    for message in _SENTENCEPIECE_SWIG_DEPRECATIONS:
        warnings.filterwarnings(
            "ignore",
            message=f"^{message}$",
            category=DeprecationWarning,
        )


def _install_sentencepiece_shutdown_filter() -> None:
    _filter_sentencepiece_swig_deprecations()

    def showwarning(message: Warning, category: type[Warning], *args: object, **kwargs: object) -> None:
        if issubclass(category, DeprecationWarning) and str(message) in _SENTENCEPIECE_SWIG_DEPRECATIONS:
            return
        _ORIGINAL_SHOWWARNING(message, category, *args, **kwargs)

    warnings.showwarning = showwarning


_filter_sentencepiece_swig_deprecations()
# SentencePiece emits one SWIG warning while the extension is shutting down,
# after pytest has restored its warning state. Reinstall the same exact filters
# at interpreter exit without changing production warning policy.
atexit.register(_install_sentencepiece_shutdown_filter)


def pytest_configure(config: object) -> None:
    """Restore filters after pytest's initial-conftest warning context exits."""
    _filter_sentencepiece_swig_deprecations()


def pytest_unconfigure(config: object) -> None:
    """Keep the exact filters active between pytest teardown and Python exit."""
    _filter_sentencepiece_swig_deprecations()
