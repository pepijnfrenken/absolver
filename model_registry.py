"""Re-load-proof model cache — survives importlib.reload between LangGraph nodes.

LangGraph reloads modules between nodes, killing standard module-level globals.
This module uses setattr on the model_registry module in sys.modules to survive.
"""
from __future__ import annotations

import sys
from typing import Any


def _self() -> Any:
    return sys.modules[__name__]


def set_model(model: Any) -> None:
    object.__setattr__(_self(), "_model", model)


def get_model() -> Any:
    return getattr(_self(), "_model", None)


def set_tokenizer(tokenizer: Any) -> None:
    object.__setattr__(_self(), "_tokenizer", tokenizer)


def get_tokenizer() -> Any:
    return getattr(_self(), "_tokenizer", None)
