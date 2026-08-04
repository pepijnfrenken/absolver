"""Recursively convert tensors to JSON-serializable types.

All LangGraph node return values pass through this before being checkpointed.
"""
from __future__ import annotations

import torch


def tensor_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: tensor_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [tensor_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(tensor_safe(v) for v in value)
    return value
