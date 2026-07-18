"""Tests for architecture detection (detector.detect_architecture).

Uses lightweight mock modules that mimic the attribute shapes of real HF
dense / MoE / diffusion-encoder models, so no network or GPU is required.
"""
from __future__ import annotations

import math
from typing import Optional

import pytest
import torch
import torch.nn as nn

from detector import UnsupportedArchitecture, detect_architecture


# ---------------------------------------------------------------------- #
# Mock building blocks
# ---------------------------------------------------------------------- #
class _Linear(nn.Module):
    """Minimal nn.Module stand-in for nn.Linear with only ``weight``.

    Must be an nn.Module so detector's ``named_modules()`` walk surfaces the
    ``o_proj``/``down_proj`` basenames.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)


class _Attention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.o_proj = _Linear(hidden, hidden)


class _MLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.down_proj = _Linear(hidden, hidden)


class _DenseLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_attn = _Attention(hidden)
        self.mlp = _MLP(hidden)


class _ExpertMLP(nn.Module):
    """One expert's MLP, exposing a down_proj like real MoE impls."""

    def __init__(self, hidden: int):
        super().__init__()
        self.down_proj = _Linear(hidden, hidden)


class _Experts(nn.Module):
    """Per-expert module bank; wraps an ``nn.ModuleList`` so ``len(experts)``
    works (detector's _count_experts uses ``len()``).
    """

    def __init__(self, n_experts: int, hidden: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [_ExpertMLP(hidden) for _ in range(n_experts)]
        )
        self.num_experts = n_experts

    def __len__(self):
        return len(self.experts)


class _MoELayer(nn.Module):
    def __init__(self, hidden: int, n_experts: int):
        super().__init__()
        self.self_attn = _Attention(hidden)
        self.block_sparse_moe = nn.Module()  # marker attribute
        self.experts = _Experts(n_experts, hidden)


class _Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _CausalLM(nn.Module):
    def __init__(self, decoder: _Decoder, hidden: int):
        super().__init__()
        self.model = decoder
        self.hidden_size = hidden
        self.config = type("Cfg", (), {"hidden_size": hidden})()


# ---------------------------------------------------------------------- #
# Helpers to build each arch
# ---------------------------------------------------------------------- #
def _make_dense(num_layers: int = 4, hidden: int = 32) -> _CausalLM:
    decoder = _Decoder([_DenseLayer(hidden) for _ in range(num_layers)])
    return _CausalLM(decoder, hidden)


def _make_moe(num_layers: int = 4, hidden: int = 32, n_experts: int = 8) -> _CausalLM:
    decoder = _Decoder([_MoELayer(hidden, n_experts) for _ in range(num_layers)])
    return _CausalLM(decoder, hidden)


def _make_diffusion(num_layers: int = 6, hidden: int = 64) -> _CausalLM:
    """A diffusion-pipeline mock whose text encoder is at ``text_encoder``."""
    text_encoder = _CausalLM(
        _Decoder([_DenseLayer(hidden) for _ in range(num_layers)]), hidden
    )

    class _DiffusionPipe(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = text_encoder

    pipe = _DiffusionPipe()
    # detect_architecture is called with the text encoder directly, per the
    # SUMMON node's contract (diffusers -> pipe.text_encoder).
    return pipe, text_encoder


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #
class TestDetectDense:
    def test_dense_basic_fields(self):
        model = _make_dense(num_layers=4, hidden=32)
        info = detect_architecture(model)
        assert info["architecture"] == "dense"
        assert info["num_layers"] == 4
        assert info["hidden_size"] == 32
        assert info["num_experts"] is None
        assert info["has_o_proj"] is True
        assert info["has_down_proj"] is True
        assert info["has_experts"] is False
        assert info["text_encoder_model"] is None

    def test_dense_target_weights_autodetected(self):
        model = _make_dense(num_layers=2, hidden=16)
        info = detect_architecture(model)
        assert "o_proj" in info["target_weights"]
        assert "down_proj" in info["target_weights"]
        assert "expert.down" not in info["target_weights"]

    def test_dense_single_layer(self):
        model = _make_dense(num_layers=1, hidden=8)
        info = detect_architecture(model)
        assert info["architecture"] == "dense"
        assert info["num_layers"] == 1


class TestDetectMoE:
    def test_moe_via_block_sparse_moe(self):
        model = _make_moe(num_layers=3, hidden=32, n_experts=8)
        info = detect_architecture(model)
        assert info["architecture"] == "moe"
        assert info["num_experts"] == 8
        assert info["has_experts"] is True
        assert info["has_o_proj"] is True
        assert "expert.down" in info["target_weights"]

    def test_moe_via_experts_attr(self):
        """MoE without block_sparse_moe but with .experts should still be detected."""

        model = _make_moe(num_layers=2, hidden=16, n_experts=4)
        # Strip the marker; .experts alone should trigger MoE.
        del model.model.layers[0].block_sparse_moe
        info = detect_architecture(model)
        assert info["architecture"] == "moe"
        assert info["num_experts"] == 4


class TestDetectDiffusion:
    def test_diffusion_text_encoder(self):
        _pipe, text_encoder = _make_diffusion(num_layers=6, hidden=64)
        info = detect_architecture(text_encoder)
        assert info["architecture"] in ("dense", "diffusion_encoder")
        # The text encoder is a standard decoder stack; detection via
        # model.model.layers is valid and yields expected shape info.
        assert info["num_layers"] == 6
        assert info["hidden_size"] == 64


class TestUnsupported:
    def test_raises_on_empty_model(self):
        class Empty(nn.Module):
            pass

        with pytest.raises(UnsupportedArchitecture):
            detect_architecture(Empty())

    def test_error_message_contains_debug_info(self):
        class Weird(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Module()  # no .layers anywhere

        with pytest.raises(UnsupportedArchitecture) as excinfo:
            detect_architecture(Weird())
        msg = str(excinfo.value).lower()
        # Debug info should mention what WAS found.
        assert "transformer" in msg or "found" in msg or "module" in msg
