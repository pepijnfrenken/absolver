"""Tests for sweep._resolve_proj / _apply_* coverage fix (huihui geometry).

The LFM2.5 forensics mission (2026-09-02) proved the "conv must never be
projected" rule WRONG: huihui's published abliteration projects ALL 32
out-projections of the hybrid arch — attn out_proj x6 (square), conv
out_proj x10 (square 2D Linear), ffn w2 x16 (the MLP out-projection,
[hidden, inter] — NOT square, output dim == hidden).

These tests pin the resolver + apply contract with stub layers shaped like
the real LFM2.5 blocks (measured from the model's own safetensors):
  - attention block: self_attn.out_proj [8,8], feed_forward.w2 [8,4]
  - conv block:      conv.out_proj [8,8], feed_forward.w2 [8,4]
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sweep import _apply_candidate, _resolve_proj


class _FFN(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        # LFM2.5-style MLP out-projection: weight [hidden, inter] — NOT
        # square. nn.Linear(in, out) stores W[out, in], so in=inter, out=hidden.
        self.w2 = nn.Linear(inter, hidden, bias=False)


class _Attn(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        # LFM2.5 names it out_proj (old harness only matched o_proj).
        self.out_proj = nn.Linear(hidden, hidden, bias=False)


class _ConvBlock(nn.Module):
    def __init__(self, hidden: int, kernel: int = 3):
        super().__init__()
        # LFM2.5 conv path: in_proj -> Conv1d (3D weight) -> out_proj.
        self.in_proj = nn.Linear(hidden, hidden * 3, bias=False)
        self.conv = nn.Conv1d(hidden, hidden, kernel, groups=hidden)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)


class _AttnLayer(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.self_attn = _Attn(hidden)
        self.feed_forward = _FFN(hidden, inter)


class _ConvLayer(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.conv = _ConvBlock(hidden)
        self.feed_forward = _FFN(hidden, inter)


class _ToyLfm(nn.Module):
    """16-layer LFM2.5-shaped hybrid: conv at even idx, attn at odd idx
    (mirrors the real layer_types layout). Nested `model.layers` matches the
    decoder contract sweep._find_layers expects (like test_excise's toy)."""
    def __init__(self, hidden: int = 8, inter: int = 4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            _ConvLayer(hidden, inter) if i % 2 == 0 else _AttnLayer(hidden, inter)
            for i in range(16)
        ])
        self.hidden_size = hidden

    @property
    def layers(self):
        return self.model.layers


@pytest.fixture
def lfm_toy():
    torch.manual_seed(0)
    return _ToyLfm()


def test_resolve_attn_out_proj_aliases(lfm_toy):
    """o_proj resolves LFM2.5's self_attn.out_proj (the naming hole b9d1fa7
    fixed for llama names must stay fixed)."""
    layer = lfm_toy.layers[1]
    mod = _resolve_proj(layer, "o_proj")
    assert mod is layer.self_attn.out_proj


def test_resolve_conv_out_only_when_2d_square(lfm_toy):
    """conv_out resolves conv.out_proj (2D square) but NEVER the 3D Conv1d
    or the in_proj (input-side) — the shape gate from the fix."""
    layer = lfm_toy.layers[0]
    assert _resolve_proj(layer, "conv_out") is layer.conv.out_proj
    # A conv block whose out_proj is NOT 2D square must resolve to None.
    layer2 = _ConvLayer(8, 4)
    layer2.conv.out_proj = nn.Conv1d(8, 8, 3, groups=8)  # 3D weight
    assert _resolve_proj(layer2, "conv_out") is None


def test_resolve_w2_non_square_is_projectable(lfm_toy):
    """w2 resolves feed_forward.w2 EVEN THOUGH it is [hidden, inter] —
    the old square-only assumption would silently drop all 16 ffn
    projections on LFM2.5 (w2 measures [2048, 8192] on the real model)."""
    for li in (0, 1):
        layer = lfm_toy.layers[li]
        w2 = _resolve_proj(layer, "w2")
        assert w2 is layer.feed_forward.w2
        assert tuple(w2.weight.shape) == (8, 4)  # non-square, still resolves
        # ffn_out is the same canonical target
        assert _resolve_proj(layer, "ffn_out") is layer.feed_forward.w2


def test_apply_all32_projects_every_out_projection(lfm_toy):
    """The huihui geometry: mpoa over ALL 16 layers with weights
    [o_proj, conv_out, w2] must touch exactly 32 weights (6 attn out_proj
    + 10 conv out_proj + 16 w2) and change each one."""
    hidden = 8
    dirs = {i: torch.randn(hidden) / hidden for i in range(16)}
    candidate = {"method": "mpoa", "dir_method": "paired",
                 "target_layers": list(range(16)),
                 "target_weights": ["o_proj", "conv_out", "w2"],
                 "alpha": 1.0, "passes": 1}

    # snapshot the 32 target weights (module refs kept for post-check)
    snapshots = {}
    for li, layer in enumerate(lfm_toy.layers):
        for wname, mod in (("o_proj", getattr(getattr(layer, "self_attn", None), "out_proj", None)),
                           ("conv_out", getattr(getattr(layer, "conv", None), "out_proj", None)),
                           ("w2", layer.feed_forward.w2)):
            if mod is not None:
                snapshots[(li, wname)] = (mod, mod.weight.detach().clone())

    _apply_candidate(lfm_toy, dirs, None, candidate)

    assert len(snapshots) == 32
    applied = candidate["_applied"]
    assert len(applied) == 32
    # every recorded target actually changed (keys are (layer, weight) —
    # a per-layer dict would collapse conv_out and w2 onto one slot)
    changed = {(a["layer"], a["weight"]) for a in applied}
    for (li, wname), (mod, w0) in snapshots.items():
        assert (li, wname) in changed, f"layer {li} {wname} was not applied"
        w = mod.weight
        rel = float((w0.detach() - w.detach()).norm()) / float(w0.norm())
        assert rel > 1e-4, f"layer {li} {wname} weight unchanged (rel={rel})"
    # untouched tensors stay untouched: conv_out resolves ONLY to out_proj
    # (never the Conv1d / in_proj), so their weights were never in scope.
    assert _resolve_proj(lfm_toy.layers[0], "conv_out") is lfm_toy.layers[0].conv.out_proj
    assert _resolve_proj(lfm_toy.layers[0], "conv_out") is not lfm_toy.layers[0].conv.conv


def test_zero_match_candidate_is_loud_not_silent(lfm_toy, caplog):
    """A candidate whose weight names resolve nowhere must be a loud no-op
    warning (never print 'Applied' with byte-identical weights)."""
    import logging
    dirs = {i: torch.randn(8) for i in range(16)}
    candidate = {"method": "mpoa", "dir_method": "paired",
                 "target_layers": list(range(16)),
                 "target_weights": ["bogus_proj"], "alpha": 1.0, "passes": 1}
    with caplog.at_level(logging.WARNING, logger="sweep"):
        _apply_candidate(lfm_toy, dirs, None, candidate)
    assert not candidate.get("_applied")
    assert any("projected ZERO weights" in r.message for r in caplog.records)
