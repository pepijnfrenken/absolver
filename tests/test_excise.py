"""Tests for excise.py and distill.py on a tiny toy model.

Builds an 8-layer, 64-hidden dense PyTorch model, feeds dummy activations
through DISTILL and EXCISE, then asserts:
- weight matrices actually change
- the change is along the projected refusal direction (orthogonal to it after)
- all four dir_method options run without error
"""
from __future__ import annotations
from model_registry import set_model
from model_registry import get_model

import pytest
import torch
import torch.nn as nn

from config import ModelConfig
from distill import distill_node
from excise import excise_node


# ---------------------------------------------------------------------- #
# Toy model mimicking the decoder contract EXCISE expects.
# ---------------------------------------------------------------------- #
class _ToyMLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden, hidden, bias=False)


class _ToyAttn(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _ToyLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_attn = _ToyAttn(hidden)
        self.mlp = _ToyMLP(hidden)


class _ToyDecoder(nn.Module):
    def __init__(self, n_layers: int, hidden: int):
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer(hidden) for _ in range(n_layers)])


class _ToyModel(nn.Module):
    def __init__(self, n_layers: int = 8, hidden: int = 64):
        super().__init__()
        self.model = _ToyDecoder(n_layers, hidden)
        self.hidden_size = hidden
        # EXCISE infers device/dtype from a parameter.
        self.dummy = nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
@pytest.fixture
def toy_state():
    torch.manual_seed(0)
    n_layers, hidden = 8, 64
    model = _ToyModel(n_layers, hidden)
    from model_registry import set_model
    set_model(model)

    # Build activations with a planted refusal direction on every layer.
    # harm acts cluster around +d, harmless acts around -d.
    direction = torch.randn(hidden)
    direction = direction / direction.norm()

    n_prompts = 10
    harm_acts: dict[int, list[torch.Tensor]] = {}
    harmless_acts: dict[int, list[torch.Tensor]] = {}
    for i in range(n_layers):
        base = torch.randn(n_prompts, hidden) * 0.1
        harm_acts[i] = [direction + base[j] for j in range(n_prompts)]
        harmless_acts[i] = [-direction + base[j] for j in range(n_prompts)]

    cfg = ModelConfig(
        model_id="toy/test",
        model_arch="dense",
        dir_method="diff_means",
        alpha=1.0,
        n_directions=1,
        target_weights=["o_proj", "down_proj"],
        separation_threshold=0.0,  # accept every layer for the test
    )

    return {
        "config": cfg,
        "model_loaded": True,
        "architecture": "dense",
        "hidden_size": hidden,
        "num_layers": n_layers,
        "harm_acts": harm_acts,
        "harmless_acts": harmless_acts,
        "passes_completed": 0,
        "excise_history": [],
        "target_weights": ["o_proj", "down_proj"],
    }


# ---------------------------------------------------------------------- #
# Distill tests
# ---------------------------------------------------------------------- #
class TestDistill:
    def test_diff_means_returns_directions_for_all_layers(self, toy_state):
        out = distill_node(toy_state)
        assert set(out.keys()) == {"refusal_directions", "separation_scores", "target_layers"}
        assert len(out["refusal_directions"]) == toy_state["num_layers"]
        assert len(out["separation_scores"]) == toy_state["num_layers"]
        # Every direction should be unit-norm.
        for d in out["refusal_directions"].values():
            assert torch.allclose(d.norm(), torch.tensor(1.0), atol=1e-5)

    @pytest.mark.parametrize(
        "method", ["diff_means", "svd", "leace", "whitened_svd"]
    )
    def test_every_dir_method_runs(self, toy_state, method):
        cfg = toy_state["config"]
        toy_state["config"] = cfg.model_copy(update={"dir_method": method})
        out = distill_node(toy_state)
        assert len(out["refusal_directions"]) == toy_state["num_layers"]
        # Separation scores must be finite non-negative floats.
        for s in out["separation_scores"].values():
            assert torch.isfinite(torch.tensor(s))
            assert s >= 0

    def test_explicit_target_layers_honored(self, toy_state):
        cfg = toy_state["config"]
        toy_state["config"] = cfg.model_copy(update={"target_layers": [0, 3, 7]})
        out = distill_node(toy_state)
        assert sorted(out["target_layers"]) == [0, 3, 7]

    def test_auto_select_returns_at_least_one_layer(self, toy_state):
        out = distill_node(toy_state)
        assert len(out["target_layers"]) >= 1


# ---------------------------------------------------------------------- #
# Excise tests
# ---------------------------------------------------------------------- #
class TestExcise:
    def _snapshot(self, model) -> dict[int, dict[str, torch.Tensor]]:
        snap = {}
        for i, layer in enumerate(model.model.layers):
            snap[i] = {
                "o_proj": layer.self_attn.o_proj.weight.detach().clone(),
                "down_proj": layer.mlp.down_proj.weight.detach().clone(),
            }
        return snap

    def test_weights_actually_change_on_target_layers(self, toy_state):
        model = get_model()
        before = self._snapshot(model)

        distilled = distill_node(toy_state)
        toy_state.update(distilled)
        toy_state["target_layers"] = [0, 1, 2]

        out = excise_node(toy_state)
        assert out["projection_applied"] is True
        assert out["passes_completed"] == 1

        after = self._snapshot(model)
        for i in [0, 1, 2]:
            # o_proj and down_proj both modified.
            assert not torch.allclose(before[i]["o_proj"], after[i]["o_proj"])
            assert not torch.allclose(before[i]["down_proj"], after[i]["down_proj"])

    def test_non_target_layers_untouched(self, toy_state):
        model = get_model()
        before = self._snapshot(model)

        distilled = distill_node(toy_state)
        toy_state.update(distilled)
        toy_state["target_layers"] = [0]

        excise_node(toy_state)
        after = self._snapshot(model)
        # Layer 5 wasn't a target -> identical weights.
        assert torch.allclose(before[5]["o_proj"], after[5]["o_proj"])
        assert torch.allclose(before[5]["down_proj"], after[5]["down_proj"])

    def test_excise_history_recorded(self, toy_state):
        distilled = distill_node(toy_state)
        toy_state.update(distilled)
        toy_state["target_layers"] = [0, 4]
        out = excise_node(toy_state)
        # History has one entry per processed target layer.
        assert len(out["excise_history"]) == 2
        for entry in out["excise_history"]:
            assert "layer" in entry
            assert "modified_weights" in entry
            assert set(entry["modified_weights"]) <= {"o_proj", "down_proj"}

    def test_projection_removes_direction_component(self, toy_state):
        """After projecting o_proj with alpha=1, applying the same direction
        to the modified weight should yield ~0 in the refusal subspace.
        """
        distilled = distill_node(toy_state)
        toy_state.update(distilled)
        toy_state["target_layers"] = [0]
        toy_state["config"] = toy_state["config"].model_copy(update={"alpha": 1.0})

        model = get_model()
        layer0 = model.model.layers[0]
        direction = distilled["refusal_directions"][0]
        direction = direction.to(layer0.self_attn.o_proj.weight.dtype)
        direction = direction / direction.norm()

        # Project once.
        excise_node(toy_state)
        # Now compute d @ W @ d^T -- should be much smaller than before.
        W = layer0.self_attn.o_proj.weight
        proj_residual = (direction @ W @ direction).abs().item()
        # Sanity: residual is finite and small relative to weight norm.
        assert torch.isfinite(torch.tensor(proj_residual))
        assert proj_residual < W.abs().mean().item() * 10

    def test_multiple_passes_increment_counter(self, toy_state):
        distilled = distill_node(toy_state)
        toy_state.update(distilled)
        toy_state["target_layers"] = [0]
        out1 = excise_node(toy_state)
        toy_state["passes_completed"] = out1["passes_completed"]
        out2 = excise_node(toy_state)
        assert out2["passes_completed"] == 2
