"""Graph-level and pipeline-safety tests for the P0/P1 audit fixes.

Covers:
  * P0-1: checkpoint round-trip restores torch.Tensor objects (a resume never
    sees python lists where tensors are expected, and DISTILL accepts the
    restored tensors).
  * P0-3: REBIRTH fails closed — never publishes on missing/failed verdicts,
    honors reflexion_final_verdict and judge_status, and atomically replaces
    the output directory on success.
  * P1-1: EXCISE rolls back to pristine weights when any projection raises.
  * P1-2: the sweep constrains candidates to methods EXCISE can realize,
    and EXCISE rejects an unrealizable method loudly.
  * P1-3: the invocation cap forces a failure verdict instead of leaving a
    non-terminal run to be treated as success.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import graph as graph_mod
from config import ModelConfig
from distill import distill_node
from excise import EXCISE_REALIZED_METHODS, excise_node
from graph import invoke_with_cap
from model_registry import get_model, set_model
from rebirth import rebirth_node


# ---------------------------------------------------------------------- #
# Toy model mirroring the EXCISE decoder contract.
# ---------------------------------------------------------------------- #
class _ToyMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.down_proj = nn.Linear(hidden, hidden, bias=False)


class _ToyAttn(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _ToyLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self_attn = _ToyAttn(hidden)
        self.mlp = _ToyMLP(hidden)


class _ToyDecoder(nn.Module):
    def __init__(self, n_layers, hidden):
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer(hidden) for _ in range(n_layers)])


class _ToyModel(nn.Module):
    def __init__(self, n_layers=8, hidden=16):
        super().__init__()
        self.model = _ToyDecoder(n_layers, hidden)
        self.dummy = nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return next(self.parameters()).device


@pytest.fixture
def toy_state():
    torch.manual_seed(0)
    n_layers, hidden = 8, 16
    model = _ToyModel(n_layers, hidden)
    set_model(model)

    direction = torch.randn(hidden)
    direction = direction / direction.norm()
    harm_acts, harmless_acts = {}, {}
    for i in range(n_layers):
        harm_acts[i] = [direction + torch.randn(hidden) * 0.1]
        harmless_acts[i] = [-direction + torch.randn(hidden) * 0.1]

    cfg = ModelConfig(
        model_id="toy/test",
        model_arch="dense",
        dir_method="diff_means",
        alpha=1.0,
        n_directions=1,
        target_weights=["o_proj", "down_proj"],
        separation_threshold=0.0,
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
# P0-1: tensor-safe checkpoint round-trip.
# ---------------------------------------------------------------------- #
class TestCheckpointRoundingtrip:
    def test_serializer_restores_tensors(self):
        ser = graph_mod._json.JsonPlusSerializer()
        payload = {
            "harm_acts": {0: [torch.randn(3)]},
            "pristine_state_dict": {"w": torch.tensor([1.0, 2.0])},
            "refusal_directions": {0: torch.tensor([0.5, 0.5, 0.5])},
        }
        restored = ser.loads_typed(ser.dumps_typed(payload))
        assert isinstance(restored["harm_acts"][0][0], torch.Tensor)
        assert isinstance(restored["pristine_state_dict"]["w"], torch.Tensor)
        assert isinstance(restored["refusal_directions"][0], torch.Tensor)

    def test_sqlite_saver_roundtrips_tensors(self, tmp_path):
        db = tmp_path / "ckpt.sqlite"
        saver = graph_mod._make_checkpointer(str(db))
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        state = {
            "harm_acts": {0: [torch.tensor([1.0, 2.0])]},
            "pristine_state_dict": {"w": torch.tensor([3.0])},
        }
        ckpt = {
            "v": 1, "ts": "x", "id": "1", "channel_values": state,
            "channel_versions": {"__start__": 1, ":channel:1": 1},
            "versions_seen": {"__input__": {}, "__start__": {}},
            "pending_sends": [],
            "metadata": {"source": "input", "step": 0, "writes": None,
                         "parents": None, "created_by": "input"},
        }
        saver.put(config, ckpt, {}, {})
        got = saver.get_tuple(config).checkpoint["channel_values"]
        assert isinstance(got["harm_acts"][0][0], torch.Tensor)
        assert isinstance(got["pristine_state_dict"]["w"], torch.Tensor)

    def test_distill_consumes_restored_tensors(self, toy_state):
        ser = graph_mod._json.JsonPlusSerializer()
        roundtripped = ser.loads_typed(
            ser.dumps_typed({"h": toy_state["harm_acts"], "b": toy_state["harmless_acts"]})
        )
        toy_state["harm_acts"] = roundtripped["h"]
        toy_state["harmless_acts"] = roundtripped["b"]
        out = distill_node(toy_state)
        assert len(out["refusal_directions"]) == toy_state["num_layers"]


# ---------------------------------------------------------------------- #
# P0-3: REBIRTH fails closed.
# ---------------------------------------------------------------------- #
def _cfg(push_to_hub=None):
    return SimpleNamespace(
        push_to_hub=push_to_hub,
        model_id="toy/test",
        method="advanced",
        dir_method="diff_means",
        alpha=1.0,
        passes=1,
        hub_token=None,
        hub_private=False,
        model_card_targets={},
    )


def _passing_state(**overrides):
    state = {
        "config": _cfg(),
        "architecture": "dense",
        "quality_pass": True,
        "judge_verdict": "pass",
        "judge_status": "ok",
        "reflexion_final_verdict": "success",
        "target_layers": [0],
        "target_weights": ["o_proj"],
        "hidden_size": 16,
        "num_layers": 8,
        "passes_completed": 1,
        "separation_scores": {0: 3.0},
        "judge_refusal_rate": 0.1,
        "judge_quality_mean": 0.8,
        "experience_db": None,
    }
    state.update(overrides)
    return state


class _FakeSaveModel:
    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "weights.bin").write_bytes(b"new")


class TestRebirthFailClosed:
    def test_failure_does_not_create_output_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = rebirth_node(
            _passing_state(judge_verdict=None, reflexion_final_verdict="incompatible")
        )
        assert out["output_path"] is None
        assert not Path("abliterated_test").exists()
        assert (tmp_path / "abliterated_test.FAILED").exists()
        assert out["metadata"]["pipeline_passed"] is False
        assert out["metadata"]["final_verdict"] == "incompatible"

    def test_reflexion_incompatible_is_a_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = rebirth_node(
            _passing_state(judge_verdict=None, reflexion_final_verdict="incompatible")
        )
        assert out["output_path"] is None
        assert out["metadata"]["pipeline_passed"] is False

    def test_degraded_judge_is_not_publishable(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = rebirth_node(_passing_state(judge_status="degraded"))
        assert out["output_path"] is None
        assert out["metadata"]["pipeline_passed"] is False

    def test_missing_verdict_fails_closed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = rebirth_node(
            _passing_state(
                quality_pass=False, judge_verdict=None,
                reflexion_final_verdict=None, judge_status=None,
            )
        )
        assert out["output_path"] is None

    def test_success_atomically_replaces_stale_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        set_model(_FakeSaveModel())
        stale = Path("abliterated_test")
        stale.mkdir()
        (stale / "stale_old.pt").write_text("old")

        out = rebirth_node(_passing_state())
        assert out["output_path"] == "abliterated_test"
        assert (stale / "weights.bin").exists()
        assert (stale / "abliteration_metadata.json").exists()
        meta = json.loads((stale / "abliteration_metadata.json").read_text())
        assert meta["pipeline_passed"] is True
        assert meta["final_verdict"] == "success"
        # The stale artifact was atomically removed.
        assert not (stale / "stale_old.pt").exists()


# ---------------------------------------------------------------------- #
# P1-1: EXCISE transactional rollback.
# ---------------------------------------------------------------------- #
class TestExciseRollback:
    def test_exception_restores_pristine_weights(self, toy_state, monkeypatch):
        distilled = distill_node(toy_state)
        toy_state.update(distilled)
        toy_state["target_layers"] = [0, 1]

        model = get_model()
        before = model.model.layers[0].self_attn.o_proj.weight.detach().clone()

        import excise as excise_mod

        def _boom(_w, _d, _alpha):
            raise RuntimeError("boom")

        monkeypatch.setattr(excise_mod, "_project_2d", _boom)
        with pytest.raises(RuntimeError):
            excise_node(toy_state)
        after = model.model.layers[0].self_attn.o_proj.weight.detach().clone()
        assert torch.allclose(before, after)


# ---------------------------------------------------------------------- #
# P1-2: sweep / EXCISE method alignment.
# ---------------------------------------------------------------------- #
class TestSweepMethodAlignment:
    def test_sweep_candidates_drop_unrealizable_methods(self):
        cfg = SimpleNamespace(
            sweep_methods=["steering", "lora", "bias_vectors", "mpoa", "advanced"],
            sweep_dir_methods=None,
            sweep_layer_sets=None,
            sweep_alphas=None,
            sweep_passes=None,
            sweep_target_weights=None,
            method="advanced",
            dir_method="diff_means",
            alpha=1.0,
            passes=1,
            target_layers=None,
            target_weights=["o_proj"],
        )
        from sweep import _build_candidates

        grid = _build_candidates(cfg)
        methods = {c["method"] for c in grid}
        assert methods <= set(EXCISE_REALIZED_METHODS)
        assert "steering" not in methods
        assert "lora" not in methods

    def test_excise_rejects_unrealizable_method(self, toy_state):
        toy_state.update(distill_node(toy_state))
        toy_state["target_layers"] = [0]
        with pytest.raises(ValueError):
            excise_node({**toy_state, "method": "steering"})


# ---------------------------------------------------------------------- #
# P1-3: invocation cap forces failure.
# ---------------------------------------------------------------------- #
class _FakeGraph:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def invoke(self, initial_state, config=None):
        self.calls += 1
        return self.outputs[min(self.calls - 1, len(self.outputs) - 1)]


class TestInvokeWithCap:
    def test_cap_reached_force_failed_verdict(self):
        g = _FakeGraph([{"judge_verdict": "fail_refusal"}])  # always non-terminal
        result = invoke_with_cap(g, {}, "t", config=None, max_invocations=2, pause=0)
        assert result["reflexion_final_verdict"] == "failed"
        assert g.calls == 2

    def test_terminal_verdict_breaks(self):
        g = _FakeGraph([{"judge_verdict": "pass"}])
        result = invoke_with_cap(g, {}, "t", config=None, max_invocations=3, pause=0)
        assert result["judge_verdict"] == "pass"
        assert g.calls == 1

    def test_missing_verdict_is_not_terminal(self):
        g = _FakeGraph([{"judge_verdict": "pending"}])
        result = invoke_with_cap(g, {}, "t", config=None, max_invocations=1, pause=0)
        assert result["reflexion_final_verdict"] == "failed"
