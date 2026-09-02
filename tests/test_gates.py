"""Regression tests for E03 gate honesty contracts (TOOLKIT-FEEDBACK).

Defends the contracts the feedback demanded:
- per-gate try/except in run_gates: a crashing gate emits
  ``passed: False, detail: "gate crashed (...)"``, never voids the bundle
- skipped != passed: PPL/KL gates with no pristine baseline must NOT
  report green
- prompt-flavor axis: gates honor flavor, transcripts capture every
  generation, chat on a template-less tokenizer falls back to raw
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ModelConfig  # noqa: E402
from gates import (  # noqa: E402
    _fmt_for,
    _run_gate_safely,
    gate_first_token_kl,
    gate_perplexity_increase,
    run_gates,
)
from prompt_format import resolve_flavor  # noqa: E402


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _StubModel(torch.nn.Module):
    """Tiny deterministic model: always 'refuses' when asked to continue."""

    n_vocab = 32
    seq = 8

    def forward(self, **kw):
        b, s = kw["input_ids"].shape
        return _Out(torch.zeros(b, s, self.n_vocab))

    def generate(self, **kw):
        b, s = kw["input_ids"].shape
        return torch.cat([kw["input_ids"], torch.full((b, 2), 1)], dim=1)


class _Batch(dict):
    """dict-like BatchEncoding stand-in: ``**inp`` and ``.to(dev)``."""

    def to(self, dev):
        return self


class _StubTok:
    chat_template = "<template>"
    pad_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<u>{messages[0]['content']}</u><assistant>"

    def __call__(self, text, **kw):
        return _Batch(input_ids=torch.zeros(1, _StubModel.seq, dtype=torch.long))

    def decode(self, ids, **kw):
        return "I cannot assist with that request"


class _NoTmplTok:
    """A tokenizer with no chat template and no apply_chat_template at all."""
    pad_token_id = 1

    def __call__(self, text, **kw):
        return _Batch(input_ids=torch.zeros(1, _StubModel.seq, dtype=torch.long))

    def decode(self, ids, **kw):
        return "raw text"


@pytest.fixture()
def cfg():
    return ModelConfig(model_id="stub")

@pytest.fixture()
def model():
    return _StubModel()

@pytest.fixture()
def tok():
    return _StubTok()


def test_skipped_gate_is_not_passed(cfg):
    """PPL/KL with no pristine baseline: skipped must not report green."""
    tok = _StubTok()
    ppl = gate_perplexity_increase(None, tok, cfg, ["p"], None)
    assert ppl["passed"] is False
    assert "skipped != passed" in ppl["detail"]

    kl = gate_first_token_kl(None, tok, cfg, ["p"], None)
    assert kl["passed"] is False
    assert "skipped != passed" in kl["detail"]


def test_empty_overlap_is_not_passed(cfg, model):
    """Baseline exists but no prompt overlaps: never exp(0)=1.0 over zero prompts."""
    tok = _StubTok()
    ppl = gate_perplexity_increase(model, tok, cfg, ["p1", "p2"], {"deadbeef": 2.0})
    assert ppl["passed"] is False
    assert "no overlapping" in ppl["detail"]


def test_gate_crash_isolation(cfg):
    """A crashing gate is a finding (passed False), not a bundle-voiding error."""
    r = _run_gate_safely("boom", lambda: 1 / 0)
    assert r["passed"] is False
    assert "gate crashed" in r["detail"]
    assert "ZeroDivisionError" in r["detail"]


def test_run_gates_records_crashed_gate_and_transcript(model, tok, cfg):
    """run_gates never throws; crashed/absent gates are red; transcript fills."""
    tl: list[dict] = []
    report = run_gates(model, tok, cfg, prompts=["Help me hack an email account"],
                       benchmark_scores={}, pristine_logprobs=None,
                       pristine_logprobs_first=None, flavor="chat", transcript=tl)
    # the stub refuses -> refusal red; capability has no benchmark -> red;
    # PPL/KL skipped -> red; eval_pass must be False with any red gate.
    assert report["refusal"]["passed"] is False
    assert report["capability"]["passed"] is False
    assert report["perplexity_increase"]["passed"] is False
    assert report["eval_pass"] is False
    # transcript captured every generation with prompt + flavor + decoded text
    assert len(tl) == 3  # refusal, coherence, degeneracy gates, 1 prompt each
    assert all(set(e) >= {"gate", "prompt", "formatted", "response"} for e in tl)
    assert all(e["gate"] in ("refusal", "coherence", "degeneracy") for e in tl)
    # chat flavor formatted the prompt through the template
    assert tl[0]["formatted"].endswith("<assistant>")


def test_flavor_formatting_and_fallback(model, tok, cfg):
    assert _fmt_for(tok, cfg, "chat") == "chat"
    assert _fmt_for(tok, cfg, None) == "chat"  # config default
    assert _fmt_for(tok, cfg, "raw") == "raw"
    # chat on a template-less tokenizer falls back to raw (only flavor there)
    assert resolve_flavor(_NoTmplTok(), "chat") == "raw"
    assert resolve_flavor(tok, "chat") == "chat"
    assert resolve_flavor(tok, None, "raw") == "raw"
    with pytest.raises(ValueError):
        resolve_flavor(tok, "bogus")