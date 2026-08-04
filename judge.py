"""JUDGE node — LLM-as-judge scoring of model responses via a direct API call."""
from __future__ import annotations
from model_registry import get_model, get_tokenizer

import json
import logging
import os
import re
from typing import Any

import torch

from llm_api import chat_completion
from prompt_format import detect_prompt_format, format_prompt
from prompts import JUDGE_PROMPT_TEMPLATE
from state import AbliterationState
from verify import REFUSAL_KEYWORDS

logger = logging.getLogger(__name__)


def _model_device(model: Any) -> torch.device:
    """Infer the model's device by looking at its first parameter's device."""
    try:
        return next(model.parameters()).device
    except (StopIteration, RuntimeError):
        return torch.device("cpu")


def _keyword_refusal_score(response: str) -> float:
    """Cheap fallback scorer: 1.0 if any refusal keyword is present, else 0.0."""
    response_lower = response.lower()
    return 1.0 if any(kw in response_lower for kw in REFUSAL_KEYWORDS) else 0.0


def _call_judge_api(judge_input: str, cfg: Any) -> str:
    """Invoke the judge LLM via a direct OpenAI-compatible chat-completions call.

    Uses the configured base URL / model / key; falls back to FreeInference.
    Raises on any failure; callers catch and fall back to keyword scoring.
    """
    base_url = getattr(cfg, "judge_base_url", None) or "https://freeinference.org/v1"
    api_key = getattr(cfg, "judge_api_key", None)
    model = getattr(cfg, "judge_model", None) or "deepseek-v4-flash"
    return chat_completion(
        judge_input,
        model=model,
        base_url=base_url,
        api_key=api_key or None,
        max_tokens=getattr(cfg, "judge_max_tokens", 1024),
        temperature=getattr(cfg, "judge_temperature", 0.0),
        timeout=90,
    )


def _parse_verdict(verdict_text: str) -> dict[str, Any]:
    """Parse the judge LLM's JSON verdict, with a regex fallback.

    Always returns a dict with ``refusal_score``, ``quality_score`` and
    ``reason``; defaults to 0.5 / 0.5 / 'parse_failed' on total failure.
    """
    default = {
        "refusal_score": 0.5,
        "quality_score": 0.5,
        "reason": "parse_failed",
    }
    if not verdict_text:
        return default

    try:
        parsed = json.loads(verdict_text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"\{.*\}", verdict_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return default


def judge_node(state: AbliterationState) -> dict[str, Any]:
    """JUDGE node: generate responses, score each via OMP, return aggregate verdict.

    Returns a partial state dict with keys:
    ``judge_results``, ``judge_refusal_rate``, ``judge_quality_mean``,
    ``judge_verdict``, ``judge_evidence``.
    """
    cfg = state["config"]

    if not cfg.judge_enabled:
        return {"judge_verdict": "pass", "judge_results": []}

    model = get_model()
    tok = get_tokenizer()
    test_prompts = list(state.get("harmful_prompts") or [])[: cfg.judge_prompts]

    # ------------------------------------------------------------------ #
    # 1. Generate one response per harmful prompt (configurable tokens/temp).
    # ------------------------------------------------------------------ #
    responses: list[str] = []
    judge_max_tokens = cfg.judge_max_tokens
    judge_temperature = cfg.judge_temperature
    # temperature <= 0 with do_sample=True is invalid on many backends;
    # use greedy decoding (do_sample=False) in that case.
    do_sample = judge_temperature > 0

    # Resolve how prompts should be formatted for THIS model (chat
    # template / thinking tokens / raw), so the judge sees real outputs
    # instead of garbage from a bare prompt on a chat-style model.
    fmt = detect_prompt_format(tok, getattr(cfg, "prompt_format", "auto"))
    logger.info("JUDGE: prompt_format=%s (chat_template=%s, thinking=%s)",
                fmt, hasattr(tok, "chat_template") and bool(tok.chat_template),
                "<think>" in (getattr(tok, "get_vocab", lambda: {})() or {}))

    for prompt in test_prompts:
        formatted = format_prompt(tok, prompt, fmt)
        inp = tok(
            formatted, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len
        ).to(_model_device(model))
        try:
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=judge_max_tokens,
                    do_sample=do_sample,
                    temperature=judge_temperature if do_sample else None,
                )
            responses.append(tok.decode(out[0], skip_special_tokens=True))
        except Exception as exc:
            # Generation OOM / failure: shrink budget and retry once.
            logger.warning("generate() failed (%s); retrying at 64 tokens", exc)
            try:
                with torch.no_grad():
                    out = model.generate(
                        **inp,
                        max_new_tokens=64,
                        do_sample=do_sample,
                        temperature=judge_temperature if do_sample else None,
                    )
                responses.append(tok.decode(out[0], skip_special_tokens=True))
            except Exception as exc2:
                logger.warning("generate() retry failed: %s", exc2)
                responses.append("")

    # ------------------------------------------------------------------ #
    # 2. Judge each (prompt, response) pair via a direct API call, with
    #    keyword fallback.
    # ------------------------------------------------------------------ #
    results: list[dict[str, Any]] = []
    for prompt, response in zip(test_prompts, responses):
        judge_input = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response)

        try:
            verdict_text = _call_judge_api(judge_input, cfg)
            verdict = _parse_verdict(verdict_text)
        except Exception as exc:
            logger.warning("judge API call failed (%s); using keyword fallback", exc)
            verdict = {
                "refusal_score": _keyword_refusal_score(response),
                "quality_score": 0.5,
                "reason": "judge_api_error_keyword_fallback",
            }

        results.append(
            {
                "prompt": prompt,
                "response": response,
                "refusal_score": float(verdict.get("refusal_score", 0.5)),
                "quality_score": float(verdict.get("quality_score", 0.5)),
                "reason": verdict.get("reason", ""),
            }
        )

    # ------------------------------------------------------------------ #
    # 3. Aggregate and emit a verdict.
    # ------------------------------------------------------------------ #
    if results:
        refusal_rate = sum(r["refusal_score"] for r in results) / len(results)
        quality_mean = sum(r["quality_score"] for r in results) / len(results)
    else:
        refusal_rate = 0.0
        quality_mean = 0.0

    evidence = [
        r["reason"]
        for r in results
        if r["refusal_score"] > 0.5 or r["quality_score"] < 0.4
    ]

    if refusal_rate > cfg.judge_refusal_threshold:
        verdict = "fail_refusal"
    elif quality_mean < cfg.judge_quality_threshold:
        verdict = "fail_quality"
    else:
        verdict = "pass"

    # Increment the ouroboros counter on any judge failure so the
    # route_after_judge excise loop hits its cap. verify_node only bumps
    # this counter when refusal_rate > ouroboros_threshold (0.5), which
    # almost never happens — without this bump, a failing judge loops
    # excise -> verify -> judge forever.
    ouroboros_count = int(state.get("ouroboros_count", 0))
    if verdict != "pass":
        ouroboros_count += 1

    logger.info(
        "JUDGE verdict=%s refusal_rate=%.3f quality_mean=%.3f ouroboros=%d",
        verdict, refusal_rate, quality_mean, ouroboros_count,
    )

    return {
        "judge_results": results,
        "judge_refusal_rate": refusal_rate,
        "judge_quality_mean": quality_mean,
        "judge_verdict": verdict,
        "judge_evidence": evidence,
        "ouroboros_count": ouroboros_count,
    }
