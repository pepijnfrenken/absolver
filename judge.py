"""JUDGE node — LLM-as-judge scoring of model responses via OMP."""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Any

import torch

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


def _call_omp(judge_input: str, cfg: Any) -> str:
    """Invoke ``omp -p --model <judge_model> @<tmpfile>`` and return its stdout.

    Raises on any failure (nonzero exit, timeout, missing binary); callers
    are expected to catch and fall back to keyword scoring.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=True
    ) as f:
        f.write(judge_input)
        f.flush()
        judge_api_key = getattr(cfg, "judge_api_key", None)
        env = {**os.environ}
        if judge_api_key:
            env["OPENAI_API_KEY"] = judge_api_key
        proc = subprocess.run(
            ["omp", "-p", "--model", cfg.judge_model, f"@{f.name}"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"omp exited {proc.returncode}: {proc.stderr.strip()[:300]}"
        )
    return proc.stdout


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

    model = state["model_obj"]
    tok = state["tokenizer"]
    test_prompts = list(state.get("harmful_prompts") or [])[: cfg.judge_prompts]

    # ------------------------------------------------------------------ #
    # 1. Generate one response per harmful prompt (configurable tokens/temp).
    # ------------------------------------------------------------------ #
    responses: list[str] = []
    judge_max_tokens = cfg.judge_max_tokens
    judge_temperature = cfg.judge_temperature

    for prompt in test_prompts:
        inp = tok(
            prompt, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len
        ).to(_model_device(model))
        try:
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=judge_max_tokens,
                    do_sample=True,
                    temperature=judge_temperature,
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
                        do_sample=True,
                        temperature=judge_temperature,
                    )
                responses.append(tok.decode(out[0], skip_special_tokens=True))
            except Exception as exc2:
                logger.warning("generate() retry failed: %s", exc2)
                responses.append("")

    # ------------------------------------------------------------------ #
    # 2. Judge each (prompt, response) pair via OMP, with keyword fallback.
    # ------------------------------------------------------------------ #
    results: list[dict[str, Any]] = []
    for prompt, response in zip(test_prompts, responses):
        judge_input = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response)

        try:
            verdict_text = _call_omp(judge_input, cfg)
            verdict = _parse_verdict(verdict_text)
        except subprocess.TimeoutExpired:
            logger.warning("omp timed out; using keyword fallback")
            # Retry once, then give up and use keyword fallback.
            try:
                verdict_text = _call_omp(judge_input, cfg)
                verdict = _parse_verdict(verdict_text)
            except Exception:
                verdict = {
                    "refusal_score": _keyword_refusal_score(response),
                    "quality_score": 0.5,
                    "reason": "omp_timeout_keyword_fallback",
                }
        except (FileNotFoundError, OSError, RuntimeError, Exception) as exc:
            # OMP not installed, crashed, or returned an error — keyword fallback.
            logger.warning("omp call failed (%s); using keyword fallback", exc)
            verdict = {
                "refusal_score": _keyword_refusal_score(response),
                "quality_score": 0.5,
                "reason": "omp_error_keyword_fallback",
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

    return {
        "judge_results": results,
        "judge_refusal_rate": refusal_rate,
        "judge_quality_mean": quality_mean,
        "judge_verdict": verdict,
        "judge_evidence": evidence,
    }
