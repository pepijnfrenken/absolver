"""REFLEXION node: KB-grounded strategy retry for the Absolver pipeline.

When the pipeline stalls (low separation, high refusal, or poor quality),
this node picks the next strategy from ``ModelConfig.reflexion_strategy_space``,
optionally consults a knowledge base via an OMP LLM, and routes the flow
to the appropriate downstream node (probe / distill / excise / rebirth).
"""
from __future__ import annotations

import os
import subprocess
import tempfile

from prompts import REFLEXION_KB_PROMPT_TEMPLATE
from state import AbliterationState


def reflexion_node(state: AbliterationState) -> dict:
    """Choose a retry strategy and the next node for a stalled run."""
    cfg = state["config"]

    # ------------------------------------------------------------------ #
    # Disabled: route straight to REBIRTH with an incompatible verdict.
    # ------------------------------------------------------------------ #
    if not cfg.reflexion_enabled:
        return {
            "reflexion_chosen_action": "rebirth",
            "reflexion_final_verdict": "incompatible",
        }

    # ------------------------------------------------------------------ #
    # Attempt counter; bail out past the configured ceiling.
    # ------------------------------------------------------------------ #
    attempt = state.get("reflexion_attempts", 0) + 1
    if attempt > cfg.reflexion_max_attempts:
        return {
            "reflexion_attempts": attempt,
            "reflexion_chosen_action": "rebirth",
            "reflexion_final_verdict": "failed",
        }

    # ------------------------------------------------------------------ #
    # KB LOADING — lazy: only when not yet loaded or on the first attempt.
    # ------------------------------------------------------------------ #
    kb_snippets = state.get("kb_snippets", []) or []
    kb_matched = state.get("kb_matched_patterns", []) or []
    if cfg.reflexion_kb_paths and (not state.get("kb_loaded") or attempt == 1):
        kb_files: list[str] = []
        for pattern in cfg.reflexion_kb_paths:
            expanded = os.path.expanduser(pattern)
            if os.path.isfile(expanded):
                kb_files.append(expanded)
            elif os.path.isdir(expanded):
                for root, _dirs, files in os.walk(expanded):
                    for f in files:
                        if f.endswith(".md"):
                            kb_files.append(os.path.join(root, f))

        kb_snippets = []
        arch = state.get("architecture", "") or ""
        hidden_str = str(state.get("hidden_size", "") or "")
        relevant_patterns: list[str] = []
        for fpath in kb_files[: cfg.reflexion_kb_max_files]:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read(5000)
                kb_snippets.append(f"--- {os.path.basename(fpath)} ---\n{text[:3000]}")
                if arch in text or hidden_str in text:
                    relevant_patterns.append(f"matched: {os.path.basename(fpath)}")
            except Exception:
                pass
        kb_matched = list(set(relevant_patterns))

    # ------------------------------------------------------------------ #
    # Diagnose why we're stuck.
    # ------------------------------------------------------------------ #
    sep_scores = state.get("separation_scores", {}) or {}
    max_sep = max(sep_scores.values(), default=0) if sep_scores else 0
    refusal = state.get("judge_refusal_rate", state.get("refusal_rate", 1.0))
    quality = state.get("judge_quality_mean", 0.5)

    # ------------------------------------------------------------------ #
    # Pick the next strategy from the fallback ladder.
    # ------------------------------------------------------------------ #
    fallback_strategies = cfg.reflexion_strategy_space or [
        "expand_prompts",
        "switch_dir_method",
        "adjust_alpha",
        "expand_target_layers",
        "switch_to_bias_vectors",
        "skip_model",
    ]
    idx = min(attempt - 1, len(fallback_strategies) - 1)
    strategy = fallback_strategies[idx]

    # ------------------------------------------------------------------ #
    # Optional LLM KB consultation — only on the first attempt.
    # ------------------------------------------------------------------ #
    kb_llm = None
    if (
        cfg.reflexion_kb_llm_consult
        and kb_snippets
        and attempt == 1
        and strategy != "skip_model"
    ):
        context = "\n\n".join(kb_snippets[:3])
        try:
            kb_prompt = REFLEXION_KB_PROMPT_TEMPLATE.format(
                arch=state.get("architecture"),
                hidden=state.get("hidden_size"),
                layers=state.get("num_layers"),
                sep=max_sep,
                refusal=refusal,
                quality=quality,
                kb=context,
            )
        except Exception:
            kb_prompt = None
        if kb_prompt:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False
                ) as f:
                    f.write(kb_prompt)
                    tmp_name = f.name
                try:
                    r = subprocess.run(
                        ["omp", "-p", "--model", cfg.judge_model, f"@{tmp_name}"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    kb_llm = r.stdout[:500] if r.stdout else None
                finally:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
            except Exception:
                kb_llm = None

    # ------------------------------------------------------------------ #
    # Map strategy -> next node action.
    # ------------------------------------------------------------------ #
    action_map = {
        "expand_prompts": "probe",
        "switch_dir_method": "distill",
        "adjust_alpha": "excise",
        "expand_target_layers": "distill",
        "switch_to_bias_vectors": "probe",
        "skip_model": "rebirth",
    }
    next_action = action_map.get(strategy, "rebirth")

    # ------------------------------------------------------------------ #
    # Build the return dict.
    # ------------------------------------------------------------------ #
    ret: dict = {
        "reflexion_attempts": attempt,
        "reflexion_history": (state.get("reflexion_history", []) or []) + [
            {
                "attempt": attempt,
                "strategy": strategy,
                "reason": f"sep={max_sep:.1f}, refusal={refusal:.2f}, quality={quality:.2f}",
                "kb_llm": kb_llm,
            }
        ],
        "reflexion_current_strategy": strategy,
        "reflexion_llm_suggestion": kb_llm,
        "reflexion_chosen_action": next_action,
        "kb_loaded": True,
        "kb_snippets": kb_snippets,
        "kb_llm_analysis": kb_llm,
        "kb_matched_patterns": kb_matched,
    }

    # ------------------------------------------------------------------ #
    # Strategy-specific state modifications.
    # ------------------------------------------------------------------ #
    if strategy == "expand_prompts":
        from prompts import EXPANDED_HARMFUL, EXPANDED_HARMLESS

        ret["harmful_prompts"] = EXPANDED_HARMFUL
        ret["harmless_prompts"] = EXPANDED_HARMLESS
    elif strategy == "switch_dir_method":
        methods = ["diff_means", "svd", "leace", "whitened_svd"]
        cur = cfg.dir_method
        nxt = (
            methods[(methods.index(cur) + 1) % len(methods)]
            if cur in methods
            else "svd"
        )
        from config import ModelConfig

        ret["config"] = ModelConfig(**{**cfg.model_dump(), "dir_method": nxt})
    elif strategy == "adjust_alpha":
        ret["alpha"] = cfg.alpha * (0.5 if quality < 0.4 else 1.5)
    elif strategy == "switch_to_bias_vectors":
        from config import ModelConfig

        ret["config"] = ModelConfig(**{**cfg.model_dump(), "method": "bias_vectors"})
    elif strategy == "skip_model":
        ret["reflexion_final_verdict"] = "incompatible"

    return ret
