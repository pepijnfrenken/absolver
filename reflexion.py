"""REFLEXION node: KB-grounded strategy retry for the Absolver pipeline.

When the pipeline stalls (low separation, high refusal, or poor quality),
this node picks the next strategy from ``ModelConfig.reflexion_strategy_space``,
optionally consults a knowledge base via a direct LLM API call, and routes
the flow to the appropriate downstream node (probe / distill / excise / rebirth).
"""
from __future__ import annotations

import os

from llm_api import chat_completion
from prompts import REFLEXION_KB_PROMPT_TEMPLATE
from state import AbliterationState

# Rotation orders for the "structural retry" strategies. These make the
# otherwise-no-op fallback actions real: e.g. switch_method actually moves the
# pipeline onto a different ablation method before re-running EXCISE.
_METHOD_ROTATION: list[str] = [
    "steering", "mpoa", "lora", "bias_vectors",
    "direct_ablation", "projected", "advanced",
]
_ALPHA_LADDER: list[float] = [2.0, 4.0, 8.0, 10.0, 20.0]
_WEIGHT_TOGGLES: list[list[str]] = [
    ["o_proj"],
    ["down_proj"],
    ["o_proj", "down_proj"],
]


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
                with open(fpath, encoding="utf-8", errors="ignore") as f:
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
        "switch_method",        # weight-projection didn't move refusal → try steering/lora/bias
        "increase_alpha",       # same method, push alpha harder (steering works at alpha 10)
        "expand_prompts",       # more prompts = more signal for the sweep
        "switch_dir_method",    # paired vs diff_means vs svd directions
        "adjust_alpha",
        "change_weights",       # o_proj -> o_proj+down_proj etc.
        "expand_target_layers",
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
                kb_llm = chat_completion(
                    kb_prompt,
                    model=getattr(cfg, "judge_model", None) or "deepseek-v4-flash",
                    base_url=getattr(cfg, "judge_base_url", None) or "https://freeinference.org/v1",
                    api_key=getattr(cfg, "judge_api_key", None) or None,
                    max_tokens=500,
                    temperature=0.0,
                    timeout=45,
                )[:500] or None
            except Exception:
                kb_llm = None

    # ------------------------------------------------------------------ #
    # Map strategy -> next node action.
    # ------------------------------------------------------------------ #
    action_map = {
        "expand_prompts": "probe",
        "switch_dir_method": "distill",
        "adjust_alpha": "excise",
        "increase_alpha": "excise",
        "switch_method": "excise",
        "change_weights": "excise",
        "expand_target_layers": "distill",
        "skip_model": "rebirth",
    }
    next_action = action_map.get(strategy, "rebirth")

    # ------------------------------------------------------------------ #
    # Build the return dict. The per-attempt history entry is kept as a
    # mutable local so the strategy branches below can annotate it with the
    # concrete mutation they applied (tried_method / alpha / target_weights)
    # for cross-attempt bookkeeping (switch_method skips these).
    # ------------------------------------------------------------------ #
    history_entry = {
        "attempt": attempt,
        "strategy": strategy,
        "reason": f"sep={max_sep:.1f}, refusal={refusal:.2f}, quality={quality:.2f}",
        "kb_llm": kb_llm,
    }
    ret: dict = {
        "reflexion_attempts": attempt,
        "reflexion_history": (state.get("reflexion_history", []) or []) + [history_entry],
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
        new_alpha = cfg.alpha * (0.5 if quality < cfg.judge_quality_threshold else 1.5)
        ret["config"] = cfg.model_copy(update=dict(alpha=min(new_alpha, 1.0)))
    elif strategy == "expand_target_layers":
        num_layers = state.get("num_layers", 0)
        existing = state.get("target_layers", [])
        if num_layers > 0:
            # Double the target set: add every other layer not already selected.
            all_layers = set(range(num_layers))
            expanded = list(set(existing) | all_layers)
            expanded.sort()
            ret["target_layers"] = expanded
    elif strategy == "switch_method":
        # Move the pipeline onto the next untried ablation method and restrict
        # the next sweep to exactly that method (so EXCISE actually tests it).
        tried = {
            h.get("tried_method")
            for h in (state.get("reflexion_history", []) or [])
            if h.get("tried_method")
        }
        tried |= {cfg.method, state.get("method"), None}
        nxt_method = next((m for m in _METHOD_ROTATION if m not in tried), None)
        if nxt_method is not None:
            ret["config"] = cfg.model_copy(update=dict(
                method=nxt_method, sweep_methods=[nxt_method],
            ))
            history_entry["tried_method"] = nxt_method
    elif strategy == "increase_alpha":
        # Push alpha up a fixed ladder and force the next sweep to test that
        # specific value (steering in particular needs high alphas).
        next_alpha = next((a for a in _ALPHA_LADDER if a > cfg.alpha), _ALPHA_LADDER[-1])
        ret["config"] = cfg.model_copy(update=dict(
            alpha=next_alpha, sweep_alphas=[next_alpha],
        ))
        history_entry["alpha"] = next_alpha
    elif strategy == "change_weights":
        # Cycle which modules get projected: o_proj -> down_proj -> both.
        cur = set(cfg.target_weights or [])
        idx = next((i for i, w in enumerate(_WEIGHT_TOGGLES) if set(w) == cur), 0)
        nxt_weights = _WEIGHT_TOGGLES[(idx + 1) % len(_WEIGHT_TOGGLES)]
        ret["config"] = cfg.model_copy(update=dict(target_weights=list(nxt_weights)))
        history_entry["target_weights"] = list(nxt_weights)
    elif strategy == "skip_model":
        ret["reflexion_final_verdict"] = "incompatible"

    return ret
