"""REFLEXION node: KB-grounded strategy retry for the Absolver pipeline.

When the pipeline stalls (low separation, high refusal, or poor quality),
this node picks the next strategy from ``ModelConfig.reflexion_strategy_space``,
optionally consults a knowledge base via a direct LLM API call, and routes
the flow to the appropriate downstream node (probe / distill / excise / rebirth).
"""
from __future__ import annotations

import os
from typing import Any

from excise import EXCISE_REALIZED_METHODS
from llm_api import chat_completion
from prompts import REFLEXION_KB_PROMPT_TEMPLATE
from state import AbliterationState

# Rotation order for the "structural retry" strategies. switch_method moves
# the pipeline onto a different ablation method before re-running EXCISE.
# P1-1: the pool is restricted to the methods EXCISE can actually execute
# (excise.EXCISE_REALIZED_METHODS) so reflexion never proposes a method that
# would silently no-op as a plain projection. Order: plain projection first,
# then MPOA (the LFM2.5 winning magnitude-preserving recipe).
_METHOD_ROTATION: list[str] = list(EXCISE_REALIZED_METHODS)
_ALPHA_LADDER: list[float] = [2.0, 4.0, 8.0, 10.0, 20.0]
_WEIGHT_TOGGLES: list[list[str]] = [
    ["o_proj"],
    ["down_proj"],
    ["o_proj", "down_proj"],
]


def _step_alpha_search(cfg: Any, state: AbliterationState) -> tuple[float, dict]:
    """Advance the alpha binary search by one reflexion iteration.

    Persisted search state (``state['alpha_search']``) is a dict:
      ``lo``, ``hi``  bounds of the current search window,
      ``current``     the alpha handed to the LAST EXCISE cycle, or None if the
                      search has not tested anything yet,
      ``tested``      list of ``{"alpha": float, "refusal": float,
                      "quality": float}`` outcomes already observed,
      ``done``        True once the search converged.

    Machine (one call == one reflexion pass, pinned to 'increase_alpha'):

      1. If the search already converged, return its chosen best unchanged
         (idempotent - never restarts a finished search).
      2. No ``current`` yet -> this is the first test: propose
         ``mid0 = (lo + hi) / 2`` and dispatch that alpha to the next sweep.
      3. Otherwise the midpoint in ``current`` has just finished the
         excise->verify->judge cycle; read its outcome from state and
           - quality below threshold -> too strong -> ``hi = current``
           - else refusal above the threshold -> too weak -> ``lo = current``
           - else (pass / both acceptable) -> we can go stronger, so
             ``lo = current`` (keep chasing the accepted-quality ceiling;
             a PASS does NOT end the search).
         The outcome is always recorded in ``tested``.
      4. If ``len(tested) < reflexion_alpha_search_iters`` AND the window is
         still wider than ``reflexion_alpha_search_eps``, propose the next
         midpoint ``mid = (lo + hi) / 2`` and keep searching. Otherwise the
         search has converged -> go to step 5.
      5. Converged: pick the best candidate (step 6 below), mark ``done``
         and return it so the caller runs a confirmatory EXCISE before REBIRTH.

    ``done = True`` is set ONLY by convergence (window tolerance or iteration
    budget), never by a passing verdict.

    Returns ``(alpha_to_test, new_search_state)``.
    """
    search = dict(state.get("alpha_search") or {})
    if not search:
        search = {
            "lo": float(cfg.alpha_search_lo),
            "hi": float(cfg.alpha_search_hi),
            "current": None,
            "tested": [],
            "done": False,
        }
    # 1) Converged search: stay put (do not restart the search).
    if search.get("done") and search.get("tested"):
        best = _best_seen(search["tested"], cfg)
        return best, search

    lo, hi = float(search["lo"]), float(search["hi"])
    current = search.get("current")

    # 2) Fresh search -> propose the first midpoint.
    if current is None:
        mid = (lo + hi) / 2.0
        search.update(lo=lo, hi=hi, current=mid, tested=[], done=False)
        return mid, search

    # 3) Record the outcome of alpha == current from the just-finished cycle.
    refusal = state.get("judge_refusal_rate", state.get("refusal_rate", 1.0))
    quality = state.get("judge_quality_mean", 0.5)
    tested = list(search.get("tested", []))
    tested.append({"alpha": float(current), "refusal": float(refusal), "quality": float(quality)})
    threshold = float(getattr(cfg, "judge_quality_threshold", 0.4))
    refusal_thr = float(getattr(cfg, "judge_refusal_threshold", 0.3))
    if quality < threshold:
        hi = current                      # too strong -> go weaker
    elif refusal > refusal_thr:
        lo = current                      # too weak -> go stronger
    else:
        # both acceptable -> we can afford a stronger edit (lower refusal);
        # nudge the lower bound up toward the current alpha and keep chasing
        # the quality ceiling on the high side.
        lo = current
    lo = min(lo, float(cfg.alpha_search_hi))
    hi = max(hi, float(cfg.alpha_search_lo))

    # 4) Keep searching while the budget remains and the window is still wider
    #    than the convergence tolerance. A passing verdict already set
    #    lo = current (hunt toward the quality ceiling); only convergence below
    #    eps or hitting the iteration budget stops the search.
    cap = int(getattr(cfg, "reflexion_alpha_search_iters", 10))
    eps = float(getattr(cfg, "reflexion_alpha_search_eps", 0.01))
    if len(tested) < cap and (hi - lo) >= eps:
        mid = (lo + hi) / 2.0
        search.update(lo=lo, hi=hi, current=mid, tested=tested, done=False)
        return mid, search

    # 5) Converged: pick the best viable alpha and mark the search done.
    search.update(lo=lo, hi=hi, current=None, tested=tested, done=True)
    best = _best_seen(tested, cfg)
    return best, search


def _best_seen(tested: list, cfg: Any) -> float:
    """Highest viable alpha: among tested midpoints that keep quality at or
    above ``judge_quality_threshold`` choose the LARGEST alpha (the strongest
    steer that still passes quality), since refusal is monotonic-decreasing in
    alpha. If no candidate keeps quality, fall back to the largest alpha
    overall (strongest steer even at a quality cost)."""
    threshold = float(getattr(cfg, "judge_quality_threshold", 0.4))
    good = [t for t in tested if t["quality"] >= threshold]
    pool = good if good else tested
    return max(pool, key=lambda t: t["alpha"])["alpha"]


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
    #
    # EXCEPTION: once a binary alpha search is in progress, the strategy is
    # pinned to 'increase_alpha' and the attempt cap is bypassed — the search
    # has its own iteration cap (reflexion_alpha_search_iters) and must be
    # allowed to step excise->verify->judge repeatedly until it converges.
    # ------------------------------------------------------------------ #
    attempt = state.get("reflexion_attempts", 0) + 1
    alpha_search = state.get("alpha_search")
    searching_in_progress = bool(
        alpha_search
        and getattr(cfg, "reflexion_alpha_binary_search", False)
        and not alpha_search.get("done")
    )
    if not searching_in_progress and attempt > cfg.reflexion_max_attempts:
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
    if searching_in_progress:
        # Keep resuming the in-flight alpha search until it converges.
        strategy = "increase_alpha"
    else:
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
        # Same shared alpha policy as increase_alpha: nudge within the common
        # [alpha_search_lo, alpha_search_hi] window (no more arbitrary cap at
        # 1.0, which silently contradicted the high-alpha ladder/search).
        lo_, hi_ = cfg.alpha_search_lo, cfg.alpha_search_hi
        factor = 1.5 if quality >= cfg.judge_quality_threshold else 0.5
        new_alpha = min(max(cfg.alpha * factor, lo_), hi_)
        ret["config"] = cfg.model_copy(update=dict(alpha=new_alpha, sweep_alphas=[new_alpha]))
        history_entry["alpha"] = new_alpha
    elif strategy == "expand_target_layers":
        # One-shot guard: never expand layers twice in a run (re-expanding is
        # a no-op anyway once all layers are selected, but this also stops the
        # strategy from re-arming across ouroboros/reflexion loops).
        history = state.get("reflexion_history", []) or []
        already_expanded = any(
            h.get("expanded_target_layers") for h in history
        )
        if not already_expanded:
            num_layers = state.get("num_layers", 0)
            existing = state.get("target_layers", [])
            if num_layers > 0:
                all_layers = set(range(num_layers))
                expanded = list(set(existing) | all_layers)
                expanded.sort()
                ret["target_layers"] = expanded
                history_entry["expanded_target_layers"] = True
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
        if getattr(cfg, "reflexion_alpha_binary_search", False):
            chosen_alpha, search_state = _step_alpha_search(cfg, state)
            ret["config"] = cfg.model_copy(update=dict(
                alpha=chosen_alpha, sweep_alphas=[chosen_alpha],
            ))
            # Persist the search state so the next (pinned) reflexion pass
            # resumes from where we left off.
            ret["alpha_search"] = search_state
            history_entry["alpha"] = chosen_alpha
        else:
            # Fallback: fixed ladder (one step up).
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
        # Terminal & idempotent: once we've declared the run incompatible we
        # stay incompatible (route_after_reflexion -> rebirth). The history
        # marker makes the decision auditable and prevents re-escalation.
        ret["reflexion_final_verdict"] = "incompatible"
        history_entry["skip_model"] = True

    return ret
