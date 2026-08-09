"""LangGraph conditional-edge router functions for the Absolver pipeline.

Each router is a pure function ``(state) -> str`` returning the name of the
next node to execute. They are wired into the graph via
``add_conditional_edges`` in ``graph.py``.
"""
from __future__ import annotations

from state import AbliterationState


def _alpha_search_active(state: AbliterationState, cfg) -> bool:
    """True if the binary alpha search is active and NOT yet converged.

    Used by the judge router so a 'pass' verdict keeps the search looping
    (midpoint -> excise -> judge -> reflexion) until the window closes,
    instead of prematurely ending at the first passing alpha.

    NOTE: the alpha binary search only works when ``judge_enabled=true`` —
    it needs the LLM judge's refusal/quality verdicts to steer toward the
    steering ceiling, so do NOT disable the judge for an alpha-search run.
    """
    if not getattr(cfg, "reflexion_alpha_binary_search", False):
        return False
    if not getattr(cfg, "reflexion_enabled", False):
        return False
    search = state.get("alpha_search")
    if not search:
        return False
    # Converged: explicitly finished, window below tolerance, or out of budget.
    if search.get("done"):
        return False
    hi = float(search.get("hi", 0))
    lo = float(search.get("lo", 0))
    eps = float(getattr(cfg, "reflexion_alpha_search_eps", 0.01))
    if (hi - lo) < eps:
        return False
    if len(search.get("tested", []) or []) >= int(getattr(cfg, "reflexion_alpha_search_iters", 10)):
        return False
    return True


def route_after_verify(state: AbliterationState) -> str:
    """Pick the next node after VERIFY (judge disabled path).

    - If the judge is enabled, always go to ``judge``.
    - If the overall quality pass is True (refusal rate acceptable AND MMLU
      score sufficient), go to ``rebirth`` directly.
    - If refusal rate exceeds the ouroboros threshold and we are under the
      retry cap, loop back to ``excise`` for another projection pass.
    - Otherwise, if REFLEXION is enabled, try ``reflexion``; else accept
      and move to ``rebirth``.
    """
    cfg = state["config"]
    if cfg.judge_enabled:
        return "judge"
    quality_pass = state.get("quality_pass", False)
    if quality_pass:
        return "rebirth"
    if state.get("refusal_rate", 0) > cfg.ouroboros_threshold:
        if state.get("ouroboros_count", 0) < cfg.max_ouroboros_passes:
            return "excise"
    if cfg.reflexion_enabled:
        return "reflexion"
    return "rebirth"


def route_after_judge(state: AbliterationState) -> str:
    """Pick the next node after JUDGE, based on ``judge_verdict``.

    - ``pass``          -> ``reflexion`` while a binary alpha search is active
                           and unconverged (keeps hunting for the quality
                           ceiling); otherwise ``rebirth``.
    - ``fail_refusal``  -> ``excise`` if under the ouroboros cap, else
      ``reflexion`` (if enabled)
    - ``fail_quality``  -> ``reflexion`` (if enabled)
    - anything else     -> ``rebirth``
    """
    cfg = state["config"]
    verdict = state.get("judge_verdict", "pass")
    if verdict == "pass":
        if _alpha_search_active(state, cfg):
            return "reflexion"
        return "rebirth"
    if verdict == "fail_refusal":
        if state.get("ouroboros_count", 0) < cfg.max_ouroboros_passes:
            return "excise"
        if cfg.reflexion_enabled:
            return "reflexion"
    if verdict == "fail_quality":
        if cfg.reflexion_enabled:
            return "reflexion"
    return "rebirth"


def route_after_reflexion(state: AbliterationState) -> str:
    """Pick the next node after REFLEXION.

    REFLEXION sets ``reflexion_chosen_action`` to the node it wants to run
    next; default to ``rebirth`` when unset.
    """
    return state.get("reflexion_chosen_action", "rebirth")
