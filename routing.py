"""LangGraph conditional-edge router functions for the Absolver pipeline.

Each router is a pure function ``(state) -> str`` returning the name of the
next node to execute. They are wired into the graph via
``add_conditional_edges`` in ``graph.py``.
"""
from __future__ import annotations

from state import AbliterationState

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

    - ``pass``          -> ``rebirth``
    - ``fail_refusal``  -> ``excise`` if under the ouroboros cap, else
                            ``reflexion`` (if enabled)
    - ``fail_quality``  -> ``reflexion`` (if enabled)
    - anything else     -> ``rebirth``
    """
    cfg = state["config"]
    verdict = state.get("judge_verdict", "pass")
    if verdict == "pass":
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
