"""LangGraph assembly for the Absolver abliteration pipeline.

Wires the eight node functions into a state graph with conditional edges
driven by :mod:`routing`. Compiles with an in-memory checkpointer so any run
can be resumed by ``thread_id``.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from distill import distill_node
from excise import excise_node
from judge import judge_node
from probe import probe_node
from rebirth import rebirth_node
from reflexion import reflexion_node
from routing import (
    route_after_distill,
    route_after_judge,
    route_after_reflexion,
    route_after_verify,
)
from state import AbliterationState
from summon import summon_node
from verify import verify_node


def build_abliteration_graph():
    """Construct and compile the abliteration state graph.

    Returns:
        A compiled :class:`StateGraph` ready for ``.invoke()`` / ``.stream()``.
    """
    builder = StateGraph(AbliterationState)

    builder.add_node("summon", summon_node)
    builder.add_node("probe", probe_node)
    builder.add_node("distill", distill_node)
    builder.add_node("excise", excise_node)
    builder.add_node("verify", verify_node)
    builder.add_node("judge", judge_node)
    builder.add_node("reflexion", reflexion_node)
    builder.add_node("rebirth", rebirth_node)

    # Linear spine.
    builder.add_edge(START, "summon")
    builder.add_edge("summon", "probe")
    builder.add_edge("probe", "distill")

    # Conditional branches.
    builder.add_conditional_edges("distill", route_after_distill)
    builder.add_edge("excise", "verify")
    builder.add_conditional_edges("verify", route_after_verify)
    builder.add_conditional_edges("judge", route_after_judge)
    builder.add_conditional_edges("reflexion", route_after_reflexion)

    # Terminal.
    builder.add_edge("rebirth", END)

    return builder.compile(checkpointer=MemorySaver())
