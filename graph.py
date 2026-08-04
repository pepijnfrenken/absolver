"""LangGraph assembly — tensor-safe via msgpack default handler monkey-patch."""
from __future__ import annotations

from functools import wraps

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from distill import distill_node
from excise import excise_node
from judge import judge_node
from probe import probe_node
from rebirth import rebirth_node
from reflexion import reflexion_node
from routing import route_after_judge, route_after_reflexion, route_after_verify
from state import AbliterationState
from summon import summon_node
from sweep import sweep_node
from verify import verify_node


# ── Monkey-patch msgpack to handle tensors ──────────────────────────────
import torch
from langgraph.checkpoint.serde import jsonplus as _jp

_orig_default = _jp._msgpack_default

def _tensor_default(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return _orig_default(obj)

_jp._msgpack_default = _tensor_default


def build_abliteration_graph():
    builder = StateGraph(AbliterationState)

    builder.add_node("summon", summon_node)
    builder.add_node("probe", probe_node)
    builder.add_node("distill", distill_node)
    builder.add_node("sweep", sweep_node)
    builder.add_node("excise", excise_node)
    builder.add_node("verify", verify_node)
    builder.add_node("judge", judge_node)
    builder.add_node("reflexion", reflexion_node)
    builder.add_node("rebirth", rebirth_node)

    builder.add_edge(START, "summon")
    builder.add_edge("summon", "probe")
    builder.add_edge("probe", "distill")
    builder.add_edge("distill", "sweep")
    builder.add_edge("sweep", "excise")
    builder.add_edge("excise", "verify")
    builder.add_conditional_edges("verify", route_after_verify)
    builder.add_conditional_edges("judge", route_after_judge)
    builder.add_conditional_edges("reflexion", route_after_reflexion)
    builder.add_edge("rebirth", END)

    return builder.compile(checkpointer=MemorySaver())
