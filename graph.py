"""LangGraph assembly — tensor-safe via msgpack default handler monkey-patch.

Checkpointing: the compiled graph uses a file-backed :class:`SqliteSaver`
(crash-safe / resumable across processes) instead of the in-memory
``MemorySaver``. A crashed run can be resumed by re-invoking the graph with
the same ``thread_id`` (see P0-1). If the optional ``langgraph-checkpoint-``
sqlite backend is unavailable, we fall back to ``MemorySaver`` with a loud
warning (the run is then NOT crash-recoverable).
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path

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

_log = logging.getLogger(__name__)

# ``langgraph-checkpoint-sqlite`` is an optional backend. ``SqliteSaver`` is
# the sync, thread-safe (per-connection lock) file-backed checkpointer.
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _HAS_SQLITE = True
except Exception:  # pragma: no cover - env without the optional backend
    SqliteSaver = None  # type: ignore[assignment,misc]
    _HAS_SQLITE = False


def _resolve_checkpoint_path(checkpoint_path: str | None) -> str:
    """Expand the checkpoint DB path, creating its parent directory.

    Defaults to ``~/absolver/checkpoints/abliteration.sqlite`` when no path
    is supplied (matches the repo's ``checkpoints/`` git-ignored directory).
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join("~", "absolver", "checkpoints", "abliteration.sqlite")
    p = Path(os.path.expanduser(checkpoint_path)).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _make_checkpointer(checkpoint_path: str | None):
    """Build a checkpoint saver: sqlite file-backed, MemorySaver fallback.

    We open the sqlite connection directly (rather than via
    ``SqliteSaver.from_conn_string``'s context manager) because the compiled
    graph outlives any single ``with`` block — the connection must remain
    open for the graph's whole lifetime. The compiled graph holds the only
    reference to the saver (and hence the connection), so it is closed when
    the graph is garbage-collected at process exit.
    """
    if not _HAS_SQLITE:
        _log.warning(
            "langgraph.checkpoint.sqlite is not available; falling back to "
            "MemorySaver — the run is NOT crash-recoverable/resumable."
        )
        return MemorySaver()
    try:
        path = _resolve_checkpoint_path(checkpoint_path)
        conn = sqlite3.connect(path, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        _log.info(
            "Using SqliteSaver checkpointer (file=%s): crashed runs can be "
            "resumed with --resume / the matching thread_id.",
            path,
        )
        return saver
    except Exception as exc:  # pragma: no cover - defensive fallback
        _log.warning(
            "SqliteSaver unavailable (%s); falling back to MemorySaver — "
            "this run is NOT crash-recoverable/resumable.",
            exc,
        )
        return MemorySaver()


def build_abliteration_graph(checkpoint_path: str | None = None):
    """Assemble and compile the Abliteration LangGraph.

    Args:
        checkpoint_path: Optional path to the sqlite checkpoint database.
            When None, defaults to
            ``~/absolver/checkpoints/abliteration.sqlite`` (dir is created).
            If the sqlite backend is unavailable, falls back to
            ``MemorySaver`` with a warning.

    Returns:
        The compiled graph backed by a persistent (file-based) checkpointer.
    """
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

    checkpointer = _make_checkpointer(checkpoint_path)
    return builder.compile(checkpointer=checkpointer)


# Verdicts that mean the pipeline reached a terminal state and should not be
# re-invoked (matches the Modal runner's terminal-set).
TERMINAL_VERDICTS = ("success", "incompatible", "failed", "pass")


def invoke_with_cap(
    graph,
    initial_state,
    thread_id: str,
    config=None,
    max_invocations: int | None = None,
    pause: float = 2.0,
) -> dict:
    """Invoke the compiled graph repeatedly, capped by ``pipeline_max_invocations``.

    The in-graph ouroboros/reflexion counters reset when reflexion routes back
    through probe/distill (their state returns drop ``ouroboros_count``), which
    caused unbounded excise->verify->judge->reflexion loops (P1-2). This outer
    cap guarantees termination by counting TOTAL ``graph.invoke`` calls, exactly
    like the Modal runner. The same stable ``thread_id`` is passed on every
    invocation so the SqliteSaver persists the run and a crash/resume keeps the
    thread continuity.

    Returns the last result dict.
    """
    if max_invocations is None:
        max_invocations = getattr(config, "pipeline_max_invocations", 3) if config else 3
    if max_invocations < 1:
        max_invocations = 1

    result = None
    for invocation in range(1, max_invocations + 1):
        result = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )
        verdict = (
            result.get("reflexion_final_verdict")
            or result.get("judge_verdict")
            or "success"
        )
        if verdict in TERMINAL_VERDICTS:
            if invocation > 1:
                _log.info(
                    "[invocation %d/%d] terminal verdict: %s",
                    invocation, max_invocations, verdict,
                )
            break
        _log.warning(
            "[invocation %d/%d] non-terminal (%s); re-invoking",
            invocation, max_invocations, verdict,
        )
        time.sleep(pause)
    else:
        _log.warning(
            "pipeline_max_invocations=%d reached; stopping",
            max_invocations,
        )
    return result


# Judge / LLM API key environment variables, in priority order (mirrors
# llm_api._resolve_key). The judge/generation path resolves the key from the
# config field first, then these env vars.
_KEY_ENV_VARS = ("JUDGE_API_KEY", "FREEINFERENCE_API_KEY", "OPENAI_API_KEY")


def _resolved_key(config) -> str | None:
    """The API key the judge/LLM path will actually use (config or env)."""
    key = getattr(config, "judge_api_key", None)
    if key:
        return key
    for name in _KEY_ENV_VARS:
        if os.environ.get(name):
            return os.environ[name]
    return None


def warn_missing_keys(config) -> None:
    """Loud startup warnings for a missing judge/LLM key (P1-3).

    The generation path runs the LOCAL model and needs no API key; the key is
    only used by the LLM-judge and reflexion KB consultation (chat_completion).
    When one of those is enabled but no key is set, surface it loudly instead
    of silently falling back to keyword scoring.
    """
    key = _resolved_key(config)
    judge_enabled = bool(getattr(config, "judge_enabled", False))
    kb_consults = bool(getattr(config, "reflexion_kb_llm_consult", False))

    if judge_enabled and not key:
        _log.warning(
            "judge_enabled=true but judge_api_key is not set — judge will "
            "fall back to keyword scoring; results may not reflect the LLM "
            "judge. Set judge_api_key in the config or one of "
            "$JUDGE_API_KEY / $FREEINFERENCE_API_KEY / $OPENAI_API_KEY."
        )

    if (judge_enabled or kb_consults) and not key:
        _log.warning(
            "No LLM API key available (judge_api_key unset and $JUDGE_API_KEY / "
            "$FREEINFERENCE_API_KEY / $OPENAI_API_KEY all missing) — any "
            "LLM-judge / LLM KB consultation will fall back and may not "
            "reflect the LLM."
        )
