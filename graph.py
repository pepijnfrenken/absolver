"""LangGraph assembly — tensor-safe via msgpack ext-code serialization.

Checkpointing: the compiled graph uses a file-backed :class:`SqliteSaver`
(crash-safe / resumable across processes) instead of the in-memory
``MemorySaver``. A crashed run can be resumed by re-invoking the graph with
the same ``thread_id`` (see P0-1). If the optional ``langgraph-checkpoint-``
sqlite backend is unavailable, we fall back to ``MemorySaver`` with a loud
warning (the run is then NOT crash-recoverable).

Resume honesty: checkpointed state may contain tensors (``harm_acts``,
``harmless_acts``, ``paired_*_acts``, ``refusal_directions``,
``pristine_state_dict``). These round-trip as real ``torch.Tensor`` objects
via a dedicated msgpack ext code, so a resumed run sees tensors (not the
python lists an earlier serializer produced). The HF model itself is NOT
checkpointed — it lives in the process-local registry and the resumed
process must reload it (see ``ensure_model_loaded`` / ``--resume`` in
``run.py``).
"""
from __future__ import annotations

import io as _io
import logging
import os
import sqlite3
import time
from pathlib import Path

import ormsgpack
import torch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde import jsonplus as _json
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
# ``langgraph``'s serializer CANNOT encode a torch.Tensor natively. The previous
# naive handler serialized tensors as ``detach().cpu().tolist()``, which
# corrupted checkpoints: ``harm_acts`` / ``harmless_acts`` /
# ``pristine_state_dict`` came back from a resume as python LISTS, and the
# tensor consumers in DISTILL/EXCISE crashed (P0-1).
#
# We instead round-trip tensors through ``torch.save`` bytes in a dedicated
# msgpack ext type. BOTH sides must be patched:
#   * serialize:  ``_msgpack_default`` — emit ``ormsgpack.Ext`` with the bytes;
#   * deserialize: the per-instance ext hook built by
#     ``_create_msgpack_ext_hook`` (the SqliteSaver's serializer is constructed
#     after this module is imported, so it picks up the patched factory).
# The ext code 8 is unused by langgraph's own codes (0-7).
_EXT_TENSOR = 8

_orig_default = _json._msgpack_default


def _tensor_default(obj):
    if isinstance(obj, torch.Tensor):
        buf = _io.BytesIO()
        torch.save(obj.detach().cpu(), buf)
        return ormsgpack.Ext(_EXT_TENSOR, buf.getvalue())
    return _orig_default(obj)


_json._msgpack_default = _tensor_default

_orig_create_ext_hook = _json._create_msgpack_ext_hook


def _tensor_create_ext_hook(allowed_modules=None):
    hook = _orig_create_ext_hook(allowed_modules)

    def _tensor_ext_hook(code, data):
        if code == _EXT_TENSOR:
            return torch.load(_io.BytesIO(data), weights_only=True)
        return hook(code, data)

    return _tensor_ext_hook


_json._create_msgpack_ext_hook = _tensor_create_ext_hook

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
        return _with_tensor_serde(MemorySaver())
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
        return _with_tensor_serde(saver)
    except Exception as exc:  # pragma: no cover - defensive fallback
        _log.warning(
            "SqliteSaver unavailable (%s); falling back to MemorySaver — "
            "this run is NOT crash-recoverable/resumable.",
            exc,
        )
        return _with_tensor_serde(MemorySaver())


def _with_tensor_serde(saver):
    """Ensure the given saver's serializer decodes torch tensors.

    ``langgraph.checkpoint.base`` instantiates a shared ``JsonPlusSerializer``
    class attribute at import time — BEFORE ``graph`` patches the
    ``_create_msgpack_ext_hook`` factory — so the saver-side deserializer may
    still be the stock one. We always bump the per-instance ``_unpack_ext_hook``
    here (idempotently) so the tensor ext code round-trips regardless of import
    order. Serialization is already handled by patching ``_msgpack_default``.
    """
    serde = getattr(saver, "serde", None)
    hook = getattr(serde, "_unpack_ext_hook", None)
    if serde is None or hook is None or getattr(hook, "_pino_tensor_wrapped", False):
        return saver

    def _tensor_ext_hook(code, data):
        if code == _EXT_TENSOR:
            return torch.load(_io.BytesIO(data), weights_only=True)
        return hook(code, data)

    _tensor_ext_hook._pino_tensor_wrapped = True
    serde._unpack_ext_hook = _tensor_ext_hook
    return saver


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


def ensure_model_loaded(config) -> dict:
    """Reload the model into the process-local registry before resuming.

    The HF model lives in ``model_registry`` (process-local) and is never
    checkpointed. ``--resume`` re-invokes the graph with the same thread_id,
    and LangGraph replays from the last checkpoint WITHOUT re-running the
    ``summon`` node — so a fresh process restoring a checkpoint has no model
    in its registry and would crash in EXCISE/VERIFY/JUDGE. Running the
    SUMMON logic up front repopulates the registry so the resumed run finds
    its model.

    Returns the SUMMON state slice (architecture, layers, ...) which carries
    the fields downstream nodes need on the resumed thread.
    """
    return summon_node({"config": config})


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

    A missing verdict is treated as NON-terminal (never silently assumed to be
    success); if the cap is hit without a terminal verdict, the last result is
    stamped ``reflexion_final_verdict="failed"`` so REBIRTH gates on failure
    instead of publishing a look-like-success artifact (P1-3).

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
        verdict = result.get("reflexion_final_verdict") or result.get("judge_verdict") or ""
        if verdict in TERMINAL_VERDICTS:
            if invocation > 1:
                _log.info(
                    "[invocation %d/%d] terminal verdict: %s",
                    invocation, max_invocations, verdict,
                )
            break
        _log.warning(
            "[invocation %d/%d] non-terminal (verdict=%r); re-invoking",
            invocation, max_invocations, verdict,
        )
        time.sleep(pause)
    else:
        # Cap reached without a terminal verdict — do NOT fall through to
        # REBIRTH as if the run succeeded. Force a failed verdict so the gate
        # in REBIRTH (and the published metadata / experience record) is
        # truthful.
        _log.warning(
            "pipeline_max_invocations=%d reached without a terminal verdict; "
            "marking the run failed.",
            max_invocations,
        )
        if result is not None:
            result = {
                **result,
                "reflexion_final_verdict": "failed",
                "judge_verdict": result.get("judge_verdict") or "fail_refusal",
            }
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
