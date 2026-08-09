"""Programmatic entry point for the Absolver pipeline.

Use :func:`run_pipeline` from notebooks or other Python code; use ``run.py``
for the CLI.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from config import load_config
from graph import build_abliteration_graph, invoke_with_cap, warn_missing_keys


def _setup_logging(config_stem: str) -> str:
    """Configure stdlib logging to a timestamped file under logs/ (INFO+).

    Mirrors run.py so programmatic callers also leave a debug trace. The node
    modules use ``logging.getLogger(__name__)``; with no root handler these
    lines are silently dropped. Safe to call any time (force resets)."""
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{config_stem}_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return str(log_path)


def run_pipeline(config_path: str | Path, thread_id: str | None = None) -> dict:
    """Run the full abliteration pipeline from a config file.

    Args:
        config_path: Path to a YAML config file.
        thread_id: Optional LangGraph thread id for checkpoint recovery.
            If omitted, derived from the config file name.

    Returns:
        The final :class:`AbliterationState` dict with all results.
    """
    config = load_config(config_path)
    _setup_logging(Path(config_path).stem)
    # P1-3: surface a missing judge/LLM key loudly instead of the silent
    # keyword-scoring fallback.
    warn_missing_keys(config)
    graph = build_abliteration_graph()

    initial = {"config": config}
    tid = thread_id or Path(config_path).stem

    # P1-2: cap TOTAL graph.invoke calls at pipeline_max_invocations so an
    # otherwise-unbounded reflexion/ouroboros loop terminates. The stable
    # ``tid`` gives the SqliteSaver checkpoint continuity for --resume.
    return invoke_with_cap(graph, initial, tid, config=config)
