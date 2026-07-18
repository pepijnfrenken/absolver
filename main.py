"""Programmatic entry point for the Absolver pipeline.

Use :func:`run_pipeline` from notebooks or other Python code; use ``run.py``
for the CLI.
"""
from __future__ import annotations

from pathlib import Path

from config import load_config
from graph import build_abliteration_graph


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
    graph = build_abliteration_graph()

    initial = {"config": config}
    tid = thread_id or Path(config_path).stem

    return graph.invoke(
        initial, config={"configurable": {"thread_id": tid}}
    )
