#!/usr/bin/env python3
"""CLI entry point for Absolver.

Usage:
    python run.py <config.yaml> [--resume <thread_id>] [--platform local|modal|molab]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from inside ~/absolver/ without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config  # noqa: E402
from config import load_config  # noqa: E402
from graph import build_abliteration_graph, invoke_with_cap, warn_missing_keys  # noqa: E402

_log = logging.getLogger(__name__)


def _setup_logging(config_stem: str) -> str:
    """Configure stdlib logging to a timestamped file under logs/ (INFO+).

    The node modules use ``logging.getLogger(__name__).info/warning``; with
    no root handler configured these lines are silently dropped, so an
    unattended run leaves almost no debug trace. Wire root logging to both
    stderr and a per-run file here. Safe to call any time (force resets)."""
    import logging
    from datetime import datetime

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


def main() -> None:
    # Modal injects "run" and script-name as positional args — strip them
    # to avoid argparse clash when used as a Modal local_entrypoint.
    argv = sys.argv[1:]
    if argv and ("modal" in sys.argv[0] or argv[0] == "run"):
        argv = [a for a in argv if a not in ("run",) and not a.endswith(".py")]

    parser = argparse.ArgumentParser(
        description="Absolver - LangGraph Abliteration Pipeline"
    )
    parser.add_argument(
        "config", help="Path to config YAML (e.g., models/ornith-9b.yaml)"
    )
    parser.add_argument(
        "--platform",
        default="local",
        choices=["local", "modal", "molab"],
        help="Where to execute the pipeline (default: local).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        nargs="?",
        const="",
        help=(
            "Resume from the last checkpoint for the given config's thread. "
            "Works because the SqliteSaver persists checkpoints across "
            "processes (P0-1). Optional value: a custom thread id; when "
            "omitted, defaults to the config-file stem (the stable thread_id)."
        ),
    )
    args = parser.parse_args()

    # --- remote dispatch -------------------------------------------------
    if args.platform == "modal":
        from connectors.modal_runner import run_pipeline_modal
        run_pipeline_modal(args.config)
        return

    if args.platform == "molab":
        from connectors.molab_runner import run_pipeline_molab
        run_pipeline_molab(args.config)
        return

    # --- local (default) -------------------------------------------------
    log_file = _setup_logging(Path(args.config).stem)
    print(f"Run log: {log_file}")
    config = load_config(args.config)
    # P1-3: surface a missing judge/LLM key loudly before running.
    warn_missing_keys(config)

    graph = build_abliteration_graph()

    initial_state = {"config": config}
    # --resume keeps the SAME thread_id so the SqliteSaver resumes the last
    # checkpoint for this config (P2-1); we never wipe the .sqlite file.
    thread_id = args.resume or Path(args.config).stem

    # P1-2: cap TOTAL graph.invoke calls at pipeline_max_invocations.
    result = invoke_with_cap(graph, initial_state, thread_id, config=config)

    print(f"\n{'=' * 50}")
    print(f"Absolver Complete: {result.get('reflexion_final_verdict', 'success')}")
    print(f"Output: {result.get('output_path', 'N/A')}")
    refusal = result.get("judge_refusal_rate")
    if refusal is not None:
        print(f"Refusal rate: {refusal:.1%}")
    quality = result.get("judge_quality_mean")
    if quality is not None:
        print(f"Quality mean: {quality:.2f}")


if __name__ == "__main__":
    main()
