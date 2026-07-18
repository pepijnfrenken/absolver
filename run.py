#!/usr/bin/env python3
"""CLI entry point for Absolver.

Usage:
    python run.py <config.yaml> [--resume <thread_id>] [--platform local|modal|molab]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from inside ~/.absolver/ without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config  # noqa: E402
from graph import build_abliteration_graph  # noqa: E402


def main() -> None:
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
        "--resume", help="Resume from a checkpoint thread id", default=None
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
    config = load_config(args.config)
    graph = build_abliteration_graph()

    initial_state = {"config": config}
    thread_id = args.resume or Path(args.config).stem

    result = graph.invoke(
        initial_state, config={"configurable": {"thread_id": thread_id}}
    )

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
