"""REBIRTH node: persist the abliterated model, push to Hub, record experience."""
from __future__ import annotations
from model_registry import get_model

import json
import logging
from pathlib import Path
from typing import Any

import torch

from state import AbliterationState

_log = logging.getLogger(__name__)


def _write_model_card(output_dir: Path, state: AbliterationState, cfg: Any) -> None:
    """Write a README.md model card before Hub push.

    Builds the benchmark comparison table from state["benchmark_scores"]
    and cfg.model_card_targets. Logs a warning on failure but never raises.
    """
    try:
        scores = state.get("benchmark_scores", {}) or {}
        targets = cfg.model_card_targets or {}

        # Gather metrics for the card header
        alpha = cfg.alpha
        passes = cfg.passes
        target_layers = state.get("target_layers", [])
        target_weights = state.get("target_weights", [])
        refusal_rate = state.get("judge_refusal_rate", state.get("refusal_rate", 0.0))

        # Build the comparison table rows
        table_rows = ""
        if scores and targets:
            rows = []
            for bench in sorted(set(targets) | set(scores)):
                t = targets.get(bench)
                a = scores.get(bench)
                if bench == "refusal":
                    continue
                if t is not None and a is not None:
                    a_pct = a * 100.0
                    delta = a_pct - t
                    rows.append(f"| {bench:12s} | {a_pct:>6.1f}%      | {t:>5.1f}%        | {delta:>+.1f}pp |")
                elif a is not None:
                    a_pct = a * 100.0
                    rows.append(f"| {bench:12s} | {a_pct:>6.1f}%      | —           | —        |")
                elif t is not None:
                    rows.append(f"| {bench:12s} | —           | {t:>5.1f}%        | —        |")
            if rows:
                table_rows = "\n".join(rows)

        model_id = getattr(cfg, "model_id", "unknown/model")
        model_short = model_id.split("/")[-1]
        method_name = getattr(cfg, "method", "advanced")
        dir_method = getattr(cfg, "dir_method", "diff_means")
        sweep_best = state.get("sweep_best") or {}

        # Sweep-aware method description: show what the sweep picked.
        if sweep_best:
            method_name = sweep_best.get("method", method_name)
            dir_method = sweep_best.get("dir_method", dir_method)
            target_layers = sweep_best.get("target_layers", target_layers)
            alpha = sweep_best.get("alpha", alpha)
            passes = sweep_best.get("passes", passes)

        lines = [
            "---",
            "license: apache-2.0",
            "language:",
            "- en",
            "tags:",
            "- abliteration",
            "- safety",
            "- absolver",
            "- alignment-removal",
            f"base_model: {model_id}",
            "pipeline_tag: text-generation",
            "---",
            "",
            f"# {model_short} Abliterated",
            "",
            f"Abliterated version of [{model_id}](https://huggingface.co/{model_id}) — refusal direction removed via the "
            f"[**Absolver**](https://github.com/) pipeline (weight projection / steering ablation).",
            "",
            "> **Done by Absolver** — the autonomous abliteration pipeline: SUMMON → PROBE → DISTILL → "
            "SWEEP → EXCISE → VERIFY → JUDGE → REBIRTH.",
            "",
            "## Method",
            "- Pipeline: SUMMON → PROBE → DISTILL → SWEEP → EXCISE → VERIFY → JUDGE → REBIRTH (LangGraph)",
            f"- Method: {method_name} (sweep-selected across advanced / mpoa / bias_vectors / direct_ablation / projected / lora)",
            f"- Direction extraction: {dir_method}, 3 directions",
            f"- Projection strength α = {alpha}, {passes} pass(es)",
            f"- Target layers: {target_layers}",
            f"- Target weights: {target_weights}",
            "",
            "## Results (model-card comparison)",
            "| Benchmark     | Abliterated | Model Card | Δ       |",
            "|---|---|---|---|",
        ]
        if table_rows:
            lines.append(table_rows)
        else:
            lines.append("| —            | —           | —          | —      |")
        lines += [
            "",
            f"- Refusal rate: {refusal_rate:.3f} (LLM-judged on harmful prompts)",
            "",
        ]

        # Pristine-baseline delta table — the real capability-impact signal.
        pristine = state.get("pristine_scores") or {}
        if pristine and scores:
            lines += [
                "## Capability impact (abliterated vs pristine, identical eval)",
                "",
                "| Benchmark     | Abliterated | Pristine | Δ       |",
                "|---|---|---|---|",
            ]
            for bench in sorted(set(pristine) | set(scores)):
                a = scores.get(bench)
                p = pristine.get(bench)
                if a is None or p is None:
                    continue
                a_pct = a * 100.0
                p_pct = p * 100.0
                delta = a_pct - p_pct
                lines.append(
                    f"| {bench:12s} | {a_pct:>6.1f}%      | {p_pct:>6.1f}%     | {delta:>+.1f}pp |"
                )
            lines.append(
                "*Benchmarks run identically (same subset, greedy, no-thinking) on the pristine base and the "
                "abliterated model — the Δ column is the true capability cost of the edit.*"
            )
            lines.append("")

        # Behavior battery section — harmful classification + benign drift.
        behavior = state.get("behavior_report") or {}
        if behavior:
            lines += ["## Behavior analysis", ""]
            harmful = behavior.get("harmful") or {}
            benign = behavior.get("benign") or {}
            if harmful:
                lines += [
                    "### Harmful-prompt response classification",
                    "",
                    "| Category   | Count | Rate  |",
                    "|---|---|---|",
                    f"| refusal    | {harmful.get('refusal', 0)} | {harmful.get('refusal_rate', 0):.1%} |",
                    f"| evasive    | {harmful.get('evasive', 0)} | — |",
                    f"| compliant  | {harmful.get('compliant', 0)} | {harmful.get('compliant_rate', 0):.1%} |",
                    f"| garbage    | {harmful.get('garbage', 0)} | — |",
                    "",
                    f"- **Delivery rate**: {harmful.get('delivery_rate', 0):.1%} of non-refusing responses actually provide substantive content "
                    "(a low delivery rate with low refusal = the edit removed refusal but left the model unable/unwilling to comply usefully).",
                    "",
                ]
            if benign:
                lines += [
                    "### Benign behavior drift (vs pristine)",
                    "",
                    "| Metric            | Value |",
                    "|---|---|",
                    f"| Mean word overlap (Jaccard) | {benign.get('mean_word_overlap', 0):.2%} |",
                    f"| Opener match rate           | {benign.get('opener_match_rate', 0):.2%} |",
                    f"| Mean length ratio           | {benign.get('mean_length_ratio', 0):.2f}× |",
                    "",
                    "*Same benign prompts through pristine and abliterated models: high overlap + high opener match = the "
                    "edit left ordinary behavior intact.*",
                    "",
                ]

        lines += [
            "## Notes",
            "- Benchmarks use compact built-in subsets (25-50 samples, greedy, no-thinking); model-card numbers are full-set + thinking mode, so a systematic gap is expected.",
            "- Capability impact is measured per-benchmark (see Δ column); the sweep selected the config that best preserves quality while removing refusal.",
            "",
        ]

        # Capability-impact section — populated when capability_map data exists.
        cap_map = state.get("capability_map") or {}
        if cap_map:
            lines += [
                "## Capabilities affected by the ablation",
                "",
                "| Capability | Peak layer | Overlap with refusal |",
                "|---|---|---|",
            ]
            for cap_name, cap_info in cap_map.items():
                peak = cap_info.get("peak_layer", "?")
                overlap = cap_info.get("top3_overlap", "?")
                lines.append(f"| {cap_name:12s} | L{peak}        | {overlap}            |")
            lines.append("")
            lines.append("*Capability map computed via activation probing: which capabilities share layers/directions "
                         "with the refusal circuit, and how much the ablation disturbs them.*")
            lines.append("")

        readme = "\n".join(lines)
        (output_dir / "README.md").write_text(readme, encoding="utf-8")
        _log.info("README.md model card written to %s", output_dir / "README.md")

    except Exception as exc:
        _log.warning("Failed to write README.md model card: %s", exc)


def rebirth_node(state: AbliterationState) -> dict[str, Any]:
    """Save the modified model to disk, optionally push to the HF Hub, write
    an ``abliteration_metadata.json`` sidecar, and record the run in the
    experience DB.

    Returns a partial state dict with ``output_path``, ``hub_push_success``,
    and ``metadata``.
    """
    cfg = state["config"]
    model = get_model()
    architecture = state.get("architecture", "")

    # ------------------------------------------------------------------ #
    # 0. Gate: only export when the pipeline actually succeeded.
    #    Failed runs (high refusal / poor quality / rejected by judge) are
    #    recorded in the experience DB but NOT persisted to disk.
    # ------------------------------------------------------------------ #
    quality_pass = state.get("quality_pass", False)
    judge_verdict = state.get("judge_verdict", None)
    failed = (judge_verdict is not None and judge_verdict != "pass") or not quality_pass

    if failed:
        _log.warning(
            "REBIRTH: pipeline failed (quality_pass=%s, judge_verdict=%s); "
            "skipping model export. Experience DB will still be updated.",
            quality_pass,
            judge_verdict,
        )

    # 1. Resolve output directory and make sure it exists.
    #    When pushing to the Hub we still need a local copy to upload from.
    output_dir = Path(
        cfg.push_to_hub or f"abliterated_{cfg.model_id.split('/')[-1]}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Save model weights — only if the pipeline passed.
    #    Diffusion targets only modify the text encoder, so we serialize its
    #    state_dict directly. For everything else we prefer save_pretrained,
    #    but fall back to a raw state_dict dump if that fails (CPU/seq2seq/
    #    custom architectures sometimes reject save_pretrained).
    if not failed:
        if architecture == "diffusion_encoder":
            torch.save(model.state_dict(), output_dir / "text_encoder.pt")
        else:
            try:
                model.save_pretrained(str(output_dir))
            except Exception as exc:
                print(f"save_pretrained failed ({exc}); falling back to state_dict")
                torch.save(model.state_dict(), output_dir / "pytorch_model.pt")

    # 3. Write the metadata sidecar — always, even on failure, so the
    #    experience DB still gets a record of what happened.
    separation_scores = state.get("separation_scores") or {}
    metadata: dict[str, Any] = {
        "model_id": cfg.model_id,
        "method": cfg.method,
        "dir_method": cfg.dir_method,
        "alpha": cfg.alpha,
        "passes": cfg.passes,
        "target_layers": state.get("target_layers", []),
        "target_weights": state.get("target_weights", []),
        "refusal_rate": state.get("judge_refusal_rate", state.get("refusal_rate")),
        "quality_mean": state.get("judge_quality_mean"),
        "mmlu_score": state.get("mmlu_score"),
        "reflexion_attempts": state.get("reflexion_attempts", 0),
        "reflexion_history": state.get("reflexion_history", []),
        "final_verdict": state.get("reflexion_final_verdict", "success"),
        "separation_scores": {str(k): v for k, v in separation_scores.items()},
        "pipeline_passed": not failed,
    }
    with (output_dir / "abliteration_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 3b. Write README.md model card — before Hub push, only when we have
    #     a pipeline pass (no point writing a card for a failed model).
    if not failed:
        _write_model_card(output_dir, state, cfg)

    # 4. Push to the HF Hub (best-effort) — only if the pipeline passed.
    hub_success = False
    if not failed and cfg.push_to_hub:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=cfg.hub_token)
            api.create_repo(
                cfg.push_to_hub,
                exist_ok=True,
                private=cfg.hub_private,
                repo_type="model",
            )
            if architecture == "diffusion_encoder":
                api.upload_file(
                    path_or_fileobj=str(output_dir / "text_encoder.pt"),
                    path_in_repo="text_encoder.pt",
                    repo_id=cfg.push_to_hub,
                    repo_type="model",
                )
            else:
                api.upload_folder(
                    folder_path=str(output_dir),
                    repo_id=cfg.push_to_hub,
                    repo_type="model",
                )
            hub_success = True
        except Exception as exc:
            print(f"Hub push failed: {exc}")
            hub_success = False

    # 5. Record in the experience DB (if SUMMON attached one).
    db = state.get("experience_db")
    if db is not None:
        try:
            db.record_attempt(
                model_id=cfg.model_id,
                architecture=architecture,
                hidden_size=state.get("hidden_size"),
                num_layers=state.get("num_layers"),
                num_experts=state.get("num_experts"),
                method_used=cfg.method,
                dir_method=cfg.dir_method,
                alpha=cfg.alpha,
                passes=state.get("passes_completed", 0),
                target_layers=state.get("target_layers", []),
                target_weights=state.get("target_weights", []),
                max_separation=max(separation_scores.values(), default=0),
                refusal_rate=state.get(
                    "judge_refusal_rate", state.get("refusal_rate")
                ),
                quality_mean=state.get("judge_quality_mean"),
                final_verdict=state.get("reflexion_final_verdict", "success"),
                reflexion_attempts=state.get("reflexion_attempts", 0),
                reflexion_history=state.get("reflexion_history", []),
            )
        except Exception as exc:
            print(f"Experience DB record failed: {exc}")

    return {
        "output_path": str(output_dir),
        "hub_push_success": hub_success,
        "metadata": metadata,
    }
