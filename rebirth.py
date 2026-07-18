"""REBIRTH node: persist the abliterated model, push to Hub, record experience."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from state import AbliterationState


def rebirth_node(state: AbliterationState) -> Dict[str, Any]:
    """Save the modified model to disk, optionally push to the HF Hub, write
    an ``abliteration_metadata.json`` sidecar, and record the run in the
    experience DB.

    Returns a partial state dict with ``output_path``, ``hub_push_success``,
    and ``metadata``.
    """
    cfg = state["config"]
    model = state["model_obj"]
    architecture = state.get("architecture", "")

    # 1. Resolve output directory and make sure it exists.
    #    When pushing to the Hub we still need a local copy to upload from.
    output_dir = Path(
        cfg.push_to_hub or f"abliterated_{cfg.model_id.split('/')[-1]}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Save model weights.
    #    Diffusion targets only modify the text encoder, so we serialize its
    #    state_dict directly. For everything else we prefer save_pretrained,
    #    but fall back to a raw state_dict dump if that fails (CPU/seq2seq/
    #    custom architectures sometimes reject save_pretrained).
    if architecture == "diffusion_encoder":
        torch.save(model.state_dict(), output_dir / "text_encoder.pt")
    else:
        try:
            model.save_pretrained(str(output_dir))
        except Exception as exc:
            print(f"save_pretrained failed ({exc}); falling back to state_dict")
            torch.save(model.state_dict(), output_dir / "pytorch_model.pt")

    # 3. Write the metadata sidecar.
    separation_scores = state.get("separation_scores") or {}
    metadata: Dict[str, Any] = {
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
    }
    with (output_dir / "abliteration_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 4. Push to the HF Hub (best-effort).
    hub_success = False
    if cfg.push_to_hub:
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
