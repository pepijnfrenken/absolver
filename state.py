"""TypedDict state schema for the Absolver LangGraph pipeline."""
from __future__ import annotations

from typing import Any, TypedDict

from config import ModelConfig


class AbliterationState(TypedDict, total=False):
    """Pipeline state. All fields are optional; nodes return partial dicts
    that LangGraph merges into this schema.

    Sections mirror the node order: SUMMON -> PROBE -> DISTILL -> EXCISE ->
    VERIFY -> JUDGE -> REFLEXION -> REBIRTH, plus the shared KNOWLEDGE BASE.
    """

    # ------------------------------------------------------------------ #
    # Config / shared
    # ------------------------------------------------------------------ #
    config: ModelConfig
    """Active configuration (may be swapped by REFLEXION)."""

    # ------------------------------------------------------------------ #
    # SUMMON
    # ------------------------------------------------------------------ #
    model_loaded: bool
    """True once the model + tokenizer are loaded."""
    model_obj: Any
    """Loaded HF model (or text encoder for diffusion pipelines)."""
    tokenizer: Any
    """Loaded HF tokenizer."""
    experience_db: Any
    """ExperienceDB handle from SUMMON."""
    architecture: str
    """Detected architecture: 'dense' | 'moe' | 'diffusion_encoder'."""
    hidden_size: int
    num_layers: int
    num_experts: int | None
    layer_types: list[str] | None
    target_weights: list[str]

    # ------------------------------------------------------------------ #
    # PROBE
    # ------------------------------------------------------------------ #
    harmful_prompts: list[str]
    harmless_prompts: list[str]
    harm_acts: dict[int, Any]
    """Per-layer stacked harmful activations [n_prompts, hidden]."""
    harmless_acts: dict[int, Any]
    """Per-layer stacked harmless activations."""
    router_logits: dict[int, Any] | None

    # ------------------------------------------------------------------ #
    # DISTILL
    # ------------------------------------------------------------------ #
    refusal_directions: dict[int, Any]
    """Per-layer refusal direction (1D or [n_dirs, hidden])."""
    separation_scores: dict[int, float]
    target_layers: list[int]

    # ------------------------------------------------------------------ #
    # EXCISE
    # ------------------------------------------------------------------ #
    projection_applied: bool
    passes_completed: int
    excise_history: list[dict[str, Any]]
    alpha: float
    """Effective alpha (may be tuned by REFLEXION)."""

    # ------------------------------------------------------------------ #
    # VERIFY
    # ------------------------------------------------------------------ #
    refusal_rate: float
    ouroboros_count: int
    mmlu_score: float | None
    quality_pass: bool

    # ------------------------------------------------------------------ #
    # JUDGE
    # ------------------------------------------------------------------ #
    judge_results: list[dict[str, Any]]
    judge_refusal_rate: float
    judge_quality_mean: float
    judge_verdict: str
    """'pass' | 'fail_refusal' | 'fail_quality'."""
    judge_evidence: list[str]

    # ------------------------------------------------------------------ #
    # REBIRTH
    # ------------------------------------------------------------------ #
    output_path: str
    hub_push_success: bool
    metadata: dict[str, Any]

    # ------------------------------------------------------------------ #
    # REFLEXION
    # ------------------------------------------------------------------ #
    reflexion_enabled: bool
    reflexion_attempts: int
    reflexion_history: list[dict[str, Any]]
    reflexion_current_strategy: str
    reflexion_llm_suggestion: str | None
    reflexion_chosen_action: str
    """Next node name after REFLEXION."""
    reflexion_final_verdict: str
    """'success' | 'failed' | 'incompatible'."""

    # ------------------------------------------------------------------ #
    # KNOWLEDGE BASE (loaded lazily inside REFLEXION)
    # ------------------------------------------------------------------ #
    kb_loaded: bool
    kb_snippets: list[str]
    kb_matched_patterns: list[str]
    kb_llm_analysis: str | None
