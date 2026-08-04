"""Configuration for Absolver: model, pipeline, eval, judge, reflexion, HF hub."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

# Default reflexion strategy ladder. Order matters: earlier = cheaper.
DEFAULT_STRATEGY_SPACE: list[str] = [
    "expand_prompts",
    "switch_dir_method",
    "adjust_alpha",
    "expand_target_layers",
    "skip_model",
]


class ModelConfig(BaseModel):
    """Top-level configuration for an Absolver run.

    Fields are grouped by concern: model loading, tokenizer, pipeline,
    evaluation, judging, reflexion, and Hugging Face Hub publishing.

    YAML configs may nest ``judge:`` and ``reflexion:`` sub-mappings;
    :func:`load_config` flattens those into the ``judge_*`` / ``reflexion_*``
    fields below before validation.
    """

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    model_id: str
    """HF Hub repo id or local path of the model to load."""
    model_arch: str = "auto"
    """Architecture family ('auto' to infer, or 'dense'/'moe'/'diffusion_encoder')."""
    dtype: str = "bfloat16"
    """Weights dtype: 'bfloat16', 'float16', 'float32', ..."""
    device: str = "auto"
    """Device map: 'auto', 'cpu', 'cuda', 'cuda:0', ..."""
    quantize: str | None = None
    """Optional quantization: '4bit' or '8bit' (bitsandbytes)."""
    revision: str | None = None
    """HF Hub revision/commit SHA."""
    trust_remote_code: bool = False
    """Allow loading of custom modeling files from the hub."""
    low_cpu_mem_usage: bool = True
    """Use lazy weight loading to reduce peak CPU memory."""
    variant: str | None = None
    """Weight file variant, e.g. 'fp16'."""
    offload_folder: str | None = None
    """Disk folder for accelerate CPU offload."""
    max_memory_gpu: int | None = None
    """Per-GPU max memory in bytes for device_map='auto'."""
    max_memory_cpu: int | None = None
    """Max CPU memory in bytes for device_map='auto'."""

    # ------------------------------------------------------------------ #
    # Compute platform
    # ------------------------------------------------------------------ #
    platform: str = "local"
    """Execution platform: 'local', 'modal', or 'molab'."""
    modal_gpu: str = "L4"
    """Modal GPU type: L4, A10G, A100, H100."""
    modal_timeout: int = 7200
    """Modal function timeout in seconds."""
    molab_url: str | None = None
    """Molab endpoint URL (overrides $MOLAB_URL)."""
    molab_token: str | None = None
    """Molab API token (overrides $MOLAB_TOKEN)."""

    # ------------------------------------------------------------------ #
    # Tokenizer
    # ------------------------------------------------------------------ #
    tokenizer_id: str | None = None
    """Override tokenizer id; defaults to `model_id`."""
    use_fast_tokenizer: bool = True
    """Prefer the Rust fast tokenizer."""
    padding_side: str = "left"
    """Token padding side ('left' or 'right')."""
    truncation: bool = True
    """Truncate inputs longer than `max_seq_len`."""

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    method: str = "advanced"
    """Pipeline method name ('advanced', 'basic', 'bias_vectors')."""
    n_directions: int = 4
    """Number of steering directions to extract per layer."""
    dir_method: str = "diff_means"
    """Direction extraction ('diff_means', 'svd', 'leace', 'whitened_svd')."""
    alpha: float = 0.5
    """Projection strength applied to each direction."""
    passes: int = 3
    """Number of refinement passes over the dataset."""
    refinement_passes: int = 0
    """Extra refinement passes after the first excise/verify cycle."""
    target_layers: list[int] = Field(default_factory=list)
    """Explicit target layer indices; empty = auto-select in DISTILL."""
    target_weights: list[str] = Field(default_factory=lambda: ["o_proj", "down_proj"])
    """Weight names to project ('o_proj', 'down_proj', 'expert.down')."""
    batch_size: int = 8
    """Per-device batch size for the pipeline."""
    max_seq_len: int = 2048
    """Maximum tokenized sequence length."""
    cache_dir: str | None = None
    """Optional HF cache directory override."""

    # ------------------------------------------------------------------ #
    # Sweep — search the ablation space instead of trusting one config.
    # ------------------------------------------------------------------ #
    sweep_enabled: bool = False
    """Try multiple ablation methods & configs, pick the best."""
    sweep_methods: list[str] = Field(default_factory=list)
    """Candidate methods: advanced, bias_vectors, direct_ablation, lora.
    Empty = single (base method)."""
    sweep_dir_methods: list[str] = Field(default_factory=list)
    """Candidate direction extraction: diff_means, svd, leace, whitened_svd.
    Empty = single (base dir_method)."""
    sweep_layer_sets: list[list[int]] = Field(default_factory=list)
    """Candidate layer sets, e.g. [[23,22],[23],[22]]. Empty = single (base)."""
    sweep_alphas: list[float] = Field(default_factory=list)
    """Candidate alphas, e.g. [0.2, 0.5]. Empty = single (base alpha)."""
    sweep_passes: list[int] = Field(default_factory=list)
    """Candidate pass counts, e.g. [1, 2]. Empty = single (base passes)."""
    sweep_target_weights: list[list[str]] = Field(default_factory=list)
    """Optional per-candidate target weights. Empty = base target_weights."""
    sweep_refusal_weight: float = 1.0
    """Objective weight on keeping refusal low (subtracted)."""
    sweep_quality_weight: float = 1.0
    """Objective weight on response quality (added)."""
    sweep_judge_enabled: bool = True
    """Re-score top sweep candidates with the real LLM judge before picking."""
    sweep_judge_finalists: int = 5
    """How many keyword-top candidates get LLM-judge re-scored."""
    sweep_judge_prompts: int = 5
    """How many harmful prompts to LLM-judge per finalist."""
    sweep_kl_quality: bool = True
    """Use Heretic-style KL divergence vs pristine model as the quality
    metric (capability-damage proxy) instead of response length."""
    sweep_kl_topk: int = 128
    """Vocab positions to compare in the KL divergence (top-K by base prob)."""

    # ------------------------------------------------------------------ #
    # Probe / verify sizing
    # ------------------------------------------------------------------ #
    n_probe_prompts: int = 20
    """Number of harmful/harmless prompts used in PROBE."""
    n_verify_prompts: int = 20
    """Number of harmful prompts used in VERIFY refusal-rate check."""
    probe_mode: str = "input"
    """Activation harvest mode: 'input' (last prompt token, harmful vs
    harmless prompts) or 'paired' (output phase: unprimed refusal vs
    affirmative-prefilled response on the SAME prompts). 'paired' matches
    the successful LFM2.5 abliteration recipe — it removes the topic and
    difficulty confounds of diff_means over different prompt sets."""
    paired_prefill: str = "Sure, I can help with that."
    """Affirmative prefill appended to harmful prompts in 'paired' probe
    mode; the model continues from this compliant start."""
    paired_max_new_tokens: int = 64
    """Response length generated per prompt in 'paired' probe mode."""
    verify_sample_size: int = 0
    """MMLU-mini sample size (0 = skip MMLU)."""
    verify_benchmarks: list[str] = Field(default_factory=list)
    """Optional override list. Empty = auto-derive benchmark set from
    model_card_targets keys that have a registered runner."""
    verify_benchmark_samples: int = 25
    """Per-benchmark sample size when using built-in subsets."""
    model_card_targets: dict[str, float] = Field(default_factory=dict)
    """Optional dict of {benchmark_name: target_score} for verdict comparison."""

    # ------------------------------------------------------------------ #
    # Evaluation thresholds
    # ------------------------------------------------------------------ #
    separation_threshold: float = 5.0
    """Minimum activation separation required to accept a direction."""
    ouroboros_threshold: float = 0.5
    """Refusal rate above which an ouroboros (retry) pass is triggered."""
    max_ouroboros_passes: int = 3
    """Max EXCISE retries before escalating to REFLEXION."""
    eval_batch_size: int = 16
    """Batch size used during evaluation."""
    eval_dataset: str | None = None

    # ------------------------------------------------------------------ #
    # Judge (LLM-as-judge via direct OpenAI-compatible API call)
    # ------------------------------------------------------------------ #
    judge_enabled: bool = False
    """Enable the LLM-as-judge VERIFY stage."""
    judge_model: str = "deepseek-v4-flash"
    """Model id used for adversarial judging (chat-completions `model`)."""
    judge_base_url: str = "https://freeinference.org/v1"
    """OpenAI-compatible base URL for the judge endpoint."""
    judge_prompts: int = 20
    """Number of prompts to send to the judge."""
    judge_refusal_threshold: float = 0.3
    """Refusal-rate threshold above which the judge rejects a sample."""
    judge_quality_threshold: float = 0.4
    """Quality-score threshold below which the judge rejects a sample."""
    judge_temperature: float = 0.0
    """Sampling temperature for the judge."""
    judge_max_tokens: int = 1024
    """Maximum tokens generated per judge response."""
    judge_api_key: str | None = None
    """API key for the judge endpoint (else pulled from env)."""
    prompt_format: str = "auto"
    """How to format prompts for generation: auto/chat/raw/thinking.
    Auto detects from the tokenizer (chat template, thinking tokens)."""

    # ------------------------------------------------------------------ #
    # Reflexion (KB-grounded strategy retry)
    # ------------------------------------------------------------------ #
    reflexion_enabled: bool = True
    """Enable the REFLEXION strategy-retry node."""
    reflexion_max_attempts: int = 3
    """Maximum reflexion retry attempts before giving up."""
    reflexion_db_path: str = "~/.absolver/experience.db"
    """SQLite path for the experience/strategy database."""
    reflexion_kb_paths: list[str] = Field(default_factory=list)
    """Paths (files or dirs) to reflexion knowledge bases."""
    reflexion_kb_max_files: int = 50
    """Max KB files to load in one reflexion pass."""
    reflexion_kb_llm_consult: bool = True
    """Allow an OMP LLM consultation over KB snippets on first attempt."""
    reflexion_strategy_space: list[str] = Field(
        default_factory=lambda: list(DEFAULT_STRATEGY_SPACE)
    )
    """Ordered fallback strategies (cheapest first)."""
    reflexion_temperature: float = 0.7
    """Sampling temperature for reflexion generation."""
    reflexion_max_tokens: int = 512
    """Maximum tokens generated per reflexion step."""
    reflexion_reflect_on_failure: bool = True
    """Trigger a reflection pass only when an attempt fails."""

    # ------------------------------------------------------------------ #
    # Hugging Face Hub
    # ------------------------------------------------------------------ #
    push_to_hub: str | None = None
    """Target repo id to push to (None/empty = don't push)."""
    hub_token: str | None = None
    """HF Hub access token (else pulled from `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`)."""
    hub_private: bool = True
    """Create the target hub repo as private if it does not exist."""

    # ------------------------------------------------------------------ #
    # Platform (local | modal | molab)
    # ------------------------------------------------------------------ #
    platform: str = "local"
    """Execution platform: 'local', 'modal', or 'molab'."""
    modal_gpu: str = "L4"
    """Modal GPU class: 'L4', 'A10G', 'A100', or 'H100'."""
    modal_timeout: int = 7200
    """Modal function timeout in seconds."""
    molab_url: str | None = None
    """Molab endpoint URL (or MOLAB_URL env var)."""
    molab_token: str | None = None
    """Molab API bearer token (or MOLAB_TOKEN env var)."""

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    seed: int = 42
    """Global RNG seed."""

    model_config = {"extra": "forbid"}


def _flatten_nested(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten ``judge:`` and ``reflexion:`` sub-mappings into top-level keys.

    Allows YAML configs to use either::
        judge:
          enabled: true
          model: foo
    or the flat form:: judge_enabled: true
    """
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in ("judge", "reflexion") and isinstance(value, dict):
            for sub_key, sub_val in value.items():
                out[f"{key}_{sub_key}"] = sub_val
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> ModelConfig:
    """Load a :class:`ModelConfig` from a YAML file at ``path``.

    Raises:
        FileNotFoundError: if ``path`` does not exist (with a hint listing
            the ``models/`` directory when it sits next to the config).
        ValueError: if the YAML is malformed or not a mapping.
        pydantic.ValidationError: if the parsed data fails validation.
    """
    p = Path(path)
    # Resolve relative paths against the absolver package root so configs
    # work on Modal (where CWD is /root/, not /absolver).
    if not p.is_absolute():
        p = (Path(__file__).resolve().parent / p).resolve()
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as exc:
        models_dir = p.parent / "models" if p.parent.name != "models" else p.parent.parent / "models"
        hint = ""
        if models_dir.exists():
            files = sorted(x.name for x in models_dir.iterdir() if x.suffix in {".yaml", ".yml"})
            hint = f". Available configs in {models_dir}: {files}" if files else ""
        raise FileNotFoundError(f"Config file not found: {p}{hint}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config at {p}: {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config at {p} must be a YAML mapping, got {type(data).__name__}"
        )

    flat = _flatten_nested(data)
    try:
        return ModelConfig(**flat)
    except ValidationError as exc:
        raise ValueError(
            f"Config validation failed for {p}.\n"
            f"Schema hint: ModelConfig fields are documented in config.py.\n{exc}"
        ) from exc


def dump_config(config: ModelConfig, path: str | Path) -> None:
    """Serialize ``config`` to YAML at ``path`` (parent dirs are created)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = config.model_dump(exclude={"judge_api_key", "hub_token"})
    except AttributeError:
        # pydantic v1 fallback
        data = {
            k: v
            for k, v in config.dict().items()
            if k not in {"judge_api_key", "hub_token"}
        }

    try:
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                data, f, sort_keys=False, default_flow_style=False, allow_unicode=True
            )
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Failed to write config to {p}: {exc}") from exc
