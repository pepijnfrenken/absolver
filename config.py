"""Configuration for Absolver: model, pipeline, eval, judge, reflexion, HF hub."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError


# Default reflexion strategy ladder. Order matters: earlier = cheaper.
DEFAULT_STRATEGY_SPACE: List[str] = [
    "expand_prompts",
    "switch_dir_method",
    "adjust_alpha",
    "expand_target_layers",
    "switch_to_bias_vectors",
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
    quantize: Optional[str] = None
    """Optional quantization: '4bit' or '8bit' (bitsandbytes)."""
    revision: Optional[str] = None
    """HF Hub revision/commit SHA."""
    trust_remote_code: bool = False
    """Allow loading of custom modeling files from the hub."""
    low_cpu_mem_usage: bool = True
    """Use lazy weight loading to reduce peak CPU memory."""
    variant: Optional[str] = None
    """Weight file variant, e.g. 'fp16'."""
    offload_folder: Optional[str] = None
    """Disk folder for accelerate CPU offload."""
    max_memory_gpu: Optional[int] = None
    """Per-GPU max memory in bytes for device_map='auto'."""
    max_memory_cpu: Optional[int] = None
    """Max CPU memory in bytes for device_map='auto'."""

    # ------------------------------------------------------------------ #
    # Tokenizer
    # ------------------------------------------------------------------ #
    tokenizer_id: Optional[str] = None
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
    target_layers: List[int] = Field(default_factory=list)
    """Explicit target layer indices; empty = auto-select in DISTILL."""
    target_weights: List[str] = Field(default_factory=lambda: ["o_proj", "down_proj"])
    """Weight names to project ('o_proj', 'down_proj', 'expert.down')."""
    batch_size: int = 8
    """Per-device batch size for the pipeline."""
    max_seq_len: int = 2048
    """Maximum tokenized sequence length."""
    cache_dir: Optional[str] = None
    """Optional HF cache directory override."""

    # ------------------------------------------------------------------ #
    # Probe / verify sizing
    # ------------------------------------------------------------------ #
    n_probe_prompts: int = 20
    """Number of harmful/harmless prompts used in PROBE."""
    n_verify_prompts: int = 20
    """Number of harmful prompts used in VERIFY refusal-rate check."""
    verify_sample_size: int = 0
    """MMLU-mini sample size (0 = skip MMLU)."""

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
    eval_dataset: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Judge (LLM-as-judge via OMP subprocess)
    # ------------------------------------------------------------------ #
    judge_enabled: bool = False
    """Enable the LLM-as-judge VERIFY stage."""
    judge_model: str = "default"
    """Model id used for adversarial judging (passed to `omp --model`)."""
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
    judge_api_key: Optional[str] = None
    """API key for the judge endpoint (else pulled from env)."""

    # ------------------------------------------------------------------ #
    # Reflexion (KB-grounded strategy retry)
    # ------------------------------------------------------------------ #
    reflexion_enabled: bool = True
    """Enable the REFLEXION strategy-retry node."""
    reflexion_max_attempts: int = 3
    """Maximum reflexion retry attempts before giving up."""
    reflexion_db_path: str = "~/.absolver/experience.db"
    """SQLite path for the experience/strategy database."""
    reflexion_kb_paths: List[str] = Field(default_factory=list)
    """Paths (files or dirs) to reflexion knowledge bases."""
    reflexion_kb_max_files: int = 50
    """Max KB files to load in one reflexion pass."""
    reflexion_kb_llm_consult: bool = True
    """Allow an OMP LLM consultation over KB snippets on first attempt."""
    reflexion_strategy_space: List[str] = Field(
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
    push_to_hub: Optional[str] = None
    """Target repo id to push to (None/empty = don't push)."""
    hub_token: Optional[str] = None
    """HF Hub access token (else pulled from `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`)."""
    hub_private: bool = True
    """Create the target hub repo as private if it does not exist."""

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    seed: int = 42
    """Global RNG seed."""

    model_config = {"extra": "ignore"}


def _flatten_nested(data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten ``judge:`` and ``reflexion:`` sub-mappings into top-level keys.

    Allows YAML configs to use either::
        judge:
          enabled: true
          model: foo
    or the flat form:: judge_enabled: true
    """
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if key in ("judge", "reflexion") and isinstance(value, dict):
            for sub_key, sub_val in value.items():
                out[f"{key}_{sub_key}"] = sub_val
        else:
            out[key] = value
    return out


def load_config(path: Union[str, Path]) -> ModelConfig:
    """Load a :class:`ModelConfig` from a YAML file at ``path``.

    Raises:
        FileNotFoundError: if ``path`` does not exist (with a hint listing
            the ``models/`` directory when it sits next to the config).
        ValueError: if the YAML is malformed or not a mapping.
        pydantic.ValidationError: if the parsed data fails validation.
    """
    p = Path(path)
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


def dump_config(config: ModelConfig, path: Union[str, Path]) -> None:
    """Serialize ``config`` to YAML at ``path`` (parent dirs are created)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = config.model_dump()
    except AttributeError:
        # pydantic v1 fallback
        data = config.dict()

    try:
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                data, f, sort_keys=False, default_flow_style=False, allow_unicode=True
            )
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Failed to write config to {p}: {exc}") from exc
