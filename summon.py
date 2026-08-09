"""SUMMON node: model + tokenizer load, experience-DB query, arch detection.

Loads the configured HuggingFace model (trying a diffusion text encoder
first when configured, then a causal LM, then a seq2seq LM), inspects its
architecture, queries the experience database for known-good
hyperparameters, and returns the loaded model + descriptor fields that
downstream nodes (PROBE / DISTILL / EXCISE) consume.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import torch

from config import ModelConfig
from detector import detect_architecture
from experience import ExperienceDB
from state import AbliterationState

logger = logging.getLogger(__name__)

# transformers / diffusers are required at runtime but the module must still
# import (e.g. for py_compile) even if they are absent in a stripped env.
try:
    import transformers  # noqa: F401
    from transformers import AutoTokenizer  # noqa: F401
except Exception:  # pragma: no cover - env without transformers
    transformers = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

try:
    import diffusers  # noqa: F401
except Exception:  # pragma: no cover - env without diffusers
    diffusers = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Model-load retry
# ---------------------------------------------------------------------------
# ``summon_node`` hits the network/hub the first time a model is downloaded.
# A flaky connection can surface as OSError / ConnectionError / RuntimeError
# ("[Errno 104] Connection reset by peer", "Failed to resolve model id", ...)
# that are transient and worth a bounded retry. OOM (MemoryError) and
# KeyboardInterrupt are NOT transient and are intentionally NOT caught, so
# they still crash loudly.
_TRANSIENT_LOAD_ERRORS = (OSError, ConnectionError, RuntimeError)
_LOAD_ATTEMPTS = 3
_LOAD_BACKOFF = (2.0, 4.0)


def _load_with_retry(load_fn: Callable[[], Any]) -> Any:
    """Run ``load_fn`` with ``_LOAD_ATTEMPTS`` tries and exponential backoff.

    Catches only transient download/load errors (OSError, ConnectionError,
    RuntimeError); re-raises the last error after all attempts. Other
    exceptions (e.g. a bad config ValueError) propagate immediately so the
    caller's backend fallback still records them. MemoryError / KeyboardInterrupt
    are never caught.
    """
    attempt = 1
    last: Exception | None = None
    while True:
        try:
            return load_fn()
        except _TRANSIENT_LOAD_ERRORS as exc:
            last = exc
            if attempt >= _LOAD_ATTEMPTS:
                break
            wait = (
                _LOAD_BACKOFF[attempt - 1]
                if (attempt - 1) < len(_LOAD_BACKOFF)
                else _LOAD_BACKOFF[-1]
            )
            logger.warning(
                "model load step (%s) hit transient %s on attempt %d/%d: "
                "%s; retrying in %.1fs",
                "from_pretrained", type(exc).__name__, attempt, _LOAD_ATTEMPTS,
                exc, wait,
            )
            time.sleep(wait)
            attempt += 1
    assert last is not None, "unreachable: loop must break or return"
    raise last


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Tokenizer-only subset of the model-load kwargs. Quantization flags and
# offload folders are not AutoTokenizer arguments and would raise.
_TOK_KWARG_KEYS = (
    "torch_dtype",
    "device_map",
    "trust_remote_code",
    "revision",
    "use_fast",
    "padding_side",
    "truncation",
    "variant",
    "cache_dir",
)


def _tokenizer_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter model-load kwargs down to what ``AutoTokenizer.from_pretrained``
    will actually accept."""
    return {k: v for k, v in kwargs.items() if k in _TOK_KWARG_KEYS}


def send_marimo_toast(title: str, subtitle: str = "") -> None:
    """Best-effort marimo toast notification.

    marimo is an optional dependency; if it (or its toast API) is unavailable,
    we simply log the message instead. Never raises.
    """
    try:
        import marimo as mo  # type: ignore

        status = getattr(mo, "status", None)
        if status is not None and hasattr(status, "toast"):
            status.toast(title + (f": {subtitle}" if subtitle else ""))
            return
        # Older / alternative marimo surfaces — best effort only.
        if hasattr(mo, "toast"):
            mo.toast(title)  # type: ignore[attr-defined]
            return
    except Exception:
        pass
    logger.info("SUMMON toast: %s | %s", title, subtitle)


def _merge_prior_into_config(cfg: ModelConfig, prior: dict[str, Any]) -> None:
    """Apply prior-experience knobs to ``cfg`` ONLY for fields the user did
    not explicitly set. Explicit user choices (notably ``target_layers``) are
    never clobbered.

    Uses pydantic v2's ``model_fields_set`` when available to detect which
    fields were explicitly provided; falls back to "empty target_layers"
    semantics otherwise.
    """
    if not prior:
        return

    explicit = set(getattr(cfg, "model_fields_set", None) or [])

    def _maybe(field: str, value: Any) -> None:
        if value is None:
            return
        if field in explicit:
            return
        current = getattr(cfg, field, None)
        # Treat empty list as "unset" for list-valued fields.
        if isinstance(current, list) and not current and isinstance(value, list):
            setattr(cfg, field, list(value))
            return
        # Don't overwrite a truthy current value the user got via default.
        if current in (None, "", 0, 0.0):
            setattr(cfg, field, value)

    _maybe("method", prior.get("method"))
    _maybe("dir_method", prior.get("dir_method"))
    _maybe("alpha", prior.get("alpha"))
    _maybe("passes", prior.get("passes"))
    _maybe("target_layers", prior.get("target_layers"))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def summon_node(state: AbliterationState) -> dict:
    """Load model + tokenizer, detect architecture, return SUMMON state slice.

    Returns a partial state dict with: model_loaded, model_obj, tokenizer,
    experience_db, architecture, hidden_size, num_layers, num_experts,
    layer_types, target_weights.
    """
    cfg: ModelConfig = state["config"]

    # 1. Init experience DB.
    db = ExperienceDB(cfg.reflexion_db_path)

    # 2. Query DB for prior attempts.
    prior: dict[str, Any] | None = db.query_best_method(cfg.model_id)
    if not prior:
        # similar_arch needs a hidden_size hint, which ModelConfig doesn't
        # carry pre-load. Try defensively; if absent, skip.
        hidden_hint = getattr(cfg, "hidden_size", None)
        if hidden_hint:
            try:
                prior = db.query_similar_arch(cfg.model_arch, hidden_hint)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("query_similar_arch failed: %s", exc)
                prior = None
    if prior:
        logger.info("Found prior experience: %s", prior)
        _merge_prior_into_config(cfg, prior)

    # 3. Load model.
    dtype = getattr(torch, cfg.dtype, None)
    kwargs: dict[str, Any] = {"trust_remote_code": cfg.trust_remote_code}
    use_device_map = cfg.device and cfg.device != "auto"
    if use_device_map:
        kwargs["device_map"] = cfg.device
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if cfg.quantize:
        try:
            from transformers import BitsAndBytesConfig
            if cfg.quantize == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                )
            elif cfg.quantize == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            logger.warning(
                "quantize=%s requested but BitsAndBytesConfig unavailable; "
                "loading unquantized.",
                cfg.quantize,
            )
    if cfg.offload_folder:
        kwargs["offload_folder"] = cfg.offload_folder
    if cfg.cache_dir:
        kwargs["cache_dir"] = cfg.cache_dir
    if cfg.variant:
        kwargs["variant"] = cfg.variant
    if cfg.revision:
        kwargs["revision"] = cfg.revision

    tok_kwargs = _tokenizer_kwargs(kwargs)

    model = None
    tokenizer = None
    is_diffusion = False
    errors = []

    # 3a. Try as a diffusion text encoder first (only when requested).
    #     diffusers ships NO Flux2KleinPipeline; for diffusion models we
    #     modify the causal-LM text encoder that lives under text_encoder/.
    if getattr(cfg, "model_arch", None) == "diffusion_encoder":
        try:
            from transformers import AutoModelForCausalLM

            model = _load_with_retry(
                lambda: AutoModelForCausalLM.from_pretrained(
                    cfg.model_id, subfolder="text_encoder", **kwargs
                )
            )
            if AutoTokenizer is not None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        cfg.model_id, subfolder="tokenizer", **tok_kwargs
                    )
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(
                        cfg.model_id, **tok_kwargs
                    )
            is_diffusion = True
            logger.info(
                "Loaded %s as diffusion text encoder (text_encoder subfolder).",
                cfg.model_id,
            )
        except Exception as exc:
            errors.append(f"diffusion: {exc}")

    # 3b. Try as a standard causal LM.
    if model is None:
        try:
            from transformers import AutoModelForCausalLM

            model = _load_with_retry(
                lambda: AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
            )
            if AutoTokenizer is not None:
                tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, **tok_kwargs)
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
            logger.info("Loaded %s as causal LM.", cfg.model_id)
        except Exception as exc:
            errors.append(f"lm: {exc}")

    # 3c. Try as a seq2seq LM.
    if model is None:
        try:
            from transformers import AutoModelForSeq2SeqLM

            model = _load_with_retry(
                lambda: AutoModelForSeq2SeqLM.from_pretrained(cfg.model_id, **kwargs)
            )
            if AutoTokenizer is not None:
                tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, **tok_kwargs)
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
            logger.info("Loaded %s as seq2seq LM.", cfg.model_id)
        except Exception as exc:
            errors.append(f"seq2seq: {exc}")

    if model is None:
        raise RuntimeError(
            "Could not load model from any backend: " + "; ".join(errors)
        )

    if tokenizer is None and AutoTokenizer is not None:
        # Last-resort tokenizer fetch from the model id.
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, **tok_kwargs)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as exc:
            logger.warning("Tokenizer load failed: %s", exc)

    # 4. Detect architecture.
    arch_info = detect_architecture(model)
    arch_info.pop("text_encoder_model", None)

    # If we loaded via the diffusion path, the detector sees the bare text
    # encoder as a dense decoder. Re-label so downstream nodes treat it as a
    # diffusion text encoder.
    if is_diffusion:
        arch_info["architecture"] = "diffusion_encoder"

    # 5. Best-effort marimo toast.
    send_marimo_toast(
        f"Absolver loaded {cfg.model_id}",
        f"Arch: {arch_info.get('architecture')}, "
        f"{arch_info.get('num_layers')} layers, "
        f"{arch_info.get('hidden_size')} hidden",
    )

    logger.info(
        "SUMMON complete: arch=%s layers=%s hidden=%s experts=%s",
        arch_info.get("architecture"),
        arch_info.get("num_layers"),
        arch_info.get("hidden_size"),
        arch_info.get("num_experts"),
    )

    # Store model & tokenizer in registry (bypasses LangGraph serialization).
    # Move to GPU when device_map wasn't used.
    if not use_device_map and torch.cuda.is_available():
        model = model.cuda()
    from model_registry import set_model, set_tokenizer
    set_model(model)
    set_tokenizer(tokenizer)
    import os; print(f"SUMMON pid={os.getpid()} model={model is not None}")

    # Honor an explicit config target_weights (e.g. the LFM2.5 winning
    # recipe uses o_proj ONLY); arch detection is the fallback.
    cfg_tw = getattr(state.get("config"), "target_weights", None) if state.get("config") else None
    target_weights = (
        list(cfg_tw) if cfg_tw
        else arch_info.get("target_weights", ["o_proj", "down_proj"])
    )
    return {
        "model_loaded": True,
        "architecture": arch_info.get("architecture"),
        "hidden_size": arch_info.get("hidden_size"),
        "num_layers": arch_info.get("num_layers"),
        "num_experts": arch_info.get("num_experts"),
        "layer_types": arch_info.get("layer_types"),
        "target_weights": target_weights,
    }
