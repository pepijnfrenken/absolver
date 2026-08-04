"""Prompt formatting for model generation — model-agnostic.

Instead of feeding raw prompt strings to whatever model is loaded, this
module inspects the *model's own tokenizer* and formats prompts the way
the model expects:

1. If the tokenizer has a ``chat_template`` (ChatML, Llama-3, Qwen, etc.)
   use ``apply_chat_template()`` with a generation prompt.
2. If the tokenizer uses thinking/CoT markers (``<think>``/``</think>``,
   e.g. MiniCPM5, DeepSeek-R1 style) keep the thinking wrapper that the
   chat template emits.
3. Fall back to the raw prompt string for base/plain models.

Behaviour is controlled by ``prompt_format``:
  - ``auto``   (default) detect from the tokenizer
  - ``chat``   force chat-template formatting (error if none)
  - ``raw``    force raw prompt (no template)
  - ``thinking`` force a thinking wrapper even if the chat template does
                 not emit one (best-effort; only used when no template).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _has_chat_template(tok: Any) -> bool:
    """True if the tokenizer exposes a usable chat template."""
    ct = getattr(tok, "chat_template", None)
    if ct:
        return True
    # Some tokenizers keep it under a different attribute.
    for attr in ("apply_chat_template", "_chat_template"):
        if hasattr(tok, attr):
            return True
    return False


def _has_thinking_tokens(tok: Any) -> bool:
    """True if the tokenizer knows thinking-mode markers."""
    try:
        vocab = getattr(tok, "get_vocab", lambda: {})()
    except Exception:
        vocab = {}
    return THINK_OPEN in vocab or THINK_CLOSE in vocab


def detect_prompt_format(tok: Any, override: str | None = None) -> str:
    """Resolve the effective prompt format for a tokenizer.

    ``override`` maps to config.prompt_format (auto/chat/raw/thinking).
    Returns one of those four strings.
    """
    fmt = (override or "auto").lower()
    if fmt != "auto":
        return fmt
    if _has_chat_template(tok):
        return "chat"
    if _has_thinking_tokens(tok):
        return "thinking"
    return "raw"


def format_prompt(tok: Any, prompt: str, fmt: str = "auto") -> str:
    """Format a single user prompt for generation with the given tokenizer.

    Returns a string ready to be tokenized. Raises ValueError when
    ``fmt=chat`` is requested but the tokenizer has no chat template.
    """
    fmt = detect_prompt_format(tok, fmt)

    if fmt == "chat":
        if not _has_chat_template(tok):
            raise ValueError(
                "prompt_format=chat requested but tokenizer has no chat template; "
                "set prompt_format: auto or raw"
            )
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            logger.warning("apply_chat_template failed (%s); raw fallback", exc)
            return prompt

    if fmt == "thinking":
        # Best-effort thinking wrapper for models that use <think> markers
        # but ship no chat template.
        return f"{THINK_OPEN}\n{prompt}\n{THINK_CLOSE}"

    return prompt
