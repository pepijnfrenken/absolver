"""Minimal OpenAI-compatible chat-completions client for the JUDGE/REFLEXION nodes.

Replaces the old `omp -p --model ...` subprocess dependency: OMP is a full
agent harness that is not installed inside Modal containers, so the judge
silently fell back to keyword scoring. A direct HTTP call to a free
OpenAI-compatible endpoint (FreeInference, Featherless, ...) works anywhere
`urllib` exists — i.e. everywhere.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://freeinference.org/v1"


def _resolve_key(cfg: Any) -> str | None:
    """Resolve the API key: config field first, then env vars."""
    key = getattr(cfg, "judge_api_key", None)
    if key:
        return key
    for env_name in ("JUDGE_API_KEY", "FREEINFERENCE_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(env_name):
            return os.environ[env_name]
    return None


def chat_completion(
    prompt: str,
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: int = 90,
) -> str:
    """POST a single-turn chat completion; return the assistant text.

    Raises on HTTP/transport errors so callers can fall back to keyword
    scoring, mirroring the old OMP behaviour.
    """
    base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    api_key = api_key or _resolve_key(None)

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Cloudflare on freeinference.org 403s urllib's default
        # "Python-urllib/3.x" user-agent (error 1010); impersonate curl.
        "User-Agent": "curl/8.5.0",
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    # Retry transient failures (429 rate-limit, 5xx, transport errors) with
    # exponential backoff. The free endpoint's in-flight ceiling is low, so
    # a 429 is expected under load; without this, every 429 degrades the
    # judge to keyword scoring. A config timeout/SSL/auth error still raises.
    max_attempts = 3
    backoff = (1.0, 2.0, 4.0)
    payload: Any = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            retryable = exc.code == 429 or exc.code in (500, 502, 503, 504)
            if retryable and attempt < max_attempts:
                wait = backoff[attempt - 1]
                try:
                    retry_after = exc.headers.get("Retry-After")
                    if retry_after:
                        wait = max(wait, float(retry_after))
                except (TypeError, ValueError):
                    pass
                logger.warning(
                    "judge API HTTP %s (attempt %d/%d); backing off %.1fs",
                    exc.code, attempt, max_attempts, wait,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"judge API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt < max_attempts:
                wait = backoff[attempt - 1]
                logger.warning(
                    "judge API unreachable (attempt %d/%d): %s; retrying in %.1fs",
                    attempt, max_attempts, exc.reason, wait,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"judge API unreachable: {exc.reason}") from exc
    assert payload is not None, "judge API failed after retries"

    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"judge API bad payload: {str(payload)[:300]}") from exc
