#!/usr/bin/env python3
"""Standalone generation transcript collector for the LFM2.5 campaign.

Both models (pristine + ablated), two prompt flavors (raw string vs
chat-templated), harmful held-out + benign prompts. Writes the transcript
to <campaign>/generation-transcript.txt.

Note: this is a STOPGAP the campaign needed and the harness `collect`
command lacks — see TOOLKIT-FEEDBACK.md (generation-transcript collector).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_format import detect_prompt_format, format_prompt
from eval_split import build_split
from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
from verify import _decode_continuation

CAMPAIGN = Path(__file__).resolve().parent
ABLATED_DIR = CAMPAIGN / "abl-mpoa-paired-a2.0-all6attn-n20"
BASE = "LiquidAI/LFM2.5-1.2B-Instruct"
MAX_NEW = 64

h = list(DEFAULT_HARMFUL)
g = list(DEFAULT_HARMLESS)
n = min(len(h), len(g))
split = build_split(h[:n], g[:n], train_size=n - 10, tune_size=5, test_size=5, seed="absolver:qwen25:v1")
harmful = list(split.test)
benign = list(DEFAULT_HARMLESS)[:3]

lines: list[str] = []


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def gen(m, t, prompt: str, chat: bool) -> str:
    fmt = detect_prompt_format(t, "auto")
    text = format_prompt(t, prompt, fmt) if chat else prompt
    inp = t(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        raw = m.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=t.eos_token_id)
    return _decode_continuation(t, raw, inp["input_ids"])


def run_model(name: str, mid: str) -> None:
    out(f"\n{'=' * 78}\n### {name}\n{'=' * 78}")
    m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True, trust_remote_code=True)
    t = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    for flavor, chat in (("RAW", False), ("CHAT", True)):
        out(f"\n--- {flavor} prompts ---")
        for p in harmful:
            out(f"\n[harmful] {p}")
            out(f"  -> {gen(m, t, p, chat)!r}")
        for p in benign:
            out(f"\n[benign ] {p}")
            out(f"  -> {gen(m, t, p, chat)!r}")


def main() -> int:
    out("# LFM2.5-1.2B-Instruct — generation transcript (pristine vs ablated)")
    out(f"# ablated: {ABLATED_DIR.name} (mpoa, paired, alpha 2.0, layers 2,5,8,10,12,14, o_proj, n=20)")
    out(f"# max_new_tokens={MAX_NEW}, greedy; held-out harmful set == the collect gate set\n")
    out("## Harmful held-out prompts")
    for p in harmful:
        out(f"  - {p}")
    out("\n## Benign prompts")
    for p in benign:
        out(f"  - {p}")
    run_model("PRISTINE", BASE)
    run_model("ABLATED (n20)", str(ABLATED_DIR))
    (CAMPAIGN / "generation-transcript.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nTranscript -> {CAMPAIGN / 'generation-transcript.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
