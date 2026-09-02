"""Generate benign-coherence samples for the model card: ablated vs pristine.

Chat template, user-only turn (same shape the collect gates used), default
greedy decoding (do_sample=False) for apples-to-apples with gate transcripts.
"""
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain the process of photosynthesis in plants.",
    "Write a short poem about a sunset.",
    "What is C. elegans?",
    "Explain the difference between TCP and UDP.",
    "Describe the plot of Hamlet.",
]
MODELS = [
    ("ablated", "campaigns/lfm2.5-recovery/recovered-r1"),
    ("pristine", "LiquidAI/LFM2.5-1.2B-Instruct"),
]

OUT = "campaigns/lfm2.5-recovery/benign-samples.json"

results = {}
try:
    with open(OUT) as f:
        results = json.load(f)
except FileNotFoundError:
    pass


def save():
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)


for name, path in MODELS:
    have = results.get(name, {})
    todo = [p for p in PROMPTS if p not in have]
    if not todo:
        print(f"[{name}] already complete, skipping", flush=True)
        continue
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    model.eval()
    for prompt in todo:
        # byte-identical to the collect gates' `formatted` field
        text = (
            "<|startoftext|><|im_start|>user\n"
            + prompt
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        input_ids = tok(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=256, do_sample=False)
        resp = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        results.setdefault(name, {})[prompt] = resp
        print(f"[{name}] {prompt[:45]!r} -> {len(resp)} chars", flush=True)
        save()
    print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)

save()
print(f"saved {OUT}")