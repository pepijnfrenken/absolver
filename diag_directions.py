"""Diagnose why Qwen2.5-1.5B sweep shows refusal=0.25/kl=0.000 everywhere.

Loads the model, collects paired directions exactly like probe.py, prints
per-layer direction norms, then applies an MPOA projection at alpha=2.0 and
re-measures refusal + KL on the same 20 prompts. If directions are degenerate
(norm ~0) or projection does nothing (KL ~0), we've found the bug.
"""
import os, sys
os.chdir("/absolver")
sys.path.insert(0, "/absolver")

import torch
from config import load_config
from summon import summon_node
from model_registry import get_model, get_tokenizer
from probe import _collect_paired_output_phase
from sweep import _apply_candidate, _quick_score
from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS

cfg = load_config("models/qwen2.5-1.5b-instruct.yaml")

from state import AbliterationState
_summon_state: AbliterationState = {
    "config": cfg, "model_loaded": False, "architecture": None,
    "hidden_size": None, "num_layers": None,
}
summon_node(_summon_state)
model = get_model()
tok = get_tokenizer()
device = next(model.parameters()).device
print("device:", device)

layers = model.model.layers if hasattr(model, "model") else model.layers
n_layers = len(layers)
print("n_layers:", n_layers)

# --- collect paired directions exactly like the probe does ---
harmful = list(DEFAULT_HARMFUL)[:20]
harmless = list(DEFAULT_HARMLESS)[:20]
print("harms:", len(harmful), "harmless:", len(harmless))

refusal_acts, affirm_acts = _collect_paired_output_phase(
    model, tok, harmful, layers, n_layers, device,
    prefill=cfg.paired_prefill, max_new_tokens=cfg.paired_max_new_tokens)

# build direction dict (diff of means, normalized per layer)
directions = {}
for i in range(n_layers):
    if i not in refusal_acts or i not in affirm_acts:
        continue
    r = torch.stack(refusal_acts[i]).mean(0).float()
    a = torch.stack(affirm_acts[i]).mean(0).float()
    d = r - a
    norm = d.norm().item()
    if norm < 1e-8:
        directions[i] = torch.zeros_like(d)
        print(f"layer {i}: ZERO direction")
        continue
    directions[i] = d / norm
    print(f"layer {i}: dir norm={norm:.4f}")

# --- baseline refusal on harmful prompts (before ablation) ---
score_cfg = type("S", (), {"max_seq_len": 1024})()
base = _quick_score(model, tok, score_cfg, harmful, max_tokens=48)
print("\nBASELINE refusal:", base["refusal"], "quality:", base["quality"])

# --- apply MPOA at alpha=2.0 on tail layers, re-score ---
cand = {"method": "mpoa", "target_layers": [26, 27], "target_weights": ["o_proj"],
        "alpha": 2.0, "passes": 1}
_apply_candidate(model, directions, None, cand, None)
after = _quick_score(model, tok, score_cfg, harmful, max_tokens=48)
print("\nAFTER MPOA alpha=2.0 on [26,27]:")
print("  refusal:", after["refusal"], "quality:", after["quality"], "kl:", after["kl"])

# KL vs pristine harmless (need base logprobs; quick proxy: just length)
print("\nDONE")
