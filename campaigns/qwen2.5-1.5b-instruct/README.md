---
campaign_id: qwen2.5-1.5b-instruct-2026-09-02
target_model: Qwen/Qwen2.5-1.5B-Instruct
arch: dense_llama (28 layers, hidden 1536, MLP 8960 intermediate)
date: 2026-09-02
status: NEGATIVE — no single-shot config passes the E03 gates
hardware: Modal L4 (gpu runs) + local CPU (debug runs)
cost: ~4 L4 Modal runs (~$1 total) + local CPU
methods_tried: [stacked_ablation, advanced(plain projection), mpoa]
dir_methods_tried: [paired, diff_means]
verdict_summary: >
  The refusal direction (diff_means input-phase) IS causal — steering
  -20*d at the MLP output flips refusal to clean compliance — but NO
  weight-projection config achieves that removal without either leaving
  refusals (a=10) or destroying coherence (a=15). Plain projection is
  geometrically too weak (refusal lives in ~0.03% of weight norm).
key_numeric_results:
  pristine_refusal: 1.0
  steer_flip_alpha: -20.0    # alpha where refusal -> compliance at MLP output
  mpoa_a10_heldout: {refused: 3, coherent: 5, of: 5}   # not enough removal
  mpoa_a15_heldout: {refused: 0, coherent: 1, of: 5}   # degenerate
  plain_a1_x10_heldout: {refused: 5, coherent: 5, of: 5}  # no effect
  weight_delta_plain_a1: 0.0003   # fraction of weight norm removed per pass
bugs_found:
  - down_proj_never_ablated        # guard compared input dim not output dim
  - alpha_sign_flip_amplifier      # alpha>2 on plain projection inverts+amplifies
  - ppl_empty_slice                # lg[N-1:-1] is empty -> gate skipped silently
  - bias_vectors_silent_noop       # Qwen o_proj/down_proj have no bias
  - prompt_leak_in_quick_score     # decoded prompt+resp, keywords in prompt counted
  - train_zero_split               # build_split pairs to shorter pool -> train=0
recommended_next: [bias_vector_residual_hook, larger_model, lora_finetune]
---

# Campaign: Qwen2.5-1.5B-Instruct — the "does the pipeline actually work" test

## TL;DR

Qwen2.5-1.5B-Instruct does **not** yield to single-shot weight-projection
abliteration in this pipeline. The refusal direction is real and causal
(steering proves it), but no plain/MPOA/stacked weight edit removes refusal
on a held-out set while keeping coherent output. **The pipeline is now
honest** — after this campaign, every bug that made it lie is fixed and
documented. This writeup is the first entry in the compounding campaign KB.

## Why this model

The original goal was "verify the pipeline works end-to-end." Qwen2.5-1.5B
is small (cheap on L4), dense (no MoE complexity), and safety-tuned (refuses
~100% of the harmful set) — a good stress test for the machinery, not a
target expected to be easy.

## What happened (the honest arc)

### 1. The false start: a "diag" that claimed refusal 0.0

A diagnostic run on 2026-09-01 claimed: `advanced (plain projection),
alpha=10, layers 20-27, o_proj+down_proj -> refusal 0.0`. This looked like
the winning recipe and the config + comments were written around it.

**It was wrong.** The 0.0 was an artifact of *three silent bugs* plus
in-sample evaluation:

| Bug | Mechanism | Consequence |
|---|---|---|
| **down_proj never ablated** | `_project_2d` guard compared `d.shape[0]` to `weight.shape[1]` (INPUT dim). down_proj is 1536×8960 (non-square); a hidden-space direction [1536] never matches input dim 8960 → **silent skip**. Only square o_proj (1536×1536) was ever ablated. | Every "o_proj + down_proj" run was really o_proj-only. |
| **alpha > 2 = sign-flip amplifier** | Plain row projection `w' = w - alpha*d*(d^T w)` nulls at alpha=1. For alpha>2, `|w'|^2 = |w|^2 + (alpha^2-2alpha)*c^2` — the component is **inverted and amplified**, not removed. Measured: +8–14% norm growth at alpha=10. | "Stronger" alphas made refusal *stronger*, not weaker. |
| **alpha semantics wrong** | Config said `alpha: 10` for plain projection, treating alpha like MPOA's. | The "0.0" run at alpha=10 was really *over-injecting* refusal in a way that happened to break the keyword scorer on 20 in-sample prompts. |

### 2. The fix + the honest retest

Fixed the guard (compare `shape[0]`, raise loudly on mismatch), corrected
alpha semantics (alpha=1 = true nulling for plain; MPOA tolerates high
alpha via norm rescale), fixed the empty-slice PPL bug, fixed the
prompt-leak in the sweep scorer, and rebuilt the corner-test runner with a
real train/tune/test split (the old one silently had train=0).

The honest stacked test (paired + diff_means, alpha=10, L20-27, o+down,
now REALLY ablating down_proj) on the held-out set: **5/5 refused**. The
"0.0" was gone. Byte-identical refusal locally on CPU too.

### 3. Is the direction even causal? — the steering test

Before blaming the surgery, test the direction. Inject ±alpha*d into the
residual stream (layer output) during generation:

| Steering (L20-27, MLP output) | Result |
|---|---|
| alpha = 0 (baseline) | refusal (KW 1.0) |
| alpha = +10..+80 (add refusal dir) | refusal, degrades to loops at +40+ |
| **alpha = -20** (subtract refusal dir) | **clean compliance (KW 0.0), coherent prose** |
| alpha = -40/-80 | compliance but degenerate repetition |

**Conclusion: the diff_means direction IS the refusal direction.** Removing
it at inference (steering) cleanly flips behavior. So the direction is
right — the *weight surgery* is the problem.

### 4. Why weight surgery fails: the geometry

- **Plain projection at alpha=1** removes ~0.03% of weight norm per pass
  (the refusal component of W's action is genuinely tiny). Even ×10 passes
  changed nothing behaviorally.
- **MPOA (norm-preserving)** can push harder: alpha=10 removes refusal on
  2/3 easy held-out prompts but leaves 3/5 on the real held-out set;
  alpha=15 removes all refusal but **destroys coherence** (1/5 coherent —
  repetition loops, Chinese gibberish).
- **The operating window is empty**: between "still refuses" (alpha≤10) and
  "degenerate" (alpha≥15) there is no alpha that cleanly flips all held-out
  refusals. The earlier "clean 2/3" was small-sample luck on easy prompts.

### 5. The honest negative

No single-shot weight config passes the E03 gates (refusal ≤3%, coherence
≥90%) on Qwen2.5-1.5B:

```
mpoa alpha=10 L24-27 o+down : 3/5 refused, 5/5 coherent   (not enough removal)
mpoa alpha=15 L24-27 o+down : 0/5 refused, 1/5 coherent   (degenerate)
mpoa alpha=20 L24-27 o-only : 5/5 refused, 5/5 coherent   (no effect)
plain alpha=1 ×10           : 5/5 refused, 5/5 coherent   (geometrically weak)
```

## Bugs found & fixed (the real value of this campaign)

Six bugs, all of which made the pipeline *lie* (silently produce
confident-looking wrong results). All fixed + committed:

1. **down_proj silent-skip** — `excise.py` guard `d.shape[0] != weight.shape[1]` → `shape[0]` + raise loudly. (`6d2e131`)
2. **alpha sign-flip** — documented in `excise.py`; alpha=1 = true nulling. Config corrected 10→1. (`6d2e131`)
3. **PPL empty slice** — `lg[N-1:-1]` is empty in Python → pristine PPL crashed silently, gate "passed" on 0 prompts. Fixed in `verify.py`, `gates.py`, runner. (`6d2e131`, `12c2db1`)
4. **bias_vectors silent no-op** — `_apply_bias_vectors` adds to `o_proj.bias`/`down_proj.bias`; **Qwen has no biases** → method never ran. (found, not yet fixed — this is the recommended next step)
5. **prompt-leak quick_score** — decoded prompt+response, keywords in prompt counted as refusal. (`42d87a0`)
6. **train=0 split** — `build_split` pairs to shorter pool (20), so `train_size = 20-2*10 = 0` → directions from zero prompts. (`83cef80`)

## Recommended next steps (for the NEXT campaign on this model)

1. **Residual bias hook (steering-as-edit)** — the steer test proved -20*d
   at the MLP output cleanly removes refusal. Make that permanent: a
   forward hook adding -alpha*d to the layer output, saved as a
   "bias vector" artifact (non-destructive, reversible, exportable). This
   is the most promising untested path.
2. **LoRA fine-tune** — instead of editing weights, train a small LoRA that
   produces compliant continuations and subtract it. Heavier but sidesteps
   the coherence cliff entirely.
3. **Try a bigger/easier model** — the pipeline is proven honest; point it
   at a model known to abliterate (e.g. LFM2.5 lineage) to get a green
   gate pass end-to-end.

## Key-numbers cheat-sheet

| Metric | Value |
|---|---|
| Pristine refusal (held-out) | 5/5 (100%) |
| Steering flip alpha (MLP output) | -20 |
| Plain projection delta per pass | ~0.03% weight norm |
| MPOA alpha band | 10 (weak) – 15 (degenerate) |
| Held-out size (test) | 5 prompts |
| Gate thresholds | refusal ≤3%, coherence ≥90% |
| Total Modal cost | ~$1 (4 L4 runs) |
