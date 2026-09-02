---
campaign_id: <model-slug>-<date>
target_model: <org/model>
arch: <dense_llama | moe | ...> (<n> layers, hidden <h>, note non-square weights)
date: <YYYY-MM-DD>
status: <POSITIVE — gates passed | NEGATIVE — no config passes | PARTIAL — describe>
hardware: <Modal L4 | local CPU | ...>
cost: <approx $ / GPU-hrs>
methods_tried: [<advanced | mpoa | stacked_ablation | bias_vectors | lora | steering | direct_ablation>]
dir_methods_tried: [diff_means, paired, svd, leace, whitened_svd]
verdict_summary: >
  <2-3 sentence honest summary. If NEGATIVE: what was tried, what the
  blocker was, what the causal test showed. If POSITIVE: the winning
  config and gate numbers.>
key_numeric_results:
  pristine_refusal: <0-1>
  gate_refusal_ablated: <0-1>
  gate_coherence_ablated: <0-1>
  gate_capability_ablated: <0-1>
  best_config: {method: <>, dir_method: <>, alpha: <>, layers: <>, weights: <>}
bugs_found:
  - <bug_slug>   # one per bug: what silently corrupted results
recommended_next: [<bias_vector_residual_hook | larger_model | lora_finetune | ...>]
---

# Campaign: <Model Name>

## TL;DR
<2-3 sentences. Positive = the recipe. Negative = the blocker + evidence it's real.

## Why this model
<What made it a good target (size, arch, tuning, goal).>

## What happened (the honest arc)
<Numbered narrative: what was tried, in order, WITH the failures and false
starts. The false starts are the value — they're what the next campaign
won't repeat. Include: the config that looked good but wasn't, the causal
test that reframed it, the geometry/measurement insight.>

### N. <milestone heading>
<Per-milestone detail: method, config, result table, what it taught.>

## Bugs found & fixed (or still open)
| Bug | Mechanism | Consequence | Status |
|---|---|---|---|
| <name> | <mechanism> | <what it corrupted> | fixed `<commit>` / OPEN |

## What the NEXT campaign on this model should try first
<Ordered, with reasoning. The steering test result often dictates this.>

## Key-numbers cheat-sheet
| Metric | Value |
|---|---|

---

## Hard-won rules (read before designing a config)

- **LFM2.5-hybrid: refusal spans ALL out-projections** (attn `out_proj` ×6,
  conv `out_proj` ×10, ffn `w2` ×16 = 32). Coverage > magnitude — huihui's
  published abliteration projects all 32 at rel_l2 0.022-0.030 and passes at
  default greedy; a 6-attention-only edit at 2× strength still refuses.
  `conv.out_proj` is a 2D hidden Linear and IS projectable — never
  blanket-exclude a projection class without a per-arch shape check.
- **Projection direction space is the OUTPUT space**: W[out, in] -= d(d^T W)
  needs d.shape[0] == out. ffn `w2` is [hidden, inter] (non-square on
  LFM2.5, measured [2048, 8192]) and still projects. Resolver weight names:
  `o_proj` (attn), `conv_out` (conv block, 2D-square gated), `w2`/`ffn_out`
  (ffn). Run `inspect` — it prints the resolved-coverage count.

## Template notes
- The YAML frontmatter is the MACHINE-READABLE part — keep every field
  filled so `harness/` scripts can mine correlations across campaigns.
- The narrative is the HUMAN-READABLE part — the false starts and the
  "why" are worth more than the final config.
- `status` drives the index: POSITIVE campaigns are the recipe source;
  NEGATIVE campaigns are the landmine map.
- Copy this file to `campaigns/<model-slug>/README.md` and fill it in as
  you go (not after — you'll forget the false starts).
