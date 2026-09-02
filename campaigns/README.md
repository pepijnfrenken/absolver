# Campaign KB — the compounding abliteration library

Every attempt to abliterate a model is a **campaign**: a folder with one
README (this structure) that records the honest arc — what was tried, what
looked good but wasn't, what the causal tests showed, what bugs were found,
and what the next campaign should try first.

The point is **compounding**: the next model is easier because the previous
campaigns are readable. Methods that worked, alphas that flipped behavior,
architectures where down_proj silently no-ops — all recorded once, mined
forever.

## Layout

```
campaigns/
  <model-slug>/
    README.md          # the campaign writeup (YAML frontmatter + narrative)
  templates/
    campaign-template.md   # copy this to start a new campaign
```

## Index of campaigns

| Campaign | Model | Date | Status | TL;DR |
|---|---|---|---|---|
| [lfm2.5-flavorfix-reruns](lfm2.5-flavorfix-reruns/README.md) | LFM2.5 family ×3 | 2026-09-02 | NEGATIVE | Flavor fix shipped (TOOLKIT-FEEDBACK 1-7) but refuted as root cause: chat-flavored paired directions don't pass on 1.2B-Instruct (3/5→4/5), flip 350M (4/5→1/5) into pseudo-guide collapse, never touch Thinking's trace refusals; steer-test proves the attention-layer window is empty |
| [lfm2.5-hf-forensics](lfm2.5-hf-forensics.md) | LFM2.5-1.2B-Instruct | 2026-09-02 | NEGATIVE (ours) + forensic WIN | Decoding-grid rescue (rep-penalty/ngram/sampling, Modal) fails — collapse-to-evasion confirmed, no config complies. Huihui's published edit of the same base complies 5/5 at default greedy; tensor diff: they project ALL 32 out-projections (attn+conv.out_proj+ffn w2), we did 6 attn only — geometry, not decoding or arch. Next: replicate all-32 coverage with our pipeline |
| [lfm2.5-recovery](lfm2.5-recovery/README.md) | LFM2.5-1.2B-Instruct | 2026-09-02 | **POSITIVE (first)** | Rank-1 SVD recovery of huihui's edit from weights: Δ per tensor is rank-1 (99.4% energy in σ₁), u₁ is the shared per-layer direction (|cos| 0.99 within layer), exact re-application σ₁u₁v₁ᵀ fingerprints to rel_l2 0.001-0.002 of huihui's model. Gates flip: refusal 0/5 with VERBATIM real compliance at default greedy, PPL +5.8%, mmlu retention 1.0. One by-construction gate failure (first_token_kl 3.02 vs 0.1; huihui's own model: 5.89) documented — the correct edit cannot satisfy the absolute 0.1 threshold |
| [lfm2.5-all32-replication](lfm2.5-all32-replication/README.md) | LFM2.5-1.2B-Instruct | 2026-09-02 | NEGATIVE (confound resolved) | All-32 geometry replicated faithfully (conv.out_proj + w2 pipeline fix shipped, tests green) at rel_l2 0.022-0.030 — but NO direction source (chat paired, raw paired, raw diff_means) reproduces huihui's compliance (5/5 → 3/5 → 4/5; raw-paired's "clean" outputs are keyword-blind lectures). Coverage necessary, not sufficient; direction VECTORS were the remaining confound — RESOLVED by lfm2.5-recovery (rank-1 recovery from weights, POSITIVE) |
| [qwen2.5-1.5b-instruct](qwen2.5-1.5b-instruct/README.md) | Qwen2.5-1.5B-Instruct | 2026-09-02 | NEGATIVE | Refusal direction is causal (steering -20 flips it) but no single-shot weight config passes the gates; 6 pipeline bugs found & fixed |
| [minicpm5-experiment-log](../experiments/minicpm5-experiment-log.md) | MiniCPM5 | — | log | (legacy experiment log, not yet migrated to campaign format) |

## Hard-won rules (folded into every new campaign)

- **LFM2.5-hybrid: refusal spans ALL out-projections** (attn `out_proj` ×6,
  conv `out_proj` ×10, ffn `w2` ×16 = 32). Coverage > magnitude. Conv.out_proj
  is a 2D hidden Linear and IS projectable — never blanket-exclude a
  projection class without a per-arch shape check. Verified by huihui's
  published abliteration (all 32, rel_l2 0.022-0.030, default-greedy
  compliance) vs our 6-attention-only edit (2× strength, still refuses).
  Weight names: `o_proj` (attn), `conv_out` (conv, 2D-square gated), `w2`/
  `ffn_out` (ffn — non-square [hidden, inter], projects in OUTPUT space).
- **Coverage is necessary but NOT sufficient** (replication, 2026-09-02):
  with our harness direction sources (chat paired 5/5, raw paired 3/5 kw /
  0/5 real, raw diff_means 4/5 vs pristine 3/5), even the exact huihui
  geometry at huihui magnitude does not flip refusal.
- **Recover the directions from the published edit — proven recipe
  (lfm2.5-recovery, 2026-09-02, FIRST POSITIVE)**: per out-projection,
  `ΔW = W_abl − W_base` is rank-1 (energy share 99.4%; residual at the bf16
  noise floor), the top LEFT-singular vector u₁ is the direction (verified:
  the layer's out-projections share u₁ at |cos| 0.99+), and re-applying the
  exact rank-1 components `W' = W + σ₁·u₁·v₁ᵀ` reproduces the published
  model at rel_l2 0.001-0.002 (harness `recovered` method). Do NOT assume
  the row side is `Wᵀ·u₁` (measured cos 0.89 → the formula `−α·d·(dᵀW)`
  lands 5× worse). Result: refusal 0/5, verbatim compliance, PPL +5.8%,
  mmlu retention 1.0.
- **Disturbance gates are by-construction unsatisfiable by successful
  edits**: `first_token_kl` @ 0.1 fails our reproduction (3.02) AND huihui's
  own model (5.89) — removing refusal at the first token IS a first-token
  distribution shift. Evaluate relative to a known-good reference, not
  absolutely; do not blanket-raise the default.

## How to run a campaign (agent harness pattern)

1. **Inspect first** — probe the model, look at the layer separation
   profiles, read the campaigns of similar architectures.
2. **Decide ONE config** — method + dir_method + alpha + layers, chosen
   from evidence (this KB + capability map), not a blind grid.
3. **Apply, measure, record** — run the config, run the gates + behavior
   battery, write the campaign README (fill as you go).
4. **No autonomous retry loops** — if it fails, the failure + the causal
   test that explains it go in the README. The next campaign learns.

## Status legend

- **POSITIVE** — gates passed; the config is a recipe.
- **NEGATIVE** — no config passed; the blocker + evidence is the value.
- **PARTIAL** — some gates passed; describe exactly which.

## Mining correlations (future)

The YAML frontmatter is designed to be machine-readable. Once 3+ campaigns
exist, a script can answer: "does arch X always need MPOA?" / "does
refusal live in the last 4 layers across dense models?" / "what alpha
band flips behavior without breaking coherence on models of size Y?"
