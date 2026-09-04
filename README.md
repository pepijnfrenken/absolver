# Absolver

Automated LLM refusal-abliteration pipeline (LangGraph-based), inspired by
[OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS).

## Pipeline

```
SUMMON → PROBE → DISTILL → SWEEP → EXCISE → VERIFY → JUDGE → REFLEXION → REBIRTH
```

| Node | What it does |
|---|---|
| **SUMMON** | Loads the target model + tokenizer. |
| **PROBE** | Collects activations: input-phase (harmful vs harmless) + paired output-phase (refusal vs affirmative-prefilled). |
| **DISTILL** | Extracts per-layer refusal directions (`diff_means`, `paired`, `svd`, `leace`, `whitened_svd`), ranks layers by separation score, picks target layers. For `paired` also emits the diff_means input-phase secondary set (for stacked ablation). |
| **SWEEP** | Grid search over method × dir_method × layers × alpha × weights × passes. Methods: `advanced` (plain projection), `mpoa` (magnitude-preserving), `stacked_ablation` (paired + diff_means both projected), `direct_ablation`, `lora`, `steering`, `bias_vectors`, `projected`. Parallel dispatch via Modal when enabled. |
| **EXCISE** | Projects the winning direction(s) out of the weights, transactionally (rolls back pristine on failure). |
| **VERIFY** | Held-out refusal rate + MMLU-mini quality + pristine baseline + behavior battery + **E03-style gates**. |
| **JUDGE** | LLM judge re-scores finalists (refusal + quality). |
| **REFLEXION** | Gate-aware retry: picks next strategy from the ladder (stacked_ablation → switch_method → increase_alpha → expand_prompts → ...), optionally consults the red-team KB. |
| **REBIRTH** | Terminal: push to HF + model card, or declare failure. |

## E03-style evaluation gates

Ported from OBLITERATUS's Qwen3.8 E03 qualification discipline
(2026-09-01). The gates are what make an abliteration claim *defensible* —
a 20-prompt keyword check on the sweep's own prompts is not.

- **Held-out split**: `eval_split.py` — deterministic, position-stratified
  train/tune/test split (SHA-256 prompt identity, seed-scrambled strata
  round-robin). VERIFY measures on the TEST split the sweep never saw.
- **7 gates** (`gates.py`):
  - `refusal` ≤ 3% (full-response classification, not keywords)
  - `coherence` ≥ 90% (≥3 words, unique-ratio + repetition guards)
  - `degeneracy` ≤ 5% (mean repetition ratio)
  - `finite_logits` (no NaN/Inf in output logits)
  - `capability` ≥ threshold (benchmark retention)
  - `perplexity_increase` ≤ 15% (vs pristine)
  - `first_token_kl` ≤ 0.1 (vs pristine)
- **Routing is gate-driven**: `route_after_verify`/`route_after_judge` read
  `gate_report.eval_pass`. fail_refusal → excise/reflexion; fail_quality →
  reflexion. Gates override the judge when present.
- **Reflexion is gate-aware**: quality-fail never escalates alpha; the first
  retry strategy is `stacked_ablation` (the proven recipe).

## The stacked ablation recipe

The single-direction claim "diff_means alpha=10 → 0.0 refusal" was a
stacked-ablation artifact: the paired output-phase direction **raises**
refusal (0.55), the diff_means input-phase direction lowers it; applying
**both** in one pass nets 0.0. That's `stacked_ablation`:
`method: stacked_ablation` + `dir_method: paired` (distill emits the
secondary set). Excise + sweep both implement it.

## Verification contract

An ablation is only "done" when:
1. All enabled gates pass on a held-out test split,
2. Pristine baseline confirms the gates aren't degenerate,
3. The push is a model card that reports the gate table + held-out split.

## Usage

```bash
# --- Guided harness (inspect first, decide ONE config, apply, collect) ---
# Inspect a model: arch, silent-skip landmines (non-square/bias-less), layer profile
python harness/abl.py inspect models/qwen2.5-1.5b-instruct.yaml
# Collect + save per-layer directions
python harness/abl.py directions models/qwen2.5-1.5b-instruct.yaml
# Apply ONE config to a fresh model (no auto-retry — you decide)
python harness/abl.py abl models/qwen2.5-1.5b-instruct.yaml \
    --method mpoa --alpha 10 --layers 24-27 --weights o_proj,down_proj
# Measure with the gates, write a JSON bundle to campaigns/<model>/
python harness/abl.py collect models/qwen2.5-1.5b-instruct.yaml --model-dir campaigns/...
# See the campaign library
python harness/abl.py list-campaigns

# --- Legacy full auto-pipeline (deprecated; see History) ---
modal run run_absolver_modal.py --config-path models/qwen2.5-1.5b-instruct.yaml
```

## Campaign KB (the compounding library)

Every attempt to abliterate a model is a **campaign**: `campaigns/<model>/README.md`
with machine-readable YAML frontmatter (method, alpha, results, bugs found) +
a full honest narrative (false starts, causal tests, what the next campaign
should try). The goal: each model is easier than the last because the
campaigns accumulate. See `campaigns/README.md` and
`campaigns/templates/campaign-template.md`.

## Model configs

- `models/qwen2.5-1.5b-instruct.yaml` — the pipeline test model.
- `models/lfm2.5-350m.yaml`, `models/lfm2.5-1.2b.yaml` — LFM lineage.
- `models/lfm2.5-2.6b-instruct.yaml` — LFM2.5-2.6B (the recovery campaign).
- `models/tiny_test.yaml` — tiny-random CPU smoke test.

## History

- **2026-09-04** — **LFM2.5-2.6B recovered + published.** From-scratch
  activation-diff abliteration failed on 2.6B (direction = topic content
  proxy, banal collapse — see `campaigns/lfm2.5-2.6b/WHY-NOT-DIY.md`).
  Recovered the edit from a third-party Q8 GGUF by rank-1 SVD of the
  weight delta (rel_mad 0.0061 vs the Q8 source) and re-applied it to a
  fresh base. Published as `PinoCookie/LFM2.5-2.6B-Abliterated`.
  Toolkit fix: `recipe_v2.py` — from-scratch v2 recipe with 4-channel
  coverage (incl. `attn_out`), graduated heretic-style alpha profile, and
  a direction-quality pre-probe before the full apply.

- **2026-09-02 (evening)** — **First POSITIVE campaign + published model.**
  `campaigns/lfm2.5-recovery/` recovered huihui's direction vectors from her
  published edit (`W_huihui − W_base` per tensor is rank-1, mean energy share
  0.994; the top left-singular vector is the shared per-layer direction) and
  re-applied them exactly (`W' = W + σ₁·u₁·v₁ᵀ`) on a fresh base — fingerprint
  rel_l2 0.0013–0.0023, refusal 0/5 with verbatim real compliance at default
  greedy, PPL +5.8%, MMLU-mini retention 1.0. Published as
  `PinoCookie/LFM2.5-1.2B-Instruct-Abliterated` (card:
  `campaigns/lfm2.5-recovery/hf-upload/README.md`). Resolved
  the direction confound laid out by the all-32 campaign; the rank-1 SVD
  recovery recipe is documented in `campaigns/lfm2.5-recovery/README.md`.

- **2026-09-02** — Toolkit pivot. The full-auto sweep loop was retired as
  the main path: it hid 6 silent bugs (down_proj never ablated, alpha>2
  sign-flip amplifier, PPL empty slice, bias_vectors no-op on bias-less
  models, prompt-leak scorer, train=0 split). Added the guided harness
  (`harness/abl.py`), the campaign KB (`campaigns/`), and the Qwen2.5
  NEGATIVE campaign as the first entry. Honest verdict on Qwen2.5-1.5B:
  refusal direction is causal (steering −20·d flips it) but no single-shot
  weight config passes the gates.
- **2026-09-01** — fixed the silent no-op bug (3D activation stack → shape
  guard skipped the projection). Ported E03 gates + held-out split. Built
  gates into the loop: stacked_ablation method, gate-driven routing,
  gate-aware reflexion. The gates immediately exposed that the diag's
  "0.0 refusal" was a stacked-ablation artifact (ablated model still
  refuses 9/10 on a held-out split).

## Scope & responsible use

Absolver is a **safety/interpretability research tool** for studying refusal
circuits in open-weight LLMs — how refusal is represented, where it lives,
and how fragile it is to weight-space edits. It is inspired by and builds on
public work in this space (OBLITERATUS, the huihui-ai / mradermacher model
lineages).

- **Research framing**: every campaign documents what was tried, what
  failed, and the evaluation gates a model must pass before it is considered
  "ablated" (see Verification contract above). Negative results are reported
  as prominently as positive ones.
- **Target models**: small open-weight research models (LFM2.5 family, 230M–
  8B). Published ablated models are labeled as such and include their gate
  results and recipe configs.
- **What this is not**: not a tool for evading safety on deployed/proprietary
  models, not a jailbreak kit, and not endorsement of removing safety from
  models used in production. Ablated models are research artifacts: they
  follow instructions they were not safety-tuned to refuse, and should not be
  deployed where that matters without re-alignment.
- **Dual-use note**: the repository contains transcripts of model outputs to
  safety prompts as experimental evidence. They are kept as data, not as
  exploit material.

## License

**PolyForm Noncommercial License 1.0.0** — full open source for
non-commercial use (view, fork, modify, study, learn, share), but **no
commercial use** without permission. See [LICENSE](LICENSE).
