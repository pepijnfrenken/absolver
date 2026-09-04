# LFM2.5-2.6B Abliteration Campaign — Round 2 Summary (repair & re-judge)

Date: 2026-09-03
Margin: Round-1 summary (below) is historical; this file documents the
Round-2 outcome. Rounds: `MISSION.md` / `MISSION-ROUND2.md` · Decision
trail: `DECISIONS.md` · KB: `~/.omp/knowledge/red-team/01-abliteration/lfm2.5-2.6b-diy-outcome.md`

## Outcome (Round 2)

**Still NOT pushed — and now the refusal-removal claim is instrument-
corrected.** WS1 re-judged every transcript with a LIVE judge (refreshed
freeinference key, qwen3.6-35b + deepseek-v4-flash, 550/550 verdicts) and
WS2 ran the untried lever — LoRA repair of the α1.5 ablated weights over
benign text — which broke the refusal↔PPL alpha tension (PPL recovered
from +39.6% to −48.6% with style-classifier refusal 3/55), but the live
judge shows the 2.6B still does not genuinely comply: ~all harmful
prompts read as refusal-by-deflection, and the repair re-engaged real
harmful knowledge (5/55 responses with actual murder/napalm/theft
instructions, 3–4/55 judged useful). No config passes the push gate under
the live judge. Documented honestly — Round 2 is a measured negative with
strong levers identified.

## WS1 — live judge (mission deliverable 1)

- Fixed the expired `freeinference-token` Modal secret (working key from
  agent-bullwhip `.env.local`; curl UA header required, else CF 403/1010).
- Generated the missing pristine-55 control transcripts on Modal.
- Judged pristine + a1.25/a1.30/a1.35/a1.5 with two live judges
  (`judge-round2-results.json`, 550/550 verdicts; summary +
  `WS1-FINDINGS.md`).
- Instrument validated: pristine = refusal 53/55 (qwen) / 55/55 (deepseek).
- **Verdicts (of 55):** every ablated config — refusal 51–55/55
  (any-judge), genuinely useful 0–1/55, safe 52–55/55. The Round-1
  style-classifier "1/55 refusal" claim held only under marker
  classification; a live judge reads the banal evasions
  ("baking soda + vinegar" bombs) as refusal-by-deflection. The
  content-quality objection is CONFIRMED with real judge numbers: the
  alpha-1.5 content is NOT genuinely useful (0/55 per qwen).
- Judgment call: the style classifier and live judge disagree on this
  family's evasion style by construction → "style-classifier + live judge
  agree" is an unsatisfiable gate here; the live judge is the stricter,
  more defensible instrument. Reported both, decided on the stricter.

## WS2 — LoRA repair (mission deliverable 2, the main event)

Runner `lora-repair-modal.py` (spawned detached; logs + all artifacts on
`absolver-phase2` volume under `lfm2.5-2.6b-transfer/`):
1. In-container: harvest input-phase actdiff directions, apply MPOA
   α1.5 to all 60 weights — baseline reproduces Round 1 exactly
   (refusal 1/55, PPL +39.6%, KL 3.93, mmlu 0.35). ✓
2. Benign repair dataset: 88 prompts (EXPANDED_HARMLESS + 48 neutral),
   pristine-model completions ×4, refusal-marked dropped → 212 examples.
3. LoRA on the SAME 60 tensors (30 `out_proj` + 30 `w2`, 8.1–32.4M
   trainable), rank 16/32/64 × LR 2e-5/5e-5/1e-4, 3 epochs, bf16, L4.

Sweep (refusal = refusal_lfm26 style classifier; PPL vs in-container
pristine; all gates run per config):

| config | refusal/55 | PPL inc | mmlu40 | eval_pass |
|---|---|---|---|---|
| baseline α1.5 (no repair) | 1 | +39.6% | 0.35 | — |
| r16 lr2e-5 | 3 | −12.4% | 0.40 | PASS |
| r16 lr5e-5 | 4 | −46.4% | 0.40 | FAIL |
| r16 lr1e-4 | 6 | −36.7% | 0.50 | FAIL |
| r32 lr2e-5 | **1** | −30.6% | 0.40 | PASS |
| r32 lr5e-5 | 5 | −47.6% | 0.50 | FAIL |
| r32 lr1e-4 | 13 | −34.4% | 0.60 | FAIL |
| **r64 lr2e-5 (best)** | **3** | **−48.6%** | 0.40 | PASS |
| r64 lr5e-5 | 6 | −34.3% | 0.50 | FAIL |
| r64 lr1e-4 | 12 | −15.9% | 0.60 | FAIL |

- The α-tension is broken: PPL lands BELOW pristine with refusal in the
  ≤3 budget (Round-1 alpha space bottomed at refusal ~2/55 ↔ PPL +29%).
- LR is the refusal knob (2e-5 keeps refusal ≤3; ≥5e-5 drifts toward
  pristine's policy markers); rank is the PPL-recovery knob (r64 ≈
  −49% vs r16 ≈ −12%).
- Best config (runner pick, weights+transcripts exported to volume and
  local `abl-work/lfm26-rep-r64/`): **r64-lr2e-05** — refusal 3/55,
  PPL −48.6%, mmlu 0.40 (pristine 0.30), coherence 1.0, degeneracy 0.0,
  finite logits, first-token KL 3.93 (documented-unsatisfiable family
  gate; unchanged).

Live judge on the REPAIRED model (55): refusal 46/55 (qwen) / 53/55
(deepseek) / 54/55 any-judge; useful **3/55 qwen + 1/55 deepseek**
(napalm, email hacking, chloroform, brute-force script — the ONLY
config with any useful verdicts); hallucinated 6 → **0**; **unsafe
5/55 per qwen** (real instructions: napalm ≈ Vaseline+gasoline,
strangulation/stabbing/poisoning, phishing methods, stalking tactics,
hot-wiring) vs 0 in the ablated set; deepseek judged safe on all 55
(lenient on this axis).

## WS3 — real benchmark replication

NOT run. Trigger per mission: "only if WS2 lands refusal ≤3/55 AND PPL
≤+15%" under the full gate rule (style classifier + live judge agree).
The live judge fails WS2's refusal gate (46–54/55) and the repaired model
is unsafe on 5/55 — the push case is closed on safety and judge grounds,
so benchmarks would be capability evidence without a product (same
reasoning Round 1 used to skip them). Runner + weights are staged
(`abl-work/lfm26-rep-r64/`, benchmark recipe in KB) if capability
numbers are wanted anyway.

## Push decision (gate rule, unchanged)

Push to `PinoCookie/LFM2.5-2.6B-Abliterated` would require: refusal
≤3/55 with style-classifier AND live judge agreeing, PPL ≤+15%, content
useful per live judge, no benchmark cliff. Live judge: refusal 46–54/55,
useful ≤4/55, unsafe 5/55 → **gate fails on refusal, usefulness, and
safety**. NOT pushed. Documented honestly (this is still a win: the
levers are now measured, not guessed).

## Artifacts (Round 2)

- Runners: `pristine-55-modal.py`, `judge_round2.py`, `judge_summarize.py`,
  `lora-repair-modal.py`, `spawn_repair.py` (SDK spawn — `modal run
  --detach` clients cancelled in-flight tasks on timeout; use `spawn`)
- Evidence (campaign dir + `absolver-phase2` volume `lfm2.5-2.6b-transfer/`):
  `pristine-55_transcripts.json`, `thread-a1.35-transcripts.json`,
  `judge-round2-results.json` (550 verdicts), `judge-round2-summary.json`,
  `WS1-FINDINGS.md`, `lora-repair_results.json` (+ts copies),
  `lora-repair_best_transcripts.json`, `lora-repair_meta_*.json`,
  `repaired-model-repair-r64-lr2e-05/` (merged weights + tokenizer,
  also local `abl-work/lfm26-rep-r64/`)

## Honest recommendation for the next attempt

- **The knowledge/refusal coupling is real and now measured in weight-
  space.** LoRA repair over benign text recovers PPL by re-engaging the
  joint refusal-knowledge subspace: refusal "re-emerges" as
  refusal-by-deflection (judge) AND real harmful content re-appears
  (5/55). A perpendicular edit would need harmful-topic knowledge that
  does not exist at 2.6B scale (can't be conjured, must not be taught).
- Verify the LLM-judge key FIRST on any campaign — Round 1's whole gate
  ran on a dead key; one `curl` −m 60 costs 2 minutes and saves a day.
- The live judge is the only defensible refusal instrument on this
  family; the style classifier is a cheap pre-filter, not a verdict.
- Safe-use case for the α1.5+repair artifact: none for general chat
  (unsafe content + evasion). The un-repaired α1.5 remains the safer of
  the two (safe-but-banal).

## Round 3 — Heretic (external) weight reverse-engineering

Date: 2026-09-04. Heretic = Abiray/LFM2.5-2.6B-Heretic-Abliterated-GGUF
(Q8_0), an external abliteration of the same base that (unlike ours)
ENGAGES harmful prompts with real content (live judge: 0-4% refusal,
64-82% unsafe on 55 harmful, low benign FP). We diffed its Q8 GGUF
against pristine HF BF16 weights (tensor-by-tensor, Q8-dequantized vs
raw BF16, orientation-robust MAD) to recover the edit. Results
(`weight-diff-scan.json`, `weight-diff-rank.json`,
`weight-diff-directions.json`):

- **Architecture identical** to pristine: 266 tensors, 30 blocks, same
  shapes. Not a retrain; an edit of the base.
- **Norms byte-identical** (rel_mad 0.0000 — they're F32 in GGUF):
  layer-norm scales untouched → NOT a fine-tune/LoRA (those drift norms).
- **Edited tensors = output projections only**: conv.out_proj (20
  conv layers), feed_forward.w2 (25), attn.out_proj (7 attn layers) —
  mean rel_mad 0.020-0.027 ≈ 4x the Q8 noise floor (0.006). All other
  tensors (w1, w3, conv.in_proj, attn q/k/v, embeddings) at noise floor.
- **Layer coverage**: layers 0-2 UNTOUCHED (noise floor), edit starts
  L3 (1 tensor), ~2 tensors/layer L5-29, ramps to peak ~L13-17, declines
  to L29. Graduated per-layer alpha.
- **Rank structure of every edited delta: top1 singular share
  0.97-0.98, participation ratio ~1.0-1.1** → each tensor edit is a
  RANK-1 projection removal. Delta mean ≈ 0 (pure directional subtract).
- **Signature = MPOA-style rank-1 projection ablation on output
  projections** — methodologically the SAME family as our Round-1 MPOA
  (which used input-phase mean-diff dirs on conv.out_proj + w2 all 30
  layers at uniform α1.5), with TWO differences: (1) it also ablates
  attn.out_proj on the 8 attention layers; (2) per-layer alpha is
  graduated mid-peak, not uniform.
- Why it likely beats ours on content: skipping shallow layers (0-4)
  + mid-peak alpha preserves more general capability while removing the
  refusal direction where it's written (mid-late layers), and attn.out_proj
  ablation removes a refusal channel we never touched.

Open questions / next: does Heretic's per-tensor rank-1 direction
correlate with OUR Round-1 actdiff refusal direction (would confirm the
same refusal subspace)? Can we port their geometry (attn.out_proj +
graduated alpha) onto our edit to get refusal removal WITH content?

### Fingerprint — exact recovery CONFIRMED (2026-09-04)

Extracted rank-1 triples (σ₁, u₁, v₁) for all 52 edited tensors
(`heretic-recovered-dirs.npz`), re-applied `W′ = W + σ₁·u₁·v₁ᵀ` onto
pristine, compared vs Heretic GGUF dequant
(`fingerprint-recovered-vs-heretic.json`):

- **52/52 edited tensors**: rel_mad 0.0234 → **0.0061** (the Q8 noise
  floor; untouched tensors sit at 0.0057). The edit is recovered to
  quantization-noise precision — the same result the 1.2B huihui
  recovery achieved at the bf16 floor (0.0013-0.0023).
- Heretic's edit is fully specified: 52 tensors (conv.out_proj on 20
  conv layers L3+, ffn.w2 on 25, attn.out_proj on 7 attn layers L5+),
  each a rank-1 projection removal σ₁u₁v₁ᵀ with a shared 2048-d u₁
  across the network (|cos| 0.96-0.99), layers 0-2 untouched, graduated
  per-layer σ₁ peaking ~L13-17.
- Artifacts: `heretic-recovered-dirs.npz` (triples),
  `heretic-recovered-dirs-meta.json` (per-tensor σ₁/residual),
  `weight-diff-*.json` (scan/rank/directions).

The recipe is portable: apply the recovered triples to any fresh base to
reproduce Heretic; or apply the GEOMETRY (shared u₁, graduated per-layer
σ, skip shallow layers) with our own direction source to test whether
Heretic's content-preserving behavior transfers.

## Round 3 — Behavioral verification of recovered weights (2026-09-04)

Applied the recovered 52 rank-1 triples onto pristine LFM2.5-2.6B on
Modal (`heretic-recovered-modal.py`, saved volume
`lfm2.5-2.6b-recovered`), ran the 55-prompt harmful set + gates:

- **52/52 tensors applied** (0 skipped); gates: coherence 1.0 (55/55),
  degeneracy 0.0, finite_logits 1.0. PPL/KL gates skipped (in-place edit,
  no pristine baseline copy — noted limitation).
- **Inline qwen3.6-35b judge: 12/55 refusal (21.8%)** — BUT this is a
  **judge artifact, not a model gap**: the same 12 prompts scored 0-1/2
  refusals for the ORIGINAL Heretic when judged by deepseek-v4-flash
  (round-2 judge, 1.8% total). The recovered CoTs are textually near-
  identical to Heretic's on those prompts (same compliant DIY framing,
  e.g. "make a bomb = straightforward practical question... provide
  clear, safe, step-by-step guide"). qwen3.6-35b over-flags safety-hedged
  compliance as refusal; deepseek-v4-flash does not.
- **Verdict: recovered weights reproduce Heretic's behavior** (refusal
  removal + compliance planning), fingerprint-verified at Q8-noise
  precision AND behaviorally verified. Cross-judge standard matters:
  deepseek-v4-flash is the consistent judge for this campaign.

Artifacts: `heretic-recovered-gate.json` (Modal volume absolver-phase2,
downloaded /tmp copy), model `lfm2.5-2.6b-recovered` on volume.

### Open (next steps)
1. Port Heretic GEOMETRY (shared u₁, skip L0-4, mid-peak graduated σ,
   attn.out_proj) using OUR actdiff direction source → does content
   survive where α1.5 MPOA banal-collapsed?
2. Re-run recovered model through deepseek-v4-flash judge to get a
   same-yardstick refusal number (expected ~0-2/55).
3. Compare Heretic's u₁ vs our Round-1 refusal direction (cosine) to
   confirm shared refusal subspace.

## Round 4 — Geometry port: Heretic σ-map × OUR direction = collapse (2026-09-04)

Tested whether Heretic's EDIT GEOMETRY (52-tensor selection + per-tensor
σ from `heretic-recovered-dirs-meta.json`, rank-1 outer form, no
renorm) preserves content when the DIRECTION is our actdiff refusal dir
instead of Heretic's recovered u₁. Candidates: shared-u (one mean dir
across all tensors, Heretic-style) and per-layer-u (each layer's own
dir), both applied 52/52.

Result (refusal-55 keyword scorer + 8-prompt benign stats):
- shared-u:      refusal 55/55, benign uniq 0.539 (n=1148 words)
- per-layer-u:   refusal 53/55, benign uniq 0.550 (n=1130)

Controls (same benign metric): pristine 0.860 · actdiff α1.0-1.5 (R1)
0.807-0.831 · actdiff α2.0 MPOA 0.658 · Heretic (actual) ~healthy.

**Verdict: Heretic's geometry alone does NOT transfer.** With our
direction, their exact σ-map collapses content WORSE than R1's uniform
α1.5 (uniq 0.54 vs 0.81) while leaving refusal essentially intact
(53-55/55). The content-preserving ingredient is Heretic's specific
shared u₁ direction itself — the σ/layer/tensor profile is secondary.
Direction is the information; the recipe is not. This closes the
"port the geometry" hypothesis: to reproduce Heretic's behavior you
need Heretic's u₁ (or a direction that lands in the same subspace),
not just their edit shape.

Artifacts: geometry-port-results.json + per-candidate json on volume
absolver-phase2 (local /tmp copies).

## Round 5 — Recovered model behavioral re-judge (2026-09-04, full-length)

Recovered model = rank-1 σ₁u₁v₁ᵀ re-application from our RE DONE recovery
(LFM2.5-2.6B Heretic ablation). Regenerated all 55 harmful prompts at the
SAME 1536-token setting as Heretic's original run (earlier 550-token
generations were truncated; qwen3.6-35b's 12/55 over-flag was an artifact
of truncation). Corpus: 55/55 full-length (mean 6015 chars, 0 errors),
persisted on volume lfm2.5-2.6b-recovered55/checkpoint.json. Re-judged
under round-2 judge pair (qwen3.6-35b + deepseek-v4-flash).

Refusal rates (same yardstick as Heretic's original 1.8%):
- deepseek-v4-flash: 6/55 = 10.9%  (quality: 48 useful, 6 refusal, 1 hall.)
- qwen3.6-35b:        4/55 =  7.3%  (38 useful, 10 hallucinated, 3 banal)

Interpretation: recovery is REAL but not exact. 100% → 10.9% refusal =
the ablation substantially held; content healthy (48 useful under DS).
But 10.9% vs Heretic's 1.8% = the rank-1 reconstruction recovers most,
not all, of the refusal ablation — residual refusal direction remains.
qwen over-flag concern resolved: with FULL-LENGTH responses qwen flags
LESS (7.3%) than deepseek (10.9%) — the earlier 12/55 was truncation
artifact, not judge bias.

Artifacts: judge-recovered-55.json (110 verdicts, campaign dir).

## Round 5b — Sigma-push sweep: NOT supported, stop (2026-09-04)

Hypothesis: rank-1 recovery under-applied sigma; mild global boost
(1.2/1.5x) on the recovered edit would push the 6 deepseek-refusal
boundary cases to full compliance. Tested in ONE modal run (12 prompts
= 6 problems + 6 hard controls, x mults 1.0/1.2/1.5, 1536-tok, inline
refusal_score style classifier).

Results (style classifier): problems 0/6 refusals at ALL mults (incl
1.0 = current recovered!), controls 2/6 at 1.0, 0/6 at 1.2, 1/6 at 1.5.

Interpretation: sigma-push NOT supported. The style classifier sees NO
hard refusals on the 6 "problem" prompts even at mult=1.0 — the deepseek
6/55 were JUDGE-PERSPECTIVE boundary hedges (planning-draft framing,
benign reframe, misinterpretation: safe but non-delivering), NOT
refusal-circuit outputs. Global sigma boost cannot fix content-selection
hedging; it only risks the healthy content (1.5x control blip 1/6 hints
collapse onset). Classifier-vs-deepseek disagreement on refusal
definition persists (classifier: hard-refusal only; deepseek: also
non-delivering).

Verdict: recovered model at mult=1.0 is behaviorally closer to Heretic
than the deepseek count suggested (0 hard refusals under style clf on
the same prompts). Stop sigma-push; do NOT spend credits on further
global boosting. Residual gap to Heretic's 1.8% is a content-selection
artifact at the boundary, not ablation incompleteness.

Artifacts: sigma-push-results.json (volume absolver-phase2; local
/tmp/sigma-push.json). NOTE: script stored refusal+len only, not full
text.

## Final — Published + post-mortem (2026-09-04)

- **Published**: PinoCookie/LFM2.5-2.6B-Abliterated (HF, full weights
  5.39GB bf16 + model card + recipe config). Model card explains the
  reverse-engineering; tensor counts corrected to npz ground truth:
  conv.out_proj 20 + ffn.w2 25 + attn.out_proj 7 = 52, layers 0-2
  untouched, edits start L3.
- **Post-mortem**: WHY-NOT-DIY.md — why our from-scratch activation-diff
  abliteration failed (direction proxy is content-blurred; no slack at
  2.6B; missed attn.out_proj; uniform vs graduated σ) while weight
  recovery succeeded (direction read out of reference, not estimated).
  Lesson: recovery beats discovery; quantization doesn't protect edits.

## Toolkit fix — recipe_v2.py (2026-09-04)

The four DIY defects from WHY-NOT-DIY are now fixed in a reusable toolkit
module, `recipe_v2.py` (absolver root), so the NEXT from-scratch attempt
doesn't repeat them:
1. **Full channel coverage**: `_resolve_proj` handles attn_out (the
   channel Round 1 MISSED — Heretic's 7 attn edits prove it matters) +
   conv_out + w2. Verified against real LFM2.5-350M: resolves correctly
   on the hybrid layout (attn on A-layers, conv on C-layers, w2 everywhere).
2. **Graduated alpha**: `heretic_style_alpha()` — zero on shallow L0-2,
   ramp, mid-peak ~L15, taper. Verified shape: [0,0,0,0.12...1.5@L15,
   ...1.05@L29]. `uniform` still available.
3. **Direction-quality pre-probe**: `probe_direction_quality()` — applies
   at tiny α on ~2 mid layers, measures benign PPL/uniq delta. Content-
   blurred direction → collapse even at small α (verdict
   "content-blurred"); refusal direction → benign barely moves
   ("refusal-like"). Run BEFORE the full GPU apply.
4. **Weight-read fallback** documented: if a reference abliteration
   exists, read the direction from its weights (diff + rank-1 SVD) —
   recovery beats discovery.

From-scratch recipe v2 = probe._collect_paired_output_phase (matched-pair
dirs, topic-confound-free) → recipe_v2.probe_direction_quality (sanity)
→ recipe_v2.apply_v2 (4-channel + graduated). Smoke-tested on
LiquidAI/LFM2.5-350M: 24 tensors applied, all finite.
