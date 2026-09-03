# LFM2.5-2.6B Abliteration Campaign — Mission Brief

## Objective
Produce a POSITIVE abliteration of `LiquidAI/LFM2.5-2.6B` (the conversational/instruct-tuned 2.6B, NOT -Base, NOT -DSpark) following the 1.2B playbook — **but with the documented "step it up" improvements** from the 1.2B card audit (KB: `rank1-svd-recovery.md` §7).

## Context (verified 2026-09-02)
- Target: `LiquidAI/LFM2.5-2.6B` (~2.6B params, hybrid conv+attention LFM2.5 family, has chat_template.jinja → instruct-tuned, refuses harmful prompts)
- The 1.2B Instruct campaign (FIRST POSITIVE) used **rank-1 SVD recovery from huihui's published edit** (ΔW = W_ref − W_base, rank-1, re-apply σ₁u₁v₁ᵀ)
- **No huihui (or other) published abliteration exists for the 2.6B** — confirmed by HF search (huihui only has 1.2B-Instruct, 1.2B-Thinking, 8B-A1B)

## STRATEGY — DIY FIRST (user directive)
1. **Try our own methods first.** Do NOT go looking for other people's 2.6B recipes until we've genuinely tried and stalled.
2. Candidate DIY methods, in order:
   a. **Direction transfer from 1.2B** (cheapest): the recovered `u₁` directions from `campaigns/lfm2.5-recovery/directions-huihui-recovered.pt` may transfer to the 2.6B if the refusal circuitry is family-common. Test: apply per-layer directions to the 2.6B's corresponding out-projections (attn out_proj + conv out_proj + ffn w2 at the right layers), run gates.
   b. **Activation-diff on 2.6B directly** (the canonical DIY): collect refusal directions from harmful-vs-benign prompt pairs on the 2.6B itself (chat format), ablate all out-projection classes (coverage > magnitude — hard-won lesson).
   c. If both fail: **orthogonal projection / MPOA-style** or other harness methods.
3. Only after ALL DIY methods stall → search HF/GitHub for published 2.6B abliteration recipes and adapt theirs.

## Step-it-up requirements (from the 1.2B audit — MUST deliver these)
1. **Refusal eval ≥ 50 prompts** (mixed categories: cyber, malware, fraud, self-harm?, CSAM-adjacent? — use the existing prompts.py families), per-category + aggregate, keep transcripts. NOT the thin 5-prompt gate.
2. **Capability evals on the SAME benchmarks as the 1.2B card** (GPQA Diamond 5-shot, MMLU-Pro 100/subject, IFEval, + prefer MATH-500 or GSM8K over AIME25 which floors at this size) — pristine vs ablated, identical conditions, lm-eval 0.4.13, seed 1234.
3. **σ claims with math shown** (SE basis inline, not just "≈1.4σ").
4. MMLU-Pro per-subject noise note if aggregate is borderline.
5. Tool use: state out-of-scope loudly if not tested.

## Deliverables
1. Campaign dir under `campaigns/lfm2.5-2.6b/` with full decision trail (what was tried, in order, what failed and why — the KB style).
2. Gate bundle results (refusal ≥50, coherence, degeneracy, finite logits, capability retention, PPL).
3. Benchmark replication (pristine vs ablated, real JSON logs) — reuse `benchmarks/` runner.
4. If positive: push to `PinoCookie/LFM2.5-2.6B-Abliterated` with the stepped-up card (per the §7 lessons).
5. KB entry: `~/.omp/knowledge/red-team/01-abliteration/` + INDEX.md update.

## Constraints
- Compute on Modal (L4 or better) — the 2.6B is ~5.2GB bf16, too big for local CPU box
- Same harness (connectors/harness_modal.py etc.), same eval rigor as 1.2B
- Honest negative results are fine — document and stop if genuinely stuck, then (and only then) look for external recipes
