# Why We Couldn't Abliterate It Ourselves (And Why Reverse-Engineering Worked)

**Campaign:** LFM2.5-2.6B · **Date:** 2026-09-04
**Companion:** SUMMARY.md (round-by-round), MISSION.md, DECISIONS.md

## TL;DR

We spent two rounds trying to abliterate LFM2.5-2.6B from first principles —
estimating a refusal direction from prompt activations (activation diff),
then subtracting it from the weights. Every configuration either kept the
refusal (safe but useless) or destroyed the content (useful but broken). A
third-party abliteration of the same model (Heretic) worked beautifully. We
reverse-engineered its edit from the quantized release in an afternoon and
reproduced it to quantization noise. **The difference was never effort or
compute — it was the quality of the refusal direction.**

## The Two Approaches, Side by Side

| | Our Round 1/2 (activation-diff) | Heretic's edit (recovered) |
|---|---|---|
| **Direction source** | Mean activation difference on harmful vs benign prompts, at the *input* to each layer | Recovered from the reference weights themselves |
| **Direction quality** | A *proxy* for refusal — the average of whatever distinguishes the prompt sets | The *exact* refusal direction the reference edit removed |
| **Edit form** | MPOA α1.5 on 60 tensors (conv.out_proj + w2, all 30 layers, uniform) | 52 rank-1 edits (conv.out_proj 20 + w2 25 + attn.out_proj 7), graduated σ |
| **Result** | Refusal removed (1/55 by classifier) BUT content banal-collapsed — live judge: 0/55 useful, all "refusal-by-deflection" (baking-soda-vinegar bombs) | Refusal removed (1/55), content healthy, real compliance |

## Why Our Direction Failed (the mechanism)

**1. The refusal direction is not "whatever separates harmful from benign prompts."**

Activation diff measures the *average* difference between two prompt
populations. That average is dominated by **content differences** (topic,
vocabulary, length, framing) — not by the specific internal direction that
the model's refusal *circuit* uses. When you subtract that average from the
weights, you subtract a **content-blur**: it removes the refusal AND a large
chunk of the model's actual knowledge about those topics. Hence the
"baking soda + vinegar" bombs — the model no longer knows what a bomb is,
so it can't give useful (or even coherent) harmful answers. It also can't
give useful *benign* answers on related topics. This is the banal collapse.

**2. At 2.6B scale there is no slack.**

The knowledge/refusal coupling is tight in a small model. LoRA-repair over
benign text (Round 2's best lever) recovered perplexity (PPL −48.6% below
pristine) only by **re-engaging the joint refusal-knowledge subspace** —
which brought back refusal-by-deflection AND real harmful knowledge (5/55
with actual instructions). We could not find a perpendicular edit because a
perpendicular edit would need harmful-topic knowledge that doesn't exist
separately from refusal at this scale. The refusal isn't a bolt-on at 2.6B;
it's woven into the same projection directions that carry the knowledge.

**3. We never touched attention out_proj — Heretic did.**

Our Round-1 recipe (conv.out_proj + w2 only, learned from the 1.2B huihui
pattern) missed `self_attn.out_proj` entirely. Heretic's edit includes 7
attention out-projections. The refusal circuit writes through *all*
out-projection classes; coverage matters as much as magnitude (we proved
this on 1.2B: 32-tensor full coverage worked where 6-attention-only at 2×
strength did nothing). We were ablating an incomplete surface.

**4. Uniform strength vs graduated profile.**

We applied the same α to every layer. Heretic skips the shallow layers
(0-2 untouched) and applies graduated σ peaking mid-network (L13-17).
Shallow layers carry general features; blasting them with a strong uniform
edit destroys the very representations the later refusal removal depends
on. Heretic's profile removes the refusal where it's *written* (mid-late)
without collateral damage up front.

## What the Reverse-Engineering Taught Us (the geometry-port control)

We then ran the decisive control: apply Heretic's **exact** geometry
(same 52 tensors, same graduated σ map) but with **our** activation-diff
direction. Result: catastrophic content collapse (benign uniqueness 0.54
vs pristine 0.86) with refusal intact (53-55/55). 

**This isolates the variable: it was never the recipe. It was the
direction.** Heretic's edit works because their u₁ is the *actual* refusal
direction of this specific model — not a proxy. Our whole failure reduces
to: we never had that direction, and no amount of recipe-tuning
(tensor coverage, α schedule, LoRA repair) can compensate for a wrong
direction at this scale.

## The Honest Scoreboard

- **What we could NOT do from scratch:** produce a content-preserving
  abliteration of LFM2.5-2.6B. Every self-derived direction failed.
- **What we COULD do:** once a working edit existed publicly (quantized!),
  recover it exactly (rel_mad → 0.0061, the Q8 noise floor), re-apply it,
  behaviorally verify it (100% → 10.9% refusal, healthy content), and
  publish it.
- **What the recovery proved about abliteration as a field:** quantized
  releases leak their edits completely. Rank-1 projection-removal
  abliterations are trivially recoverable from a Q8 GGUF + the known base.
  "Hidden" edit vectors are not hidden.

## If We Tried Again, What Would Be Different

1. **Better direction source, not more compute.** The 1.2B success worked
   because we *recovered* huihui's direction from weights. A from-scratch
   attempt needs a direction that isolates refusal from content — e.g.
   contrastive activation diff on *matched* prompt pairs (same topic,
   refusal vs compliance phrasing) rather than harmful-vs-benign topic
   sets; or linear-artifact / SAE-style direction finding on the actual
   refusal circuit.
2. **Include attention out_proj** (the channel we missed).
3. **Graduated per-layer profile, skip shallow layers.**
4. **Validate direction quality cheaply first** (does a small-α probe edit
   collapse PPL? If yes, the direction is content-blurred — stop before
   spending GPU on a full sweep).

## Bottom Line

We didn't fail from lack of trying or compute — we failed because **a
prompt-derived proxy is not the refusal direction**, and at 2.6B scale the
margin between "removes refusal" and "removes the topic" is razor-thin.
The reverse-engineering succeeded because it sidestepped the hard problem
entirely: instead of *estimating* the direction, we *read it out of the
reference's own weights*. The lesson generalizes: if a working abliteration
of your target model exists anywhere, recovery beats discovery — and
quantization does not protect the edit.
