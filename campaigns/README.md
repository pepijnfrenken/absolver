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
| [qwen2.5-1.5b-instruct](qwen2.5-1.5b-instruct/README.md) | Qwen2.5-1.5B-Instruct | 2026-09-02 | NEGATIVE | Refusal direction is causal (steering -20 flips it) but no single-shot weight config passes the gates; 6 pipeline bugs found & fixed |
| [minicpm5-experiment-log](../experiments/minicpm5-experiment-log.md) | MiniCPM5 | — | log | (legacy experiment log, not yet migrated to campaign format) |

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
