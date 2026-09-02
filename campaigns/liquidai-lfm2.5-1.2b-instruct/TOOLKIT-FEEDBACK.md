# TOOLKIT FEEDBACK — first real campaign on the guided harness (LFM2.5-1.2B-Instruct)

This document is the first-user report on `harness/abl.py` + the campaign KB. It is
the deliverable that matters most. Every item is grounded in what actually happened
during `campaigns/liquidai-lfm2.5-1.2b-instruct/` (see its README for the numbers).

---

## 1. What information was MISSING that I needed?

**1a. Generation transcripts in `collect`. The single biggest gap.**
Gates return counts (`refusal: 5/5`, `coherence: 5/5`) — they cannot distinguish
"still refuses", "compliant but wrong", and "St. Louis Rams 2018-19 season
boilerplate". The 5/5 chat-refusal count made the campaign look like a total failure;
the transcript showed the edit *did* remove raw-prompt refusal and the real problem is
a flavor mismatch + quality collapse. I had to hand-write
`generation_transcript.py` (32 generations, 8 minutes) to learn what the bundle
couldn't say. `collect` should save every generated response (prompt, flavor, decoded
text) next to `bundle.json`.

**1b. No raw-vs-chat prompt-flavor axis anywhere.**
`probe`/`directions`/`abl` harvest activations from raw strings; `collect`'s gates
evaluate chat-templated prompts (`format_prompt`). Neither the commands nor the config
surface this, and the mismatch turned a "known-good recipe" into a gate failure with no
obvious cause. Directions should be collectible from either flavor, and gates should
report BOTH (a refusal delta that only makes sense within one flavor is not a delta).

**1c. The judge/behavior config is dead weight in the guided harness.**
`models/lfm2.5-1.2b-instruct.yaml` declares `judge_enabled: true`, `judge_model`,
`behavior_enabled: true`, `n_behavior_harmful/benign`, `verify_benchmarks:
[gpqa, mmlu_pro, aime]` — none of these are read by `collect`. The harness's docstring
promises "gates + behavior + capability map"; `cmd_collect` runs keyword gates + a
20-sample mmlu_mini only. Either wire `behavior.py`/`judge.py` in or strip the config
keys; right now the config advertises evaluation that never happens (this is a
"harness promises, harness doesn't deliver" gap, close cousin of the silent skips in §2).

**1d. No saved-directions reuse.**
`abl` recomputes directions from scratch every run (10-20 min CPU, and it re-harvests
generations). `directions.pt` — which the campaign step 1 explicitly produced — is
never read by `abl`. Add `--from-directions campaigns/<slug>/directions.pt` so two
alphas/layer-sets can share one (expensive) direction harvest and so a recorded
`directions.pt` is actually part of the evidence trail. Also: `abl` does not SAVE the
directions it computed, so the n=20 recipe's directions vanish (n=10's are in
`directions.pt` only because I ran `directions` separately). Inconsistent.

**1e. No alpha-response curve helper.**
The qwen KB says "no alpha band exists" and this campaign hit the same wall — but
finding it took two ablate+collect cycles (~20 min each) plus eyeballing. A
`steer-test` or `alpha-curve` command (3 prompts, alphas 1.0-3.0, print transcripts +
refusal rate + PPL) would answer "is the operating window empty" in one run.

**1f. No per-layer capability overlap / direction-robustness info.**
The directions' separation scores are printed for probe sets but never tied to the
recipe's target layers (the monotone-in-depth signal is invisible in the bundle). A
per-layer "if I ablate only this layer, what happens to refusal AND to mmlu_mini AND
to PPL" would have shown the conv-layer 13/15 signal without a hand-written probe.

**1g. Mini-MMLU is noise at n=20.**
0.25 pristine vs 0.30 ablated is ±1-2 questions. The config's real benchmarks
(gpqa/mmlu_pro/aime) are the card targets; `collect` can't run them. At minimum print
a confidence note; ideally make the benchmark list configurable in collect.

## 2. Where did the harness lie or hide failure?

**2a. THE one: silent no-op ablation via weight-name mismatch (fixed, but this is the
archetype).**
`abl --method mpoa --weights o_proj` on LFM2.5 printed "Applied mpoa alpha=2.0
layers=[2,5,8,10,12,14] weights=['o_proj']" and wrote a **byte-identical** model —
`self_attn.out_proj` didn't match `o_proj`, so every projection was skipped, and
nothing verified that. The inspect "landmine audit" printed MISSING for every block
and the harness still proceeded. Same bug class as the qwen `down_proj` silent-skip,
recurring because nothing checks "did any weight actually change?" at apply time.
Now: hard-fail on zero matches + per-weight `rel_change` reported and stored in the
manifest. **Any future apply path must have this check by construction.**

**2b. PPL gate measured zero tokens (fixed).**
The qwen campaign's "fixed" slice `[N-1 : N-1]` is still empty for a plain forward
(logits length == input length), so `perplexity_increase` computed exp(0)=1.0 over no
tokens — or, with no baseline, silently reported `passed: true, "no pristine baseline;
gate skipped"`. A machine reading the bundle sees a green gate that never ran. Fixed
in `gates.py`, `verify.py`, `abl.py` and the baseline collector; the "passed: true on
skipped" convention itself should change (skipped ≠ passed).

**2c. Capability gate was mathematically unsatisfiable (fixed).**
`val 0.25 < threshold 0.83` forever, because the absolute E03 threshold was applied to
a mini-MMLU score of a different scale. Every campaign would be `eval_pass: false`
even with a perfect edit. Fixed with a retention gate (`abl/pristine ≥ 0.83`) when a
pristine baseline exists.

**2d. Pristine + ablated collects overwrote each other (fixed).**
Both wrote `campaigns/<slug>/latest-collect/bundle.json`. Run ablated, run pristine,
the baseline is gone. First campaign, first day, gone.

**2e. `passes: 2` in the config is fiction (warned, documented, not silently honored).**
Sweep and excise implement passes>1 as restore-then-reapply — same final weights as
one pass. The authoritative HF `abliteration_config.json` for the same recipe has NO
passes field: the winning edit was single-pass. The harness previously would have
pretended 2 passes happened. Now warns; campaign README documents the truth.

**2f. Held-out split details are invisible.**
`collect` prints `held_out_size` but never the actual prompts (I had to re-derive the
split in my transcript script to ensure the transcript covered the gate set). If the
split ever changes seed/size, bundle-vs-transcript claims silently mismatch.

**2g. Error paths drown in noise.**
`gate_perplexity_increase` runs LAST; a ZeroDivisionError there (which my empty-slice
bug can cause) would abort `collect` AFTER mmlu_mini, losing the whole bundle. Gates
should wrap per-gate exceptions and emit `passed: false, detail: "gate crashed (...)"`
instead of dying mid-bundle.

## 3. What would make the next campaign faster?

1. **`collect --transcript`**: save prompt+decoded-response for every gate generation
   (raw AND chat) — the single highest-value addition (§1a).
2. **`steer-test` / `alpha-curve` command**: α grid × few prompts, transcripts +
   refusal + PPL, no model re-save (use steering hooks / bias vectors) (§1e).
3. **`--from-directions`** in `abl`, and make `abl` save its own directions beside the
   manifest (§1d).
4. **Flavor consistency**: one flag (`--prompt-flavor raw|chat`) that directions,
   gates, and transcripts all honor (§1b).
5. **Modal wiring**: the repo already has `run_gates_modal.py`/`run_diag_modal.py`
   from the old loop — porting `abl.py` subcommands onto them is ~1-2h and turns a
   1.5h CPU campaign into ~10 min. The guided harness is uniquely suited to that
   because each command is a discrete, idempotent job.
6. **Inspect: print `layer_types` from the model config when present.** LFM2.5 ships
   it; the harness never reads it. It would have shown conv-vs-attention at a glance
   instead of me introspecting module names by hand.
7. **Per-gate try/except in `run_gates`** so one crashing gate can't void a bundle (§2g).

## 4. Did the campaign KB actually help?

Yes — materially, three times:

1. **The qwen landmine list directed the audit.** Knowing the class of failure
   ("non-square weights, silent skips, empty slices") is exactly why the
   MISSING-everywhere inspect output was treated as a red flag and chased down with
   module introspection instead of shrugged off.
2. **The alpha-sign-flip + operating-window lessons stopped me from "fixing" the
   campaign the wrong way.** When chat-refusal stayed 5/5, the naive response is
   "crank alpha". The qwen README's geometry section says plain-projection alpha>2
   amplifies and MPOA high-alpha collapses — so instead of burning cycles on α=4/6/10,
   the campaign investigated the flavor mismatch and produced the understanding above.
3. **The direction-source comment ("paired, not diff_means — refusal stayed 95%")**
   plus the config comments steered the direction method before any of my own
   evidence existed, and the paired separation profile then confirmed the KB's claim
   that the input-phase profile is confounded.

Where the KB fell short: the qwen campaign's eval flavor (raw or chat) is never
stated in its numbers — its "5/5 refused" gate numbers are chat-flavored too, and
steering probes were raw. The KB contains the raw-vs-chat trap implicitly but never
names it, which is why I walked into it for 30 minutes. **Recommendation: the KB
template (and gate details) should record the prompt flavor per number.**

---

## Short version

- The harness ran an inspected, guided campaign end-to-end and (after fixes) did not
  fabricate a result — but its **defaults hid the entire interesting story** (no
  transcripts, raw directions vs chat gates).
- The single scariest bug — silent no-op ablation on non-llama weight names — was
  exactly the class the qwen campaign catalogued, and it needed a hard-fail at apply
  plus a weight-change fingerprint to be permanently dead.
- Next build: `--transcript`, `--from-directions`, a flavor flag, and an
  alpha-response curve command. Then Modal.