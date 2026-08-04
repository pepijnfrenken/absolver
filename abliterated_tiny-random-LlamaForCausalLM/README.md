---
license: apache-2.0
language:
- en
tags:
- abliteration
- safety
- absolver
- alignment-removal
base_model: hf-internal-testing/tiny-random-LlamaForCausalLM
pipeline_tag: text-generation
---

# tiny-random-LlamaForCausalLM Abliterated

Abliterated version of [hf-internal-testing/tiny-random-LlamaForCausalLM](https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM) — refusal direction removed via the [**Absolver**](https://github.com/) pipeline (weight projection / steering ablation).

> **Done by Absolver** — the autonomous abliteration pipeline: SUMMON → PROBE → DISTILL → SWEEP → EXCISE → VERIFY → JUDGE → REBIRTH.

## Method
- Pipeline: SUMMON → PROBE → DISTILL → SWEEP → EXCISE → VERIFY → JUDGE → REBIRTH (LangGraph)
- Method: mpoa (sweep-selected across advanced / bias_vectors / direct_ablation / projected / lora)
- Direction extraction: paired, 3 directions
- Projection strength α = 1.0, 1 pass(es)
- Target layers: [0]
- Target weights: ['o_proj']

## Results (model-card comparison)
| Benchmark     | Abliterated | Model Card | Δ       |
|---|---|---|---|
| —            | —           | —          | —      |

- Refusal rate: 0.000 (LLM-judged on harmful prompts)

## Notes
- Benchmarks use compact built-in subsets (25-50 samples, greedy, no-thinking); model-card numbers are full-set + thinking mode, so a systematic gap is expected.
- Capability impact is measured per-benchmark (see Δ column); the sweep selected the config that best preserves quality while removing refusal.
