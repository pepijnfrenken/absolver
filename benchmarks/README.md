# Benchmark replication — LFM2.5-1.2B-Instruct (pristine vs ablated)

Replicates the runnable subset of LiquidAI's posted benchmark suite
([LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct))
on the ablated model `PinoCookie/LFM2.5-1.2B-Instruct-Abliterated`, always
against the pristine base under **identical conditions**.

LiquidAI's posted suite:

| Model | GPQA | MMLU-Pro | IFEval | IFBench | Multi-IF | AIME25 | BFCLv3 |
|---|---|---|---|---|---|---|---|
| LFM2.5-1.2B-Instruct | 38.89 | 44.35 | 86.23 | 47.33 | 60.98 | 14.00 | 49.12 |

## What we replicate — and what we honestly cannot

| LiquidAI column | lm-eval task | Replicated? |
|---|---|---|
| GPQA | `gpqa_diamond_n_shot` (GPQA Diamond, 198 q, 5-shot, acc_norm) | yes — note: lm-eval prompt, not byte-identical to ArtificialAnalysis methodology; approximated |
| MMLU-Pro | `mmlu_pro` (TIGER-Lab/MMLU-Pro, 14 subjects) | yes — full or documented subset (see below) |
| AIME25 | `aime25` (math-ai/aime25, 30 problems) | yes |
| IFEval | `ifeval` (google/IFEval, 541 prompts, strict/loose) | yes |
| IFBench | — (custom harness) | **not replicated** — no lm-eval task; custom eval suite |
| Multi-IF | — (custom harness) | **not replicated** — no lm-eval task; multi-turn IF suite |
| BFCLv3 | — (custom harness) | **not replicated** — no lm-eval task; requires tool-call API harness |

No numbers are copied from LiquidAI's posting or faked: every reported number
comes from a real lm-eval run by `run_lmeval_modal.py` whose JSON log is
quoted.

## Protocol (identical conditions)

- Harness: lm-eval 0.4.13, HF transformers 5.14.1, torch 2.13.0, bfloat16
- GPU: Modal L4 (24 GB), `batch_size: auto`
- Seed: `1234`; no `gen_kwargs` overrides (per-task YAML defaults, incl.
  `do_sample: false`); same tasks/args for both models
- Models: `LiquidAI/LFM2.5-1.2B-Instruct` (pristine),
  `PinoCookie/LFM2.5-1.2B-Instruct-Abliterated` (ablated, bf16 identical
  weights to reference edit, rel_l2 0.0013–0.0023)
- Result JSON logs: `benchmarks/results/` (copied from the Modal
  `absolver-phase2` volume path `benchmarks/<tag>/results.json`)

## Reproduce

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/pino/absolver
# pilot / full runs — one call spawns both models in parallel
.venv/bin/modal run benchmarks/run_lmeval_modal.py \
  --model both --tasks gpqa_diamond_n_shot,aime25 --tag run1
.venv/bin/modal run benchmarks/run_lmeval_modal.py \
  --model both --tasks mmlu_pro,ifeval --limit 3000 --tag run2
# pull logs
modal volume get absolver-phase2 benchmarks/run1/pristine/results.json benchmarks/results/run1-pristine.json
# analyze
.venv/bin/python benchmarks/analyze_results.py \
  benchmarks/results/run1-pristine.json benchmarks/results/run1-ablated.json
```

## Files

- `run_lmeval_modal.py` — Modal runner (eval + `--push-readme`)
- `analyze_results.py` — delta table + per-capability verdicts from two JSON logs
- `results/` — pulled JSON logs (real runs, sha-verifiable against volume)