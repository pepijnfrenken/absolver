# Benchmark replication — decision trail

Mission: replicate LiquidAI's posted suite (pristine vs ablated), capability-hit
analysis, card repair + relink. See `/tmp/benchmark-replication-mission.md`.

| # | decision | change | before | after | verdict | note |
|---|----------|--------|--------|-------|---------|------|
| 1 | Use lm-eval-harness 0.4.13 (installed locally for inventory) as the harness; tasks verified: `gpqa_diamond_n_shot`, `aime25`, `ifeval`, `mmlu_pro` all exist. BFCLv3/Multi-IF/IFBench absent → will report "not replicated". | benchmarks/run_lmeval_modal.py | — | — | kept | GPQA via lm-eval (5-shot acc_norm per task config `gpqa_diamond_n_shot` = the classical task; AA methodology not byte-equivalent — will say so). |
| 2 | Both models load via transformers 5.14.1 native `Lfm2ForCausalLM` (no modeling .py on any of the 3 hub repos; trust_remote_code is a no-op). Pin transformers==5.14.1 + torch==2.13.0 (mirror the harness image). | image pins | — | — | kept | verified via repo file listings; pilot confirms load. |
| 3 | Pristine == ablated on HF repo file-level; KB card == pushed README byte-identical (sha match). Card edits go into KB copy, then pushed from that exact content. | — | — | — | kept | |
| 4 | Identical conditions protocol: same seed 1234, same tasks, same batch_size auto, no gen_kwargs overrides (per-task YAML defaults), bf16, L4. | runner | — | — | kept | aime25 default max_gen_toks 32768 kept (until tokens bound gen in practice). |
| 5 | mmlu_pro full (12,432) NOT feasible on L4: pilot measured 5.3s/sample batch-1 generate-until → 19h/model. Documented subset: `--limit 100` = 100/subject × 14 = 1,400 samples, first-N, seed 1234, IDENTICAL on both models. | benchmarks runner | pilot: 350 smp/31:54 | subset 1,400 | — | kept | full-run cost documented in card. |
| 6 | GPQA: task default is 0-shot (`gpqa_diamond_n_shot` config num_fewshot: 0). AA/LiquidAI use 5-shot → re-ran `--num-fewshot 5` for closer parity; 0-shot numbers kept as secondary evidence. | runner +num_fewshot | 0-shot: 21.21/23.74 | 5-shot run in flight | — | kept | card reports the 5-shot run as primary; says AA methodology is not byte-identical. |
| 7 | IFEval full (541) feasible; both models run. prompt-strict: pristine 54.71 → ablated 52.68 (Δ -2.03pp, 96.3% retained) → preserved. | measured | — | — | — | kept | identical conditions, seed 1234. |
| 8 | Pilot GPQA/AIME full (0-shot): GPQA 21.21→23.74 (+2.53pp, ~1.1×, within stderr), AIME25 0.00→6.67 (0/30→2/30, binomial noise). No hit on these; ablated ≥ pristine. | measured | — | — | — | kept | small-n caveat on card. |
| 9 | ifeval needs langdetect+immutabledict+nltk, absent from plain lm_eval install → added to image deps; JSON dump needs `default=str` (torch/numpy dtypes in config); tags must be per-model (parallel runs clobbered `results.json`). | runner | — | — | — | kept applied | 3 failed launches, all cheap; recorded for future runs. |