# MiniCPM5-1B Abliteration Experiment Log

## Model Info
- **Model**: openbmb/MiniCPM5-1B (1.08B params, 24 layers, hidden=1536, LlamaForCausalLM)
- **Architecture**: Dense decoder, standard Llama attention + MLP
- **Key feature**: Built-in ```` chat template with `enable_thinking` toggle
- **GPU**: NVIDIA L4 (24GB), Modal cloud
- **Method**: diff-of-means weight projection on o_proj + down_proj

## Experiment History

### Phase 1: Initial Pipeline (v1-v2, standalone script)
| Run | Alpha | Layers | Probe Mode | HarmBench | MMLU/MATH | Notes |
|-----|-------|--------|------------|-----------|-----------|-------|
| v1 | 0.5 | 10 (23-14) | raw text | 88.5% (177/200) | 32% MMLU | First success. 10 layers = too aggressive |
| v2 | 0.2 | 5 (23-19) | raw text | 81% (162/200) | 36% MMLU | Lower alpha helps MMLU slightly |

### Phase 2: Chat Template Experiments (v3 series)
| Run | Alpha | Layers | Probe | Eval | HarmBench | MMLU/MATH | Finding |
|-----|-------|--------|-------|------|-----------|-----------|---------|
| v3a | 0.2 | 5 | chat no-think | chat no-think | 35% (70/200) | 26% MMLU | Mode mismatch kills HarmBench |
| v3b | 0.5 | 5 | chat no-think | chat no-think | 33% (65/200) | 22% MMLU | Worse with higher alpha |
| v3c | 0.5 | 10 | chat default | chat default | 88.5% (177/200) | 28% MMLU | Default template works |
| v3d | 0.5 | 10 | chat default | MMLU no-think | 88.5% (177/200) | 22% MMLU | No-think MMLU worse |

### Phase 3: MiniCPM5 Official Benchmarks (v3 final)
| Run | Alpha | Layers | HarmBench | MATH-500 | BBH | GPQA |
|-----|-------|--------|-----------|----------|-----|------|
| v3-math-think | 0.5 | 10 | 88.5% (177/200) | 26% (13/50) | 60% (9/15) | skipped |

**Original scores**: MATH-500=91.6%, BBH=71.9%, GPQA=26.3%

### Phase 4: Layer Ablation Sweep (v4)
[SINGLE RUN: same probe, same distill, restore pristine between configs]

| Config | Layers | Alpha | HarmBench (80) | MATH-500 (30) | Verdict |
|--------|--------|-------|----------------|---------------|---------|
| Baseline | - | - | ? | ? | Unabliterated baseline |
| L23-only | [23] | 0.5 | ? | ? | - |
| L23-22 | [23,22] | 0.5 | ? | ? | - |
| L23-21 | [23,22,21] | 0.5 | ? | ? | - |
| L23-20 | [23..20] | 0.5 | ? | ? | - |
| L23-19 | [23..19] | 0.5 | ? | ? | - |

## Key Findings

### Finding 6: Unabliterated MiniCPM5-1B has weak safety alignment
**Evidence**: Baseline HarmBench = 67/80 unblocked (83.8%, only 16.2% refusal rate).  
**Implication**: The model's refusal circuit is weak to begin with. Our "88.5% unblocked" represents +4.7% improvement over baseline, not +88.5% over full refusal.  
**Important**: This changes the framing — we're not "breaking" strong safety, we're removing the last remnants of RLHF alignment from a weakly-aligned model. The delta to report is +4.7pp, not +88.5pp.

### Finding 7: MATH-500 extraction must use regex for \\boxed{} in thinking output
**Evidence**: Baseline MATH = 5/30 (16.7%) with last-line extraction, should be ~90% (27/30).  
**Cause**: Thinking mode outputs ``  ...  ``, with `\\boxed{42}` in the middle, not on the last line.  
**Fix**: Use `re.search(r'\\boxed\\{([^}]+)\\}', resp)` before falling back to last-line extraction.
**Evidence**: v3a/b (mismatched) gave 33-35% HarmBench vs 88.5% (matched).  
**Cause**: The refusal direction is encoded in thinking-mode activations. Probing without chat template captures different activations.  
**Rule**: PROBE and EVAL must use identical chat template settings.

### Finding 2: Thinking mode preserves capabilities, no-think kills them
**Evidence**: MATH-500 with `enable_thinking=False` = 16% vs `enable_thinking=True` = 26%.  
**Mechanism**: MiniCPM5's math/reasoning ability lives in the `` block. Disabling thinking removes the model's reasoning capability entirely.  
**Rule**: For capability benchmarks, use thinking mode (matching original eval). For HarmBench, use default template (which includes thinking).

### Finding 3: L23 has extreme separation spike (score 85-96)
**Evidence**: Across all runs, L23 scores 85-96 while L22 scores 60-69. Ratio of L23/L22 ≈ 1.4x.  
**Implication**: L23 contains both refusal direction AND critical reasoning circuits. Abliterating it damages reasoning.  
**Hypothesis**: A surgical strike on L23 alone (or L23-22) at low alpha may suppress refusal while keeping reasoning intact.

### Finding 4: BBH is relatively robust (-12% with 10 layers)
**Evidence**: BBH 60% vs original 71.9% — only 12% degradation with 10 layers at alpha=0.5.  
**Implication**: BBH tasks (navigation, boolean logic, date understanding) use different circuits than refusal. Good sign for capability preservation.

### Finding 5: MATH-500 collapses with 10 layers (-66%)
**Evidence**: MATH 26% vs original 91.6% — catastrophic.  
**Implication**: Math reasoning shares circuits with refusal in layers 14-23. Reducing target layers is essential.

## Absolver Pipeline Bugs (for LangGraph improvement)

1. **summon.py:156** — `device_map="auto"` splits small models across CPU/GPU
2. **probe.py:87** — Missing `.squeeze(0)` causes 3D activation stacks
3. **excise.py:179** — Missing `.data` on weight params causes autograd crash  
4. **config.py:236** — CWD-relative path resolution fails on Modal
5. **connectors/modal_runner.py** — Missing `.gitignore` uploads 45K files
6. **run.py CLI** — argparse clashes with Modal typer
7. **excise.py:132** — Rank guard truncates direction to wrong dimension

## Next Steps
- [ ] Analyze v4 sweep results
- [ ] Find sweet spot: layers/alpha that gives >80% HarmBench with MATH >70%
- [ ] Fix GPQA column name bug
- [ ] Push best model to HF Hub
- [ ] Port findings into LangGraph absolver improvements
