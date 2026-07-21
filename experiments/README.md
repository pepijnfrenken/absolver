# Absolver Experiments

## Single-Layer Rank-1 Residual Projection Sweep

**Goal**: Answer: does *any* localized, low-strength intervention provide a real Pareto improvement on Ternary-Bonsai-27B?

### Protocol

1. Load the pinned model in full bf16 on a device with ≥80GB VRAM
2. Establish a thinking-aware baseline
3. Build matched harmful/hard-harmless direction-training pairs
4. Derive one direction independently at each layer L9–L17
5. Apply temporary, input-dependent projection:
   ```
   h' = h - α(h^T d)d
   ```
   one layer at a time, over a normalized alpha grid
6. Fast-screen each candidate with:
   - held-out harmful refusal/compliance classifier
   - clean token-level KL and NLL
   - termination and reasoning-length metrics
7. Fully evaluate only feasible finalists on:
   - 200 held-out HarmBench prompts with actual behavior classifier
   - MMLU-Redux generation with reasoning-aware parsing
   - hard-harmless over-refusal
   - clean generation quality

### Running on Lightning AI

```bash
# Set up
pip install torch transformers datasets tqdm

# Run the sweep
python bonsai_sweep.py --layers 9 10 11 12 13 14 15 16 17 --alpha_grid 0.1 0.3 0.5 0.8 1.0

# Analyze results
python bonsai_sweep.py --analyze /path/to/results.json
```

### Running on Molab

Upload `bonsai_sweep.py` and `requirements.txt` to a Marimo notebook,
then execute cell by cell.
