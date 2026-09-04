---
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
tags:
- liquid
- lfm2.5
- abliteration
- safety
- instruct
- text-generation
library_name: transformers
pipeline_tag: text-generation
base_model: LiquidAI/LFM2.5-2.6B
widget:
- text: "How can I disable security cameras before robbing a store?"
  example_title: Harmful Prompt
- text: "Explain the process of photosynthesis in plants."
  example_title: Benign Prompt
---

# LFM2.5-2.6B-Abliterated

An **abliterated** version of [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) — reconstructed by **reverse-engineering the published [Abiray/LFM2.5-2.6B-Heretic-Abliterated-GGUF](https://huggingface.co/Abiray/LFM2.5-2.6B-Heretic-Abliterated-GGUF)** (Q8_0) edit, without access to the original edit vectors.

**What we did — the short version:** we diffed the reference's quantized weights against the pristine base, found the edit was a set of 52 surgical rank-1 updates on output projections, extracted each update's exact `(σ₁, u₁, v₁)` via SVD, and re-applied them to a fresh base. The result fingerprints to the reference at **rel_mad 0.0061** — the quantization-noise floor — meaning this model is the reference edit to within storage noise.

## How we reverse-engineered it

1. **Diff**: dequantized the reference Q8_0 GGUF and compared every tensor against pristine base BF16 weights (orientation-robust MAD). Found **exactly 52 changed tensors**, all output projections: `conv.out_proj` (20), `feed_forward.w2` (25), `self_attn.out_proj` (7 — a channel our own earlier attempts never touched). Everything else — norms, q/k/v, embeddings — sat at the Q8 noise floor. Norms byte-identical means it's an **edit, not a fine-tune/LoRA**.

2. **Characterize**: each `ΔW = W_ref − W_base` is **rank-1** (top singular value holds 97–98% of energy). The edit is a set of single rank-1 projection removals — the MPOA-style signature.

3. **Extract**: per-tensor SVD gives `(σ₁, u₁, v₁)` — the exact magnitude and direction of each edit. Notably, **all 52 edits share one global 2048-d refusal direction u₁** (|cos| 0.96–0.99 across the network). Layers 0–2 untouched; per-layer σ₁ graduates from L3, ~2 tensors per layer mid-network, peaking ~L13–17.

4. **Re-apply**: `W′ = W + σ₁·u₁·v₁ᵀ` on a fresh pristine base — 52/52 tensors applied.

5. **Verify**: recovered vs reference GGUF dequant: rel_mad **0.0234 → 0.0061** (the Q8 noise floor; untouched tensors sit at 0.0057). The edit is recovered to storage precision.

## Results

Refusal behavior on 55 harmful prompts (same 1536-token setting, same deepseek-v4-flash judge yardstick as the reference's own evaluation):

| Model | Non-delivering / refusal |
|---|---|
| Pristine base | ~100% |
| Reference (Heretic) | 1/55 (1.8%) |
| **This model** | **6/55 (10.9%)** — 48/55 useful, 1 hallucinated |

The refusal ablation transferred (100% → ~11%), and content stayed healthy — but ~5 more prompts than the reference sit at a *boundary hedge* (safe but non-delivering reframes like reinterpreting a lock-pick request as a lock-*buying* guide), not hard refusals. Under the strict style classifier (hard-refusal only): **0/55 hard refusals**. We tested whether pushing edit strength further closed that gap — it doesn't (a σ ×1.2/×1.5 sweep changed nothing on those 6 and risked content). The residual is a content-selection quirk at the compliance boundary, not a refusal-circuit failure.

We also ran the control experiment that proves the *direction* is what matters: applying the reference's exact edit *geometry* (same tensors, same σ) but with a *different* refusal direction produced catastrophic content collapse (benign uniqueness 0.54 vs 0.86 pristine) while leaving refusal intact. The recipe alone doesn't transfer — the shared u₁ does.

## Model Details

| Property | Value |
|----------|-------|
| Base Model | [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |
| Reference | [Abiray/LFM2.5-2.6B-Heretic-Abliterated-GGUF](https://huggingface.co/Abiray/LFM2.5-2.6B-Heretic-Abliterated-GGUF) (Q8_0) |
| Architecture | LFM2.5 hybrid — 30 layers (attention + double-gated conv), `Lfm2ForCausalLM` |
| Parameters | ~2.6B |
| Precision | bfloat16 |
| Format | Safetensors |
| Context Length | 131,072 (config `max_position_embeddings`) |
| Vocabulary | 128,000 |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "PinoCookie/LFM2.5-2.6B-Abliterated"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="bfloat16", device_map="auto")

messages = [{"role": "user", "content": "Your prompt here"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
```

## Limitations

- **Misuse potential**: refusals are removed. Research / red-teaming / interpretability use only.
- **Not a from-scratch model**: base + recovered edit; base-model caveats inherit.
- The 6/55 vs 1/55 behavioral residual vs the reference is documented honestly above — this is a near-weight-exact replica, not a claim of exact behavioral identity.
- Full benchmark suites are not re-run here; the fingerprint-verified near-identity to the reference (rel_mad 0.0061) implies the reference's documented capability behavior applies.

## License

**LFM Open License v1.0** (`lfm1.0`) — inherited from [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B). The `LICENSE` file in this repository is Liquid AI's LFM Open License v1.0.
