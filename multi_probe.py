"""
Phase 1: Multi-direction probe — collect hidden states for 5 data categories.
Run once, save all activations to disk for analysis.
"""
import modal, torch, os, json
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch>=2.0", "transformers>=4.30", "datasets>=3.0",
                     "accelerate", "safetensors", "pyyaml", "numpy")
    .add_local_dir(str(PROJECT_DIR), remote_path="/absolver",
        ignore=[".venv", ".git", "__pycache__", "*.pyc", ".aiwg", ".claude",
                "abliterated_*", ".mypy_cache", ".pytest_cache", ".ruff_cache"])
)

app = modal.App("multi-probe-v1")

@app.function(image=image, gpu="L4", timeout=1800,
              secrets=[modal.Secret.from_name("huggingface-token")])
def multi_probe():
    os.chdir("/absolver")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS

    MODEL_ID = "openbmb/MiniCPM5-1B"
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
    ).cuda().eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    layers = model.model.layers
    num_layers = len(layers)
    hidden_size = layers[0].input_layernorm.weight.shape[0]
    print(f"Arch: {num_layers} layers, hidden={hidden_size}")

    # --- Data categories ---
    categories = {
        "harmful": list(DEFAULT_HARMFUL)[:30],
        "harmless": list(DEFAULT_HARMLESS)[:30],
    }

    # MATH-500
    ds_math = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(30))
    categories["math"] = [item["problem"] for item in ds_math]

    # GPQA
    try:
        ds_gpqa = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train").select(range(30))
        categories["gpqa"] = [item["Question"] for item in ds_gpqa]
    except Exception:
        print("GPQA unavailable, skipping")
        categories["gpqa"] = []

    # BBH
    bbh_prompts = []
    for task in ["boolean_expressions", "navigate", "date_understanding"]:
        try:
            ds = load_dataset("lukaemon/bbh", task, split="test").select(range(10))
            bbh_prompts.extend([item["input"] for item in ds])
        except Exception:
            pass
    categories["bbh"] = bbh_prompts[:30]

    # --- Probe all categories ---
    all_acts = {}

    def make_hook(li, store):
        def hook(_m, _i, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.dim() == 3:
                store[li].append(hs[:, -1, :].detach().squeeze(0).cpu().float())
            elif hs.dim() == 2:
                store[li].append(hs.detach().squeeze(0).cpu().float())
        return hook

    for cat_name, prompts in categories.items():
        if not prompts: continue
        print(f"Probing {cat_name} ({len(prompts)} prompts)...")
        store = defaultdict(list)
        handles = [layers[i].register_forward_hook(make_hook(i, store)) for i in range(num_layers)]
        try:
            for p in prompts:
                msgs = [{"role": "user", "content": p}]
                fmt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = tok(fmt, return_tensors="pt", truncation=True, max_length=256)
                inp = {k: v.cuda() for k, v in inp.items()}
                with torch.no_grad():
                    model(**inp)
        finally:
            for h in handles: h.remove()

        # Stack per layer: [n_prompts, hidden]
        all_acts[cat_name] = {li: torch.stack(v).float() for li, v in store.items()}
        print(f"  Collected {len(all_acts[cat_name])} layers")

    # --- Save to disk ---
    out = {}
    for cat_name, layer_acts in all_acts.items():
        out[cat_name] = {str(li): v.numpy().tolist() for li, v in layer_acts.items()}
    out["meta"] = {"model_id": MODEL_ID, "num_layers": num_layers,
                    "hidden_size": hidden_size, "categories": list(all_acts.keys())}

    with open("/tmp/multi_probe_acts.json", "w") as f:
        json.dump(out, f)
    
    # Also save quick summary stats
    summary = {}
    for cat_name in all_acts:
        summary[cat_name] = {}
        for li in range(num_layers):
            if li in all_acts[cat_name]:
                acts = all_acts[cat_name][li]  # [n, d]
                summary[cat_name][str(li)] = {
                    "mean_norm": acts.mean(dim=0).norm().item(),
                    "std": acts.std(dim=0).mean().item(),
                    "n": acts.shape[0],
                }
    with open("/tmp/multi_probe_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved to /tmp/multi_probe_acts.json and /tmp/multi_probe_summary.json")
    print(f"Categories: {list(all_acts.keys())}")
    return summary["harmful"]["23"]["mean_norm"]  # quick sanity

@app.local_entrypoint()
def main():
    result = multi_probe.remote()
    print(f"Done. Harmful L23 mean norm: {result:.2f}")
