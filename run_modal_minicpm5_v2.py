"""Direct Modal abliteration of MiniCPM5-1B — bypasses LangGraph for full control."""
import modal
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent  # absolver/

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.0", "transformers>=4.30",
        "pydantic>=2", "pyyaml>=6", "huggingface-hub>=0.20", "numpy>=1.24",
        "accelerate", "safetensors",
    )
    .add_local_dir(
        str(PROJECT_DIR),
        remote_path="/absolver",
        ignore=[
            ".venv", ".git", "__pycache__", "*.pyc",
            ".aiwg", ".claude", "abliterated_*",
            ".mypy_cache", ".pytest_cache", ".ruff_cache",
            ".modalignore", ".gitignore", "experiments",
            "connectors", "tests",
        ],
    )
)

app = modal.App("absolver-minicpm5-v2")


@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def abliterate_minicpm5() -> dict:
    import json, os, sys, time
    import torch
    import yaml
    from collections import defaultdict

    os.chdir("/absolver")
    sys.path.insert(0, "/absolver")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # --- Load config ---
    with open("models/minicpm5-1b.yaml") as f:
        cfg = yaml.safe_load(f)

    MODEL_ID = cfg["model_id"]  # openbmb/MiniCPM5-1B
    ALPHA = cfg["alpha"]         # 0.5
    PASSES = cfg["passes"]       # 2
    TARGET_WEIGHTS = cfg["target_weights"]  # ["o_proj", "down_proj"]
    BATCH_SIZE = cfg["batch_size"]  # 4

    print(f"=== Absolver: {MODEL_ID} ===", flush=True)

    # --- Load model directly (no device_map, no accelerate) ---
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = model.cuda()
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s on {torch.cuda.get_device_name(0)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- Detect architecture ---
    layers = None
    for name, mod in model.named_children():
        if hasattr(mod, "layers"):
            layers = mod.layers
            break
    if layers is None and hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers

    num_layers = len(layers)
    hidden_size = layers[0].input_layernorm.weight.shape[0] if hasattr(layers[0], "input_layernorm") else model.config.hidden_size
    print(f"Arch: {num_layers} layers, hidden={hidden_size}", flush=True)

    # --- Probe: harvest activations ---
    from prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS

    harmful_prompts = list(DEFAULT_HARMFUL)[:cfg["n_probe_prompts"]]
    harmless_prompts = list(DEFAULT_HARMLESS)[:cfg["n_probe_prompts"]]

    harm_acts: dict[int, list[torch.Tensor]] = defaultdict(list)
    harmless_acts: dict[int, list[torch.Tensor]] = defaultdict(list)

    def make_hook(layer_idx: int, store: dict):
        def hook(_module, _inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.dim() == 3:
                store[layer_idx].append(hs[:, -1, :].detach().squeeze(0).cpu().float())
            elif hs.dim() == 2:
                store[layer_idx].append(hs.detach().squeeze(0).cpu().float())
        return hook

    def run_prompts(prompts, store):
        handles = [layers[i].register_forward_hook(make_hook(i, store)) for i in range(num_layers)]
        try:
            for p in prompts:
                inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
                inp = {k: v.cuda() for k, v in inp.items()}
                with torch.no_grad():
                    model(**inp)
        finally:
            for h in handles:
                h.remove()

    print("PROBE: harmful...", flush=True)
    run_prompts(harmful_prompts, harm_acts)
    print("PROBE: harmless...", flush=True)
    run_prompts(harmless_prompts, harmless_acts)

    print(f"PROBE done: {len(harm_acts)} harm layers, {len(harmless_acts)} harmless layers", flush=True)

    # --- Distill: diff-of-means per layer ---
    print("DISTILL: computing refusal directions...", flush=True)

    directions = {}
    scores = {}

    for i in range(num_layers):
        h_list = harm_acts.get(i)
        b_list = harmless_acts.get(i)
        if not h_list or not b_list:
            continue

        harm_stack = torch.stack(h_list).float().cuda()
        harmless_stack = torch.stack(b_list).float().cuda()

        diff = harm_stack.mean(dim=0) - harmless_stack.mean(dim=0)
        norm = diff.norm() + 1e-8
        directions[i] = (diff / norm).cpu()
        scores[i] = norm.item()

    # Rank by separation score
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    # Select target layers: top-5 with score > threshold
    threshold = cfg.get("separation_threshold", 3.0)
    target_layers = [li for li, sc in ranked[:10] if sc > threshold]
    if not target_layers:
        target_layers = [li for li, sc in ranked[:5]]

    print(f"Top separation scores: {[(li, f'{sc:.1f}') for li, sc in ranked[:5]]}", flush=True)
    print(f"Target layers: {target_layers}", flush=True)

    # --- Excise: weight projection ---
    print(f"EXCISE: projecting {len(target_layers)} layers, alpha={ALPHA}, {PASSES} passes...", flush=True)

    def project_weight(weight: torch.Tensor, direction: torch.Tensor, alpha: float):
        """Project weight away from refusal direction using absolver's formula."""
        d = direction.to(weight.device, weight.dtype)
        # Ensure 1D
        if d.dim() > 1:
            d = d.reshape(-1)[:weight.shape[0]]
        d = d / d.norm().clamp(min=1e-8)
        # W -= alpha * d ⊗ (d @ W)
        proj = d @ weight  # [hidden] @ [hidden, inner] -> [inner]
        weight.sub_(alpha * torch.einsum("i,j->ij", d, proj))

    for pass_idx in range(PASSES):
        print(f"  Pass {pass_idx + 1}/{PASSES}...", flush=True)
        for li in target_layers:
            layer = layers[li]
            d = directions[li].cuda()

            for wname in TARGET_WEIGHTS:
                if wname == "o_proj":
                    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                        project_weight(layer.self_attn.o_proj.weight.data, d, ALPHA)
                    if hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
                        project_weight(layer.linear_attn.out_proj.weight.data, d, ALPHA)
                elif wname == "down_proj":
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                        project_weight(layer.mlp.down_proj.weight.data, d, ALPHA)
                    if hasattr(layer, "feed_forward") and hasattr(layer.feed_forward, "down_proj"):
                        project_weight(layer.feed_forward.down_proj.weight.data, d, ALPHA)

    # --- Verify: check refusal rate ---
    print("VERIFY: testing refusal rate...", flush=True)

    verify_prompts = list(DEFAULT_HARMFUL)[:cfg["n_verify_prompts"]]

    refusal_count = 0
    refusal_keywords = ["sorry", "cannot", "can't", "unable", "not able",
                        "apologize", "illegal", "against", "不", "抱歉", "不能"]

    for p in verify_prompts:
        inp = tok(p, return_tensors="pt", truncation=True, max_length=128)
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0], skip_special_tokens=True)
        if any(kw in resp.lower() for kw in refusal_keywords):
            refusal_count += 1

    refusal_rate = refusal_count / len(verify_prompts)
    print(f"Refusal rate: {refusal_count}/{len(verify_prompts)} = {refusal_rate:.1%}", flush=True)

    # --- Save model ---
    print("Saving abliterated model...", flush=True)
    save_path = "/tmp/minicpm5-1b-abliterated"
    model.save_pretrained(save_path, safe_serialization=True)
    tok.save_pretrained(save_path)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    elapsed = time.perf_counter() - t0

    result = {
        "model_id": MODEL_ID,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "max_separation": max(scores.values()) if scores else 0,
        "target_layers": target_layers,
        "refusal_rate": refusal_rate,
        "passes": PASSES,
        "alpha": ALPHA,
        "gpu": torch.cuda.get_device_name(0),
        "peak_memory_gb": round(peak_mem, 2),
        "elapsed_seconds": round(elapsed, 1),
        "save_path": save_path,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main():
    result = abliterate_minicpm5.remote()
    print(f"\n✅ Done. Refusal rate: {result.get('refusal_rate', '?')}")


# Also support direct `modal run` for simplicity
@app.function(image=image, gpu="L4", timeout=3600,
              secrets=[modal.Secret.from_name("huggingface-token")])
def run():
    """Entrypoint for `modal run`."""
    return abliterate_minicpm5.remote()
