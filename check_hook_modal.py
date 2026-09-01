"""Verify the Modal container has the probe.py fix: print act shapes."""
import modal
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11", force_build=True)
    .uv_pip_install("torch>=2.0", "transformers>=4.30", "langgraph>=0.3",
                    "pydantic>=2", "pyyaml>=6", "huggingface-hub>=0.20",
                    "numpy>=1.24", "accelerate>=0.20")
    .env({"PYTHONPATH": "/absolver"})
    .add_local_dir(str(PROJECT_DIR), remote_path="/absolver",
                   ignore=[".venv", ".git", "__pycache__", "*.pyc", ".aiwg",
                           ".pytest_cache", ".mypy_cache", ".ruff_cache",
                           "abliterated_models", "experiments"])
)

app = modal.App("absolver-hook-check")

@app.function(image=image, gpu="L4", timeout=1200, retries=0)
def check() -> dict:
    os.chdir("/absolver")
    sys.path.insert(0, "/absolver")
    import torch
    from collections import defaultdict
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from probe import _make_hook, _find_layers

    m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.bfloat16
    ).to("cuda")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    layers = _find_layers(m, "dense")
    store = defaultdict(list)
    h = layers[0].register_forward_hook(_make_hook(0, store))
    inp = tok("I hate you", return_tensors="pt", truncation=True, max_length=128)
    inp = {k: v.to("cuda") for k, v in inp.items() if hasattr(v, "to")}
    with torch.no_grad():
        m(**inp)
    h.remove()
    shape = tuple(store[0][0].shape)

    # Also confirm the fixed source is in the container
    import inspect
    from probe import _make_hook as mh
    src = inspect.getsource(mh)
    has_squeeze = ".squeeze(0)" in src

    return f"act_shape={shape} hook_has_squeeze={has_squeeze} git_head={os.popen('git log --oneline -1 2>/dev/null').read().strip()}"
