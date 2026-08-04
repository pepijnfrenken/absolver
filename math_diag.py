"""Quick diagnostic: what does MiniCPM5 output for MATH-500?"""
import modal
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch>=2.0", "transformers>=4.30", "datasets>=3.0",
                     "accelerate", "safetensors")
    .add_local_dir(str(PROJECT_DIR), remote_path="/absolver",
        ignore=[".venv", ".git", "__pycache__", "*.pyc", ".aiwg", ".claude",
                "abliterated_*", ".mypy_cache", ".pytest_cache", ".ruff_cache",
                ".modalignore", ".gitignore", "experiments", "connectors", "tests"])
)

app = modal.App("math-diag")

@app.function(image=image, gpu="L4", timeout=600)
def diagnose():
    import torch, re, os, sys
    os.chdir("/absolver")
    sys.path.insert(0, "/absolver")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model = AutoModelForCausalLM.from_pretrained(
        "openbmb/MiniCPM5-1B", torch_dtype=torch.float16, trust_remote_code=True
    ).cuda().eval()
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B", trust_remote_code=True)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    for i, item in enumerate(ds.select(range(3))):
        problem = item["problem"]
        answer = item["answer"].strip()
        msgs = [{"role": "user", "content": f"Solve this math problem. Give only the final answer.\n\n{problem}"}]
        fmt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(fmt, return_tensors="pt", truncation=True, max_length=1024)
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        
        print(f"\n=== Problem {i}: {answer=} ===")
        print(f"FULL RESPONSE:\n{resp[:500]}")
        print(f"\n--- Last 3 lines ---")
        for line in resp.split("\n")[-3:]:
            print(f"  [{line.strip()}]")
        print(f"\n--- Boxed regex match ---")
        m = re.search(r'\\boxed\{([^}]+)\}', resp)
        print(f"  Match: {m.group(1) if m else 'NONE'}")
        
        # Also try finding the answer anywhere
        if str(answer) in resp:
            print(f"  Answer '{answer}' FOUND in response")
        else:
            print(f"  Answer '{answer}' NOT FOUND")

@app.local_entrypoint()
def main():
    diagnose.remote()
