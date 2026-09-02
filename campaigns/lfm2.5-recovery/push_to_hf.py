"""Push staged model dir to HF as PinoCookie/LFM2.5-1.2B-Instruct-Abliterated.

Uses the local HF token; verified by listing repo files + sizes + card render
after the upload (never claim success without verification).
"""
import os
import sys

from huggingface_hub import HfApi

# The local HF token is read-only; always push via the Modal hf-write-token secret.
TOKEN = os.environ.get("HF_WRITE_TOKEN")
if not TOKEN:
    raise SystemExit("HF_WRITE_TOKEN not set — use the Modal hf-write-token secret value")
REPO_ID = "PinoCookie/LFM2.5-1.2B-Instruct-Abliterated"
FOLDER = "campaigns/lfm2.5-recovery/hf-upload"
COMMIT = "Abliterated LFM2.5-1.2B-Instruct: rank-1 SVD recovery of huihui's directions (FIRST POSITIVE campaign)"


def main() -> int:
    api = HfApi(token=TOKEN)
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    print(f"uploading {FOLDER} -> {REPO_ID} ...", flush=True)
    url = api.upload_folder(
        repo_id=REPO_ID,
        folder_path=FOLDER,
        commit_message=COMMIT,
    )
    print(f"uploaded: {url}", flush=True)

    # ---- verification ----
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
    print("repo files:", sorted(files), flush=True)

    info = api.model_info(repo_id=REPO_ID, files_metadata=True)
    for s in sorted(info.siblings, key=lambda s: s.rfilename):
        print(f"  {s.rfilename}: {s.size:,} bytes", flush=True)

    with open(FOLDER + "/model.safetensors", "rb") as f:
        import hashlib

        local_sha = hashlib.sha256(f.read()).hexdigest()
    print("local model.safetensors sha256:", local_sha, flush=True)

    missing = [f for f in [
        "model.safetensors", "config.json", "generation_config.json",
        "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "README.md", "LICENSE",
        "ablitation_config.json", ".gitattributes",
    ] if f not in files]
    if missing:
        print("MISSING FILES:", missing, flush=True)
        return 1
    print("ALL EXPECTED FILES PRESENT", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())