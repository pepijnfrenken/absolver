#!/usr/bin/env python3
"""Clean stale Python bytecode and ensure _project_2d is patched.

Run this from the repo root after pulling patched source files. It:
1. Recursively deletes every __pycache__ directory under ~/absolver/
2. Recursively deletes every .pyc file under ~/absolver/
3. Verifies _project_2d in excise.py has no try/except; replaces it if it does.
"""

from __future__ import annotations

import os
import shutil

ROOT = os.path.expanduser("~/absolver")
EXCISE = os.path.join(ROOT, "excise.py")

# Current body of _project_2d that still contains the try/except guard.
OLD_BODY = """    d = d.to(dtype=weight.dtype)
    try:
        weight.sub_(alpha * torch.einsum("i,j->ij", d, d @ weight))
    except RuntimeError as exc:
        raise RuntimeError(
            "EXCISE: in-place weight modification failed — the model appears "
            "to be quantized (4-bit / 8-bit). Refusal-direction projection "
            "requires dequantized (fp16/bf16/fp32) weights. Re-load the model "
            "without `quantize` before running the pipeline."
        ) from exc
"""

# Desired 4-line body: flatten, dtype cast, normalize, in-place projection.
NEW_BODY = """    d = d.flatten()
    d = d.to(dtype=weight.dtype)
    d = d / d.norm()
    weight.sub_(alpha * torch.einsum("i,j->ij", d, d @ weight))
"""


def wipe_pycache(root: str) -> None:
    """Recursively remove __pycache__ directories and .pyc files."""
    removed_dirs = 0
    removed_files = 0

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Remove __pycache__ directories in-place so os.walk won't descend into them.
        for dirname in list(dirnames):
            if dirname == "__pycache__":
                full_path = os.path.join(dirpath, dirname)
                print(f"Removing directory: {full_path}")
                shutil.rmtree(full_path)
                removed_dirs += 1
                dirnames.remove(dirname)

        # Remove loose .pyc files.
        for filename in filenames:
            if filename.endswith(".pyc"):
                full_path = os.path.join(dirpath, filename)
                print(f"Removing file: {full_path}")
                os.remove(full_path)
                removed_files += 1

    print(
        f"Removed {removed_dirs} __pycache__ directory(s) and "
        f"{removed_files} .pyc file(s)."
    )


def patch_project_2d(path: str) -> None:
    """Replace _project_2d body with the 4-line version if it still has try/except."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    if "def _project_2d" not in text:
        print("WARNING: _project_2d not found in excise.py")
        return

    # Isolate the _project_2d function region (up to the next top-level def/class).
    func_start = text.index("def _project_2d")
    candidates = [
        loc
        for loc in (
            text.find("\ndef ", func_start + 1),
            text.find("\nclass ", func_start + 1),
        )
        if loc != -1
    ]
    next_top_level = min(candidates) if candidates else -1
    func_region = text[func_start:next_top_level] if next_top_level != -1 else text[func_start:]

    if "try:" not in func_region and "except RuntimeError" not in func_region:
        print("_project_2d already has no try/except; no change.")
        return

    if OLD_BODY not in text:
        print(
            "WARNING: _project_2d has try/except but the old body text did not match "
            "exactly; manual review needed."
        )
        return

    text = text.replace(OLD_BODY, NEW_BODY, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Patched _project_2d to 4-line version (flatten, to(), normalize, einsum).")


if __name__ == "__main__":
    wipe_pycache(ROOT)
    print()
    patch_project_2d(EXCISE)
