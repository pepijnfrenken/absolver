"""Spawn the LoRA-repair Modal function detached via the SDK.

`modal run --detach` still lets a terminated client cancel its in-flight
input (measured: bash timeout -> client SIGTERM -> task killed). A
Function.spawn() call returns immediately and the function runs
server-side with no client-held input to cancel. Monitor via
`modal app logs <app_id>` and the absolver-phase2 volume.

Usage:
    python3 campaigns/lfm2.5-2.6b/spawn_repair.py \
        --ranks 16,32,64 --lrs 2e-5,5e-5,1e-4 --epochs 3 [--smoke] [--save-weights]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CAMP = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ranks", default="16,32,64")
    p.add_argument("--lrs", default="2e-5,5e-5,1e-4")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--save-weights", action="store_true")
    args = p.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lora_repair_modal", CAMP / "lora-repair-modal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # noqa: BLE001

    with mod.app.run(detach=True):
        call = mod.run_lora_repair.spawn(
            ranks_str=args.ranks, lrs_str=args.lrs, epochs=args.epochs,
            save_weights=args.save_weights, smoke=args.smoke)
        call_id = call.object_id
        app_id = call.app_id if hasattr(call, "app_id") else getattr(call, "app_id", None)
    print(f"spawned call {call_id}")
    print(f"app: {app_id}")
    print("monitor: modal app logs " + (app_id or ""))


if __name__ == "__main__":
    main()
