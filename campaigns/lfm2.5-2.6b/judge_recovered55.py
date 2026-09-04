"""Re-judge the recovered-Heretic 55 transcripts under the round-2 judge pair.

Round-3 follow-up: the inline qwen3.6-35b judge flagged 12/55 refusals on
the recovered model. The round-2 evidence shows qwen3.6-35b over-flags
safety-hedged compliance (original Heretic scored 1.8% under
deepseek-v4-flash). This scores the SAME 55 recovered responses under
BOTH round-2 judges for a same-yardstick comparison vs pristine and
original Heretic.

Reads:  /tmp/rec-gate.json  (recovered run gate record, has transcript)
Writes: judge-recovered-55.json (campaign dir)
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_round2 import MODELS, judge_pair  # noqa: E402

CAMP = Path(__file__).resolve().parent
OUT = CAMP / "judge-recovered-55.json"

gate = json.loads(Path("/tmp/rec-gate.json").read_text())
rows = gate["transcript"]
print(f"[judge-rec] {len(rows)} recovered responses to judge x {len(MODELS)} models")

results: dict = {}
if OUT.exists():
    results = json.loads(OUT.read_text())
    print(f"[judge-rec] resuming: {sum(len(v['verdicts']) for v in results.values())} verdicts exist")

done_keys = {
    (v.get("prompt"), v.get("response"), v.get("model"))
    for rec in results.values()
    for v in rec.get("verdicts", [])
}

new = []
for row in rows:
    for model in MODELS:
        key = (row.get("prompt", ""), row.get("response", ""), model)
        if key not in done_keys:
            new.append((model, row))

with ThreadPoolExecutor(max_workers=5) as pool:
    futures = {
        pool.submit(judge_pair, row["prompt"], row["response"], model): (model, row)
        for model, row in new
    }
    n_ok = 0
    for fut in as_completed(futures):
        model, row = futures[fut]
        try:
            v = fut.result()
        except Exception as exc:  # noqa: BLE001
            print(f"[judge-rec] FAIL {model}: {exc}", flush=True)
            continue
        n_ok += 1
        results.setdefault("recovered-her", {"verdicts": []})["verdicts"].append(
            {"model": model, "prompt": row.get("prompt", ""),
             "response": row.get("response", ""), **v})
        if n_ok % 20 == 0:
            OUT.write_text(json.dumps(results, indent=2))
            print(f"[judge-rec] {n_ok}/{len(new)} judged", flush=True)

OUT.write_text(json.dumps(results, indent=2))
print(f"[judge-rec] done; {n_ok} new verdicts -> {OUT}")
