# Absolver Status

## Current State (2026-07-27)

**Pipeline: operational end-to-end on CPU with tiny model.**

### Graph (9 nodes)
All stages wired correctly in `StateGraph(AbliterationState)`:

| Node   | Function         | Edges                             | Status |
|--------|------------------|-----------------------------------|--------|
| summon | summon_node      | START → summon → probe            | ✅     |
| probe  | probe_node       | probe → distill                   | ✅     |
| distill| distill_node     | distill → (excise\|probe\|reflexion) | ✅  |
| excise | excise_node      | excise → verify                   | ✅     |
| verify | verify_node      | verify → (judge\|rebirth\|excise\|reflexion) | ✅ |
| judge  | judge_node       | judge → (rebirth\|excise\|reflexion) | ✅  |
| reflexion| reflexion_node | reflexion → (probe\|distill\|excise\|rebirth) | ✅ |
| rebirth| rebirth_node     | rebirth → END                     | ✅     |

### What Works
- **All 33 tests pass** (`uv run pytest tests/ -v`): detector (9), excise+distill (12), experience (12)
- **Full dry-run pipeline** on `hf-internal-testing/tiny-random-LlamaForCausalLM` (CPU, 2 layers, 32 hidden) completes in ~13s
- **Pipeline output**: model saved to disk (`abliterated_*`), metadata written, refusal directions extracted for all layers
- **All 4 direction-extraction methods** (`diff_means`, `svd`, `leace`, `whitened_svd`) tested
- **Experience DB**: SQLite-backed with corruption recovery, summary stats, query-by-architecture
- **Architecture detection**: dense, MoE, diffusion text encoder
- **4 cleanup utility files** removed (`fix_cache_and_excise.py`)

### Changes Made This Session

1. **Package structure**: Added `tests/conftest.py` so `uv run pytest` works without `PYTHONPATH`
2. **Removed duplicate config fields**: `platform`, `modal_gpu`, `modal_timeout`, `molab_url`, `molab_token` were declared twice in `ModelConfig` (lines 62-71 and 183-192)
3. **Fixed `UnsupportedArchitecture` → `UnsupportedArchitectureError`** (ruff N818)
4. **Fixed `_safe_svd` `S[0].item` → `S[0].item()`** (missing `()` real bug, auto-fixed)
5. **Fixed `diffs.squeeze` → `diffs.squeeze()`** (missing `()` real bug, auto-fixed)
6. **Fixed `detector.py` docstring** — class body missing `"""` after rename, causing syntax error
7. **Type hint consistency**: All node functions now use `dict[str, Any]` (was mixed `dict` / `dict[str, Any]`)
8. **Added imports**: `Any` to `main.py` and `reflexion.py`
9. **Suppressed N806 for math variables** (U, S, Vt, X, Sigma, W — conventional linear algebra notation)
10. **Fixed unused imports**: `Optional` in `modal_runner.py`, `time`/`uuid` in `molab_runner.py`, sorted imports in all connectors
11. **Fixed unused variable**: `result` → `_` in `modal_runner.py`
12. **Removed `MemorySaver` checkpointer** from compiled graph — HF model objects aren't msgpack-serializable, crashes on any real pipeline run
13. **Added `accelerate`** to `pyproject.toml` dependencies (required by `device_map="auto"`)
14. **Removed** `fix_cache_and_excise.py` (obsolete one-time utility)
15. **ruff: clean** — zero diagnostics
16. **Import check: clean** — all 16 modules import without errors

### Known Issues / Attention Needed

| Issue | Severity | Notes |
|-------|----------|-------|
| `MemorySaver` removed from graph | **P1** | Model objects can't be checkpointed. If resume capability is needed, implement a split serializer (store model refs separately, or re-`summon` on resume). |
| `verify.py` `run_mmlu_mini` prints `f"Load of MMLU config failed: {exc}"` on every import | **P2** | `_MMLU_MINI` is a global with try/except at module level. Better to lazy-load or suppress the warning. |
| ~~`judge.py` requires OMP binary~~ | **FIXED** | `_call_omp` replaced with `_call_judge_api` — direct OpenAI-compatible call via `llm_api.py` (FreeInference default). Works in Modal containers, no OMP install needed. |
| ~~`reflexion.py` requires OMP binary~~ | **FIXED** | KB consultation now uses `llm_api.chat_completion` — direct API call, same as judge. |
| `summon.py` imports transformers inside function | **P3** | `from transformers import BitsAndBytesConfig` inside `summon_node` — could be top-level guarded import. |
| `config.py` has `model_arch: str = "auto"` but detector returns `"dense"\|"moe"\|"diffusion_encoder"` | **P3** | Auto mapping from string "auto" to `detect_architecture` output isn't defined in config — handled at runtime. |
| `judge.py` `_keyword_refusal_score` imported `REFUSAL_KEYWORDS` from `verify` — correct reuse | - | Verified: both modules use the same 26-keyword list. |
| 4 `.pyc` / `__pycache__` dirs cleaned | - | Rebuild on next import. |

### Test Coverage (33 tests)

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_detector.py` | 9 | Architecture detection (dense, MoE, diffusion, error cases) |
| `tests/test_excise.py` | 12 | Distill (6 methods/configs) + Excise (6 weight projection checks) |
| `tests/test_experience.py` | 12 | ExperienceDB (schema, query, upsert, JSON, corruption recovery) |

No tests exist for: summon, verify, judge, reflexion, rebirth, routing, graph assembly, prompts.

### Dry-Run Test

```python
graph = build_abliteration_graph()
cfg = load_config("models/tiny_test.yaml")  # hf-internal-testing/tiny-random-LlamaForCausalLM
result = graph.invoke({"config": cfg})
# refusal_rate: 0.0, quality_pass: True, output_path: abliterated_tiny-random-LlamaForCausalLM
```