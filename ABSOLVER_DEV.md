# Absolver Dev Loop

You are working on the Absolver abliteration pipeline at `/home/pino/absolver/`. This is a LangGraph pipeline for automated safety removal in open-source LLMs.

## Current State
- All stages exist: summon → probe → distill → excise → verify → judge → reflexion → rebirth
- Graph builds and imports cleanly (9 nodes)
- 33 P0/P1 bugs from GPT-5.6-sol audit are fixed
- 3 test files exist
- `uv run python3 -c "from graph import build_abliteration_graph; g = build_abliteration_graph(); print(g.nodes.keys())"` works

## Your Job

1. **Check the codebase** — read all the key files (graph.py, each stage, config, tests)
2. **Run the tests** — `uv run pytest tests/ -v` to see what passes/fails
3. **Fix test failures** — any failing tests should be fixed
4. **Run a dry pipeline test** — try running the pipeline with a minimal config (no GPU needed, use CPU mode)
5. **Clean up**:
   - Remove dead code, unused imports
   - Fix any obvious bugs or inconsistencies
   - Make sure type hints are consistent
   - Check config schema matches what the stages expect
6. **Document what you find** — create a STATUS.md with current state, what works, what doesn't, what needs attention
7. **Loop** — after fixes, re-run tests, repeat until clean

## Environment
- Use `uv run python3` to run things (not bare `python3`)
- Dependencies are installed via `uv sync` (already done)
- No GPU needed — the pipeline should work in CPU-only mode for basic verification
- If you need to add dependencies, add them to pyproject.toml

## Rules
- Never use `hf repo cp` or similar
- Work incrementally — fix one thing, test it, move on
- Keep a STATUS.md updated
- If something blocks you (missing dep, complex refactor), note it and move to the next task
