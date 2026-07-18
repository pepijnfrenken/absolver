# Absolver — Modal & Molab Connector Plan

## Files to Create

### 1. `connectors/__init__.py`
Exports both connectors. `run_pipeline_with_modal(config_path)`, `run_pipeline_with_molab(config_path)`.

### 2. `connectors/modal_runner.py`
A Modal app that:
- Builds a Modal image with py311, torch, transformers, langgraph, pydantic, pyyaml
- Mounts the local absolver project directory into the container
- Creates a `@modal.function(gpu=GPU_TYPE, timeout=7200)` entrypoint  
- Loads the config YAML, builds the LangGraph graph, invokes the pipeline
- Streams logs back, saves results to Modal Volumes or HF Hub
- Takes GPU type from config or CLI arg (L4, A10G, A100, H100)
- Handles HF_TOKEN for model downloads + uploads

Key config fields for Modal:
```yaml
platform: modal
modal_gpu: L4          # L4 | A10G | A100 | H100
modal_timeout: 7200
modal_volumes:         # persistent storage
  - /cache
```

### 3. `connectors/molab_runner.py`
A Python script that:
- Takes a Molab endpoint URL + API token (from env or config)
- Connects to the Molab session via the Marimo kernel API
- Uploads the full absolver project as a ZIP/base64
- Installs deps on the remote kernel (torch, transformers, langgraph)
- Executes the pipeline step by step via kernel/execute
- Streams results back
- Supports `--url` and `--token` flags

Key config fields for Molab:
```yaml
platform: molab
molab_url: https://sb-xxxxx.sb.molab.run
molab_token: "xxx"
```

### 4. Update `run.py`
Add `--platform` flag:
```bash
python run.py models/ornith-9b.yaml --platform modal     # runs on Modal
python run.py models/ornith-9b.yaml --platform molab      # runs on Molab
python run.py models/ornith-9b.yaml                       # runs locally (default)
```

## Modifications

### `config.py` — Add platform fields
```python
platform: str = "local"           # local | modal | molab
modal_gpu: str = "L4"
modal_timeout: int = 7200
modal_sync: bool = True           # sync results back to local
molab_url: str | None = None
molab_token: str | None = None
```

### `main.py` — Add `run_pipeline_modal()` and `run_pipeline_molab()`
```python
def run_pipeline(config_path, platform="local"):
    if platform == "modal":
        return run_pipeline_modal(config_path)
    elif platform == "molab":
        return run_pipeline_molab(config_path)
    else:
        return run_pipeline_local(config_path)
```

## Key Patterns (from existing code)

Modal pattern (run-abliteration-modal.py):
- `modal.Image.debian_slim().uv_pip_install(...)`
- `modal.App("absolver-{model_name}")`
- `@app.function(image=image, gpu=GPU_TYPE, timeout=7200, retries=0)`
- `modal.Mount.from_local_dir(...)` for local project files

Molab pattern (molab-connect.py):
- REST API at `{URL}/api/kernel/execute`
- Bearer token auth
- SSE streaming response (data: {...})
- Code execution via POST with `{"code": "..."}`
- `Marimo-Session-Id` header from `/api/sessions`
