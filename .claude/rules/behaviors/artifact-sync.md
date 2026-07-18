<!-- aiwg:managed v2026.7.12 bundled -->
# AIWG Behavior: artifact-sync

Provider surface: Claude Code rules
Provider: claude
Native source: ../.hermes/node/lib/node_modules/aiwg/agentic/code/frameworks/sdlc-complete/behaviors/artifact-sync/BEHAVIOR.md

Keep the SDLC artifact index current by reacting to changes in .aiwg/

This provider does not expose OpenClaw-style native behavior directories. AIWG installs this generated behavior rule so the provider still receives the behavior contract instead of silently skipping it.

## Activation

Apply this behavior whenever the session, daemon, chat bridge, Mission Control loop, or provider runtime sees a matching trigger from the source behavior metadata. If a trigger cannot be observed natively, treat this file as provider-context guidance for the closest available rule, instruction, hook, or AGENTS-style surface.

## Source Behavior

```markdown
---
name: artifact-sync
version: 1.0.0
description: Keep the SDLC artifact index current by reacting to changes in .aiwg/
  directories.
platforms:
- claude-code
- opencode
- warp
- openclaw
- codex
metadata:
  triggers:
  - sync artifacts
  - rebuild artifact index
  - update artifact index
  scope: daemon
inputs:
- name: force
  type: boolean
  required: false
  default: false
  description: Force a full index rebuild even if no changes detected
hooks:
  on_file_write:
  - filter: .aiwg/**/*.md
    action: run_script
    script: scripts/incremental-sync.sh
scripts:
  main: scripts/main.sh
  incremental-sync: scripts/incremental-sync.sh
manifest:
  category: sdlc
  requires:
    bins:
    - node
  outputs:
  - type: index
    path: .aiwg/reports/
---

# Artifact Sync

Keep the SDLC artifact index current by reacting to changes in `.aiwg/` directories.

## When Triggered via NLP

Run a full artifact index rebuild using `aiwg index build`. Report the number of artifacts indexed and any orphaned references found.

## When Triggered via Hooks

### on_file_write (.aiwg/**/*.md)

When any artifact in the `.aiwg/` directory tree changes:
- Update the artifact index incrementally
- Validate that @-mentions in the changed artifact resolve correctly
- Log the change for the artifact changelog
```
