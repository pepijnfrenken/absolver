<!-- aiwg:managed v2026.7.12 bundled -->
# AIWG Behavior: build-monitor

Provider surface: Claude Code rules
Provider: claude
Native source: ../.hermes/node/lib/node_modules/aiwg/agentic/code/behaviors/build-monitor/BEHAVIOR.md

Track build health by monitoring build tool completions and running scheduled

This provider does not expose OpenClaw-style native behavior directories. AIWG installs this generated behavior rule so the provider still receives the behavior contract instead of silently skipping it.

## Activation

Apply this behavior whenever the session, daemon, chat bridge, Mission Control loop, or provider runtime sees a matching trigger from the source behavior metadata. If a trigger cannot be observed natively, treat this file as provider-context guidance for the closest available rule, instruction, hook, or AGENTS-style surface.

## Source Behavior

```markdown
---
name: build-monitor
version: 1.0.0
description: Track build health by monitoring build tool completions and running scheduled
  build checks.
platforms:
- claude-code
- opencode
- warp
- openclaw
- codex
metadata:
  triggers:
  - monitor build
  - check build health
  - build status
  scope: daemon
inputs:
- name: command
  type: string
  required: false
  description: Build command to run
  default: npm run build
hooks:
  on_tool_complete:
  - tool: build
    action: run_script
    script: scripts/post-build-check.sh
  - tool: tsc
    action: run_script
    script: scripts/post-build-check.sh
  on_schedule:
  - cron: 0 */4 * * *
    action: run_script
    script: scripts/scheduled-build.sh
scripts:
  main: scripts/main.sh
  post-build-check: scripts/post-build-check.sh
  scheduled-build: scripts/scheduled-build.sh
manifest:
  category: build
  requires:
    bins:
    - node
  outputs:
  - type: report
    path: .aiwg/reports/build/
  composable_with:
  - test-watcher
---

# Build Monitor

Track build health by reacting to build tool completions and running periodic build verification.

## When Triggered via NLP

Run the build command and report success/failure with diagnostics. Output build metrics (duration, warnings, errors) to `.aiwg/reports/build/`.

## When Triggered via Hooks

### on_tool_complete (build, tsc)

After a build or TypeScript compilation completes, validate the output:
- Check for new warnings introduced
- Verify output artifact sizes haven't grown unexpectedly
- Log build duration for trend tracking

### on_schedule (every 4 hours)

Run a clean build from scratch to detect environmental drift or dependency issues that incremental builds miss.
```
