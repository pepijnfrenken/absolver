<!-- aiwg:managed v2026.7.12 bundled -->
# AIWG Behavior: quality-gate-watcher

Provider surface: Claude Code rules
Provider: claude
Native source: ../.hermes/node/lib/node_modules/aiwg/agentic/code/frameworks/sdlc-complete/behaviors/quality-gate-watcher/BEHAVIOR.md

Enforce SDLC quality gate criteria on commits and pull requests.

This provider does not expose OpenClaw-style native behavior directories. AIWG installs this generated behavior rule so the provider still receives the behavior contract instead of silently skipping it.

## Activation

Apply this behavior whenever the session, daemon, chat bridge, Mission Control loop, or provider runtime sees a matching trigger from the source behavior metadata. If a trigger cannot be observed natively, treat this file as provider-context guidance for the closest available rule, instruction, hook, or AGENTS-style surface.

## Source Behavior

```markdown
---
name: quality-gate-watcher
version: 1.0.0
description: Enforce SDLC quality gate criteria on commits and pull requests.
platforms:
- claude-code
- opencode
- warp
- openclaw
- codex
metadata:
  triggers:
  - watch quality gates
  - check gate criteria
  - enforce quality
  scope: daemon
inputs:
- name: phase
  type: enum
  values:
  - inception
  - elaboration
  - construction
  - transition
  required: false
  description: SDLC phase to validate gates for (auto-detected if omitted)
- name: strict
  type: boolean
  required: false
  default: false
  description: Fail on any unmet gate criterion (vs warn)
hooks:
  on_commit:
  - action: run_script
    script: scripts/check-commit-gates.sh
  on_pr_open:
  - action: run_script
    script: scripts/check-pr-gates.sh
scripts:
  main: scripts/main.sh
  check-commit-gates: scripts/check-commit-gates.sh
  check-pr-gates: scripts/check-pr-gates.sh
manifest:
  category: quality
  requires:
    bins:
    - node
  outputs:
  - type: report
    path: .aiwg/reports/quality/
  composable_with:
  - security-sentinel
  - test-watcher
---

# Quality Gate Watcher

Enforce SDLC quality gate criteria automatically on commits and pull requests.

## When Triggered via NLP

Run a full gate evaluation for the current or specified SDLC phase. Report which criteria are met, which are unmet, and what actions are needed to pass the gate.

## When Triggered via Hooks

### on_commit

Lightweight gate check on each commit:
- Verify conventional commit format
- Check that modified files have associated tests
- Validate no `.aiwg/` artifacts are stale relative to code changes

### on_pr_open

Comprehensive gate evaluation when a PR is opened:
- Run full phase gate criteria check
- Verify all required artifacts exist and are current
- Check test coverage thresholds
- Validate security review status
- Post gate status as PR comment
```
