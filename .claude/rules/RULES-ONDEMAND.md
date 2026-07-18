<!-- aiwg:managed vunknown bundled -->
# On-Demand Rules (not inlined at startup)

These MEDIUM/LOW-enforcement rules are not loaded into every session, to keep
the standard-context startup budget small. They still apply when relevant —
fetch any rule body on demand:

```bash
aiwg show rule <name>
```

- `auto-reply-chains` — `aiwg show rule auto-reply-chains`
- `best-output-selection` — `aiwg show rule best-output-selection`
- `config-in-environment` — `aiwg show rule config-in-environment`
- `conversable-agent-interface` — `aiwg show rule conversable-agent-interface`
- `criticality-panel-sizing` — `aiwg show rule criticality-panel-sizing`
- `dev-idempotent-builds` — `aiwg show rule dev-idempotent-builds`
- `disposable-processes` — `aiwg show rule disposable-processes`
- `few-shot-examples` — `aiwg show rule few-shot-examples`
- `hitl-patterns` — `aiwg show rule hitl-patterns`
- `human-gate-display` — `aiwg show rule human-gate-display`
- `index-generation` — `aiwg show rule index-generation`
- `it-asset-authority` — `aiwg show rule it-asset-authority`
- `it-service-health` — `aiwg show rule it-service-health`
- `logs-as-event-streams` — `aiwg show rule logs-as-event-streams`
- `mention-wiring` — `aiwg show rule mention-wiring`
- `no-binary-blobs` — `aiwg show rule no-binary-blobs`
- `ops-issue-tracking` — `aiwg show rule ops-issue-tracking`
- `progressive-disclosure` — `aiwg show rule progressive-disclosure`
- `qualified-references` — `aiwg show rule qualified-references`
- `reasoning-sections` — `aiwg show rule reasoning-sections`
- `reproducibility` — `aiwg show rule reproducibility`
- `sdlc-orchestration` — `aiwg show rule sdlc-orchestration`
- `sec-access-audit-frequency` — `aiwg show rule sec-access-audit-frequency`
- `self-maintenance` — `aiwg show rule self-maintenance`
- `stateless-processes` — `aiwg show rule stateless-processes`
- `sys-immutable-base` — `aiwg show rule sys-immutable-base`
- `thought-protocol` — `aiwg show rule thought-protocol`
