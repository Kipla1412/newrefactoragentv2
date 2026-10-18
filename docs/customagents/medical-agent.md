# MedicalAgent (medical knowledge + Neo4j/wiki)

`MedicalAgent` answers clinical knowledge questions using the **medical wiki**
and **Neo4j knowledge graph** tools. It is the `AgentType.MEDICAL` agent.

## Files

| File | Role |
|---|---|
| `customagents/medicalagent/medicalagent.py` | `MedicalAgent` class |
| `customagents/medicalagent/medicalprompt.py` | `MEDICAL_PROMPT` |
| `customagents/medicalagent/config.yaml` | per-agent YAML config |

## Configuration

```yaml
name: medical
model: clinical_model
prompt: medical
tools:
  - memory
  - wiki_search
  - neo4j_query
```

This is the only agent configured with `wiki_search` and `neo4j_query`. Those
two tools live in `tools/builtin/wiki.py` and `tools/builtin/neo4j.py` and are
resolved through the **existing ToolRegistry** — the config only selects them.

## Runtime flow

> **No HTTP route.** `medical` does not appear in `ROUTER_AGENT_REQUIREMENTS`
> in `api/main.py`, so no endpoint is mounted for it. It is constructed through
> `AgentFactory.create(AgentType.MEDICAL, config, session)` and driven
> programmatically (scripts, tests, or a future route). Enabling/disabling it
> still affects bootstrap and factory construction.

`MedicalAgent.run(message)` uses the minimal loop shape shared with `EHRAgent`:

1. Initializes the session; sets `agent_name = "MedicalAgent"`.
2. `get_system_prompt(config, role_prompt=self.system_prompt)` — no
   `user_memory` argument.
3. Adds the user message (if any) and forwards the `_agentic_loop()` events
   unchanged.
4. `start_mlflow_run("Medical Agent Query")` / `end_mlflow_run()`; errors
   become `agent_error`.

Because its tools are graph/wiki lookups, the base agent invokes
`wiki_search` / `neo4j_query` mid-loop and grounds the final answer in their
results.

## Testing

- `test/test_agent_gating.py` — `medical` is absent from the OpenAPI paths
  (no route), but disabling it must not break bootstrap.
- `test/test_registries.py` — `medical` prompt resolves and its tool list
  validates against the ToolRegistry.
