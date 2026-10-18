# EHRAgent (FHIR tools over MCP)

`EHRAgent` answers requests by calling **FHIR tools exposed over MCP**. It is
the `AgentType.EHR` agent.

## Files

| File | Role |
|---|---|
| `customagents/ehragent/ehragent.py` | `EHRAgent` class |
| `customagents/ehragent/ehrprompt.py` | `EHR_AGENT_PROMPT` |
| `customagents/ehragent/config.yaml` | per-agent YAML config |
| `api/routers/ehr.py` | `POST /api/agent/mcpagent` route |

## Configuration

```yaml
name: ehr
model: clinical_model
prompt: ehr
# No tools key: the FHIR MCP tools are registered dynamically at runtime,
# so this agent's tool set is left unrestricted.
```

This is one of the two agents that deliberately **omits** `tools:`. The FHIR
tools come from the MCP Manager (see `tools/mcp/`), which registers them after
connecting to `FHIR_BASE_URL`. They do not exist at YAML-validation time, so
listing them statically would fail the loader.

> If the MCP server is unreachable the logs show
> `[MCP CONNECTION ERROR] Client failed to connect: All connection attempts
> failed`. That is benign for non-EHR requests, but the EHR agent has no tools
> until the connection succeeds.

## Runtime flow — `POST /api/agent/mcpagent` (streaming)

`EHRAgent.run(message, skill=None)` uses the minimal loop shape also used by
`MedicalAgent`:

1. Initializes the session; sets `agent_name = "EHRAgent"`.
2. `get_system_prompt(config, role_prompt=self.system_prompt)` — note it does
   **not** pass `user_memory` (unlike consult/intake).
3. Adds the user message (if any) and runs `_agentic_loop()`, **forwarding
   events unchanged** — no per-event `agent` tagging, unlike consult.
4. Wraps the run in `start_mlflow_run("EHR Agent Request")` /
   `end_mlflow_run()`; exceptions become an `agent_error` event.

**Optional skill.** `skill` names a skill activated through the reusable
`Agent.use_skill()` seam for this request only — see
**[../skills.md](../skills.md)** and the bundled `ehr-fhir-retrieval` skill.
The route passes it from the request body (`MCPRequest.skill`).

## Tool flow

```
MCP Manager ──connects──> FHIR_BASE_URL
     └─ registers fhir_* tools into the ToolRegistry
AgentFactory ── tools key omitted ──> EHRAgent unrestricted
_agentic_loop ── tool_choice ──> calls fhir_* tools
```

## Testing

- `test/test_agent_gating.py` — `/api/agent/mcpagent` presence follows the `ehr`
  enabled flag.
- `test/test_registries.py` — `ehr` prompt resolves; a config with no `tools`
  key validates as "unrestricted".
