# DemographicsAgent (patient demographics summary)

Builds a structured demographics summary for **one** patient from FHIR
`Patient`, `RelatedPerson`, and `Coverage` resources, with an explicit
"missing or unavailable" section. It never infers a value and never labels a
contact an emergency contact or a policy active without supporting data.

Built the same way as the EHR and consult agents: a concrete class with its own
prompt, plus the `patient-demographics-summary` skill for the workflow.

## Files

| File | Role |
|---|---|
| `customagents/demographicsagent/demographicsagent.py` | `DemographicsAgent` — the agent class |
| `customagents/demographicsagent/config.yaml` | model, prompt, skill (no `tools`) |
| `customagents/demographicsagent/demographicsprompt.py` | `DEMOGRAPHICS_PROMPT` — minimal role statement |
| `skills/library/patient-demographics-summary/SKILL.md` | the retrieval workflow and output format |
| `api/routers/demographics.py` | `POST /api/agent/demographics` route |

## Configuration

```yaml
name: demographics
model: clinical_model
prompt: demographics
skill: patient-demographics-summary
# tools: omitted — see below
```

`name` is the identity used by `agents.yaml` and the `AgentType.DEMOGRAPHICS`
enum. The class is registered under that name in `DEFAULT_AGENT_CLASSES`, the
same as every other agent.

### Why `tools:` is omitted

The FHIR tools are **MCP tools registered at runtime**, named
`<server>__<tool>` (e.g. `fhir__get_patient`), so they cannot be listed
statically. `allowed_tool_names` is an exact-match allowlist, so naming any tool
would filter the FHIR tools *out* and the agent could retrieve nothing.

Leaving the agent unrestricted is bounded in this deployment:

- the default registry contains no shell or filesystem tools — only `memory`,
  `wiki_search`, and `neo4j_query`;
- the only configured MCP server is `fhir`, created from `FHIR_BASE_URL`;
- the caller's token is carried into the MCP connection
  (`session.auth_token`), so the FHIR server enforces access.

Note that `MCPTool.is_mutating()` returns `True` for every call, and
`ApprovalManager.check_approval` auto-approves when there is no shell command
and no out-of-tree path. MCP calls are therefore **not** gated by a human
confirmation today — true for the EHR agent too, and worth knowing if you add a
mutating MCP server.

## Runtime flow — `POST /api/agent/demographics`

1. The route takes the session from `app.state.session_manager`, sets
   `session.auth_token` from the request, and re-initializes the MCP connection
   with it (same pattern as `/api/agent/mcpagent`).
2. **The conversation is cleared before the agent runs.** A summary covers one
   patient, so a reused `session_id` must not carry the previous patient's
   details into this turn — a correctness *and* a privacy problem. The trade-off
   is deliberate: this endpoint has no follow-up conversation, every call is
   independent.
3. `AgentFactory.create(AgentType.DEMOGRAPHICS, …)` resolves `DemographicsAgent`
   and applies its config: model, prompt, and default skill.
4. `DemographicsAgent.run(message)` builds the system prompt, activates the skill
   through `Agent.use_skill` for this request only, and runs the shared
   `_agentic_loop()`.
5. The loop calls the FHIR MCP tools, feeds results back, and streams
   `AgentEvent`s. The route renders them as NDJSON
   (`tool_start`, `tool_result`, `text_delta`, `text_complete`, `error`).
6. The skill is deactivated when the request ends.

Requires `consultagent:chat`.

## What the skill enforces

| Rule | Detail |
|---|---|
| Grounding | every populated field must trace to retrieved data; no inventing or deriving |
| Identity | stop and report if the returned resource is not the requested identifier |
| Emergency contact | never labelled unless the data explicitly identifies one |
| Coverage | never described as active unless the resource says so; an open `period` is not evidence |
| Missing data | a mandatory **Missing or Unavailable** section separating *not present*, *not retrieved*, and *retrieval failed* |
| Failures | invalid identifier → report and stop (never fall back to a name search); tool failure → report it and still produce a partial summary |
| Privacy | minimum necessary; report authorization refusal rather than working around it |

## Relationship to other agents

- **`ehr`** — same FHIR-over-MCP machinery and the same class shape.
  `demographics` is narrower: one patient, one resource set, one workflow.
- Both omit `tools:` for the same reason (runtime MCP discovery).

## Testing

`test/test_demographics_agent.py` (25 tests):

| Area | Coverage |
|---|---|
| Skill loading | discovery, metadata, frontmatter validity, all three resource types, every agreed field |
| Instruction contract | no invention/inference, no assumed emergency contact, no assumed active coverage, identity verification, the three-way missing-data distinction, retrieval-before-"no data", no name-search fallback, privacy section |
| Registration | `AgentType.DEMOGRAPHICS`, registered under its own name in `DEFAULT_AGENT_CLASSES`; prompt registered; route mounted with `message`/`session_id`/`skill` |
| Configuration | shipped config sets prompt + skill, no `tools`, and no `agent_class` |
| Construction | builds through `AgentFactory` as a `DemographicsAgent` with prompt and skill applied |
| ContextManager | skill workflow + role prompt both in the system message; scoped and cleared; no-skill path; unknown skill reported without an LLM call |
| FHIR tool execution | a `fhir__*` tool runs with the model's arguments, its output is fed back on the next turn, and a summary is produced |
| Failure handling | a failing FHIR tool yields partial results rather than an abort; an unknown tool does not break the turn; MCP-style mutating calls are not blocked |
| Request isolation | the route clears the context **before** the agent runs, so a reused `session_id` cannot leak the previous patient's data |
| Isolation | the skill does not leak into a follow-up request |
