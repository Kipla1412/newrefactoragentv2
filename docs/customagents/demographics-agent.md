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

Adapted from LangCare's `langcare-mcp-fhir` project
(`skills/core/patient-data-summary/patient-demographics-summary/SKILL.md`). The
steps now match the tools this server actually exposes.

| Step | Detail |
|---|---|
| Patient | `fhir__get_patient_by_id` (or `fhir__get_patient_core_by_id`): identifiers, official/maiden name, `birthDate` + age, gender, US Core race/ethnicity extensions, addresses by `use`, telecom, preferred language, marital status, deceased, managing organization |
| Per-field fallback | `fhir__list_patient_{names,identifiers,telecom,addresses,communications}` when the patient resource comes back sparse |
| Contacts | `fhir__list_patient_contacts` — embedded `Patient.contact` entries; relationship from `relationship.coding.code`, `period` used to confirm it is current |
| Nominated PCP | `fhir__list_patient_general_practitioners` |
| Coverage | **not available on this server** — flagged, never guessed |
| Advance directives | **not available on this server** (no Consent/DocumentReference tool) — flagged, never guessed |
| Allergy status | **not available** (no AllergyIntolerance tool) — reported as unknown |
| Missing data | a mandatory **MISSING DATA FLAGS** section |
| Output | `PATIENT DEMOGRAPHICS / CONTACT INFORMATION / EMERGENCY CONTACTS / INSURANCE COVERAGE / ADVANCE DIRECTIVES / MISSING DATA FLAGS` |

Two traps the skill warns about:

- `fhir__list_patient_links` is **not** the contacts tool — it returns
  `Patient.link` (merged-record / replaced-by), not next-of-kin.
- `fhir__list_patient_contacts` returns embedded `Patient.contact`, not
  separate RelatedPerson resources.

> **Where the stricter rules live.** The upstream skill has no "never invent"
> rule and no explicit prohibition on assuming an emergency contact or active
> coverage. Those live in the agent's role prompt (`DEMOGRAPHICS_PROMPT`), which
> is prepended to the skill in the same system message — see
> `customagents/demographicsagent/demographicsprompt.py`.

## Relationship to other agents

- **`ehr`** — same FHIR-over-MCP machinery and the same class shape.
  `demographics` is narrower: one patient, one resource set, one workflow.
- Both omit `tools:` for the same reason (runtime MCP discovery).

## Testing

`test/test_demographics_agent.py` (25 tests):

| Area | Coverage |
|---|---|
| Skill loading | discovery, metadata, frontmatter validity, all three resource types, every agreed field |
| Instruction contract | asserts the shipped LangCare skill's contents: `fhir_read`/`fhir_search`, the **MISSING DATA FLAGS** list, relationship-based emergency contacts, active-only Coverage with plan details, advance directives, the `Patient.contact` fallback, the allergy-review distinction, and the id-format note |
| Registration | `AgentType.DEMOGRAPHICS`, registered under its own name in `DEFAULT_AGENT_CLASSES`; prompt registered; route mounted with `message`/`session_id`/`skill` |
| Configuration | shipped config sets prompt + skill, no `tools`, and no `agent_class` |
| Construction | builds through `AgentFactory` as a `DemographicsAgent` with prompt and skill applied |
| ContextManager | skill workflow + role prompt both in the system message; scoped and cleared; no-skill path; unknown skill reported without an LLM call |
| FHIR tool execution | a `fhir__*` tool runs with the model's arguments, its output is fed back on the next turn, and a summary is produced |
| Failure handling | a failing FHIR tool yields partial results rather than an abort; an unknown tool does not break the turn; MCP-style mutating calls are not blocked |
| Request isolation | the route clears the context **before** the agent runs, so a reused `session_id` cannot leak the previous patient's data |
| Isolation | the skill does not leak into a follow-up request |
