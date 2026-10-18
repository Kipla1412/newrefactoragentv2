# Agents

Every agent in this framework is a class extending `agent.agent.Agent` plus a
`config.yaml` describing what it uses. `AgentFactory` is the only code that
knows the concrete classes; everything else refers to agents by their
**logical name** (`soap`, `consult`, `pdf_chat`, …).

For the loader, registries, `enabled` gating, and the SOLID collaborator
breakdown, see **[../customagents.md](../customagents.md)**.

## Shared anatomy

| Piece | Where |
|---|---|
| Agent class | `customagents/<folder>/<agent>.py` |
| Prompt | `customagents/<folder>/<prompt>.py` |
| Config | `customagents/<folder>/config.yaml` (or `*.config.yaml`) |
| Route | `api/routers/<router>.py`, mounted from `ROUTER_AGENT_REQUIREMENTS` |
| Enabled flag | `agents.yaml` → `agents.<name>.enabled` |

Shared lifecycle inside every agent's entry point:

```
session.initialize()                     # if not already done
session.agent_name = "<AgentName>"       # MLflow tagging
context_manager.clear()                  # report agents only
context_manager.set_system_prompt(...)   # via get_system_prompt(...)
context_manager.add_user_message(...)
session.start_mlflow_run(...)
    async for event in self._agentic_loop(): ...
session.end_mlflow_run()
```

Two response shapes:

- **Streaming** (`run(...)`) — yields `AgentEvent`s, served as NDJSON.
- **Buffered** (`generate(...)` / `recommend_questions(...)`) — drains the loop
  internally and returns parsed JSON.

## Route map

| Endpoint | Agent(s) required | Doc |
|---|---|---|
| `POST /api/agent/doctoragent` | `doc` | [doc-agent](doc-agent.md) |
| `POST /api/agent/consult` | `consult` **or** `intake` | [consult-agent](consult-agent.md) |
| `POST /api/agent/intake` | `consult` **or** `intake` | [intake-agent](intake-agent.md) |
| `POST /api/agent/soap` | `soap` | [soap-agent](soap-agent.md) |
| `POST /api/agent/assessment` | `assessment` | [assessment-agent](assessment-agent.md) |
| `POST /api/agent/clinical-extraction` | `clinical_extraction` | [clinical-extraction-agent](clinical-extraction-agent.md) |
| `POST /api/agent/consultation-workflow` | `soap` + `assessment` + `clinical_extraction` | [clinical-extraction-agent](clinical-extraction-agent.md) |
| `POST /api/agent/document` | `document_retrieval` | [document-retrieval-agent](document-retrieval-agent.md) |
| `POST /api/agent/pdf-chat` | `pdf_chat` | [pdf-chat-agent](pdf-chat-agent.md) |
| `POST /api/agent/mcpagent` | `ehr` | [ehr-agent](ehr-agent.md) |
| `/ws/...` (voice) | `voice_intake` **or** `voice_consult` | [voice-intake](voice-intake-agent.md), [voice-consult](voice-consult-agent.md) |
| `/ws/...` (diarize) | `diarize` | [diarize-agent](diarize-agent.md) |
| _(none)_ | `medical` | [medical-agent](medical-agent.md) |

## Index

### Report / documentation
- **[soap-agent.md](soap-agent.md)** — doctor-patient conversation → SOAP note
- **[assessment-agent.md](assessment-agent.md)** — transcript → assessment report
- **[clinical-extraction-agent.md](clinical-extraction-agent.md)** — SOAP/assessment → FHIR concepts

### Conversation
- **[consult-agent.md](consult-agent.md)** — streaming clinical consultation
- **[intake-agent.md](intake-agent.md)** — pre-visit intake (text)
- **[doc-agent.md](doc-agent.md)** — follow-up question recommendations

### Knowledge / retrieval
- **[medical-agent.md](medical-agent.md)** — wiki + Neo4j graph (no HTTP route)
- **[document-retrieval-agent.md](document-retrieval-agent.md)** — RAG over indexed documents
- **[ehr-agent.md](ehr-agent.md)** — FHIR tools over MCP
- **[pdf-chat-agent.md](pdf-chat-agent.md)** — chat with an uploaded PDF

### Voice
- **[voice-intake-agent.md](voice-intake-agent.md)** — spoken intake
- **[voice-consult-agent.md](voice-consult-agent.md)** — spoken consultation
- **[diarize-agent.md](diarize-agent.md)** — speaker-attributed transcription

Model selection, fallbacks, and routing for all of these are documented in
**[../litellm.md](../litellm.md)**. Reusable per-request instruction sets
(skills) and how an agent activates one are documented in
**[../skills.md](../skills.md)**.
