# ConsultingAgent (interactive medical consultation)

`ConsultingAgent` is the conversational front door of the framework. It is the
`AgentType.CONSULT` agent and the first stage of the
`/consultation-workflow` composite endpoint.

## Files

| File | Role |
|---|---|
| `customagents/consultagent/consultagent.py` | `ConsultingAgent` class |
| `customagents/consultagent/consultprompt.py` | `CONSULT_PROMPT` |
| `customagents/consultagent/config.yaml` | per-agent YAML config |
| `api/routers/consult.py` | `POST /agent/consult` route |

## Configuration

```yaml
name: consult
model: clinical_model    # openai/gpt-4.1 via ModelRegistry
prompt: consult          # logical name -> CONSULT_PROMPT
tools:
  - memory              # the ONLY registered tool available to consult
```

Enabled in `agents.yaml` via `consult: { enabled: true }`.

## Runtime flow — `POST /agent/consult` (streaming)

`api/routers/consult.py` returns an **NDJSON / SSE** `StreamingResponse`:

```python
async def consult_api(request, data: ChatRequest):
    session = await request.app.state.session_manager.get_session(user_id, config)
    agent   = AgentFactory.create(AgentType.CONSULT, config, session)
    async def event_stream():
        async for event in agent.run(data.message):
            yield (json.dumps(event.model_dump()) + "\n")   # one JSON object / line
    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
```

`ChatRequest`: `{ message: str }`  →  `ChatResponse`:
`{ type, message, agent:"consult" }`, or `ErrorResponse` `{ error }`.

## `ConsultingAgent.run(message)`

1. Initializes the session context manager if needed.
2. Sets `session.agent_name = "DoctorAI"` for MLflow tagging.
3. Builds the full system prompt from `CONSULT_PROMPT` + registry/system
   prompt + user memory, then installs it on the context manager.
4. **Greets once per session** — on a new session (no `greeted` flag in
   `session.metadata`) it emits a `text_complete` greeting
   (`"Hi! I'm your medical assistant…"`) and returns immediately if
   the message is empty, so the first real turn is a clean handshake.
5. On subsequent turns it adds the user message and runs the shared
   `_agentic_loop()`, forwarding each `AgentEvent` (with `agent` set to
   `consult`), accumulating `TEXT_DELTA` chunks.
6. **End-of-consultation detection** — if the reply contains any
   `END_PATTERNS` string
   (`"consultation is complete"`, `"see a doctor"`, `"follow-up
   instructions"`, `"take care"`, `"stay healthy"`, …), the loop emits a
   final `status_end` event to signal the session can be archived /
   handed to the SOAP agent.
7. `start_mlflow_run(message)` / `end_mlflow_run()` wrap the call for
   observability when `MLFLOW_ENABLED=true`; any error becomes an
   `agent_error` event rather than crashing the stream.

### Streaming contract seen by clients

Each SSE line is a JSON `AgentEvent`:

```json
{"type": "text_delta",  "data": {"content": "..."}, "agent": "consult"}
{"type": "text_complete", "data": {"content": "Can you tell me about…"}, "agent": "consult"}
{"type": "tool_call",   "data": {...}, "agent": "consult"}
{"type": "status_end",  "data": {},    "agent": "consult"}
```

(The schema examples in `consult.py` also show the flattened
`ChatResponse` shape `{type, message, agent}`. The wire format you actually
receive is the `AgentEvent` JSON above.)

## Relationship to other agents

- **Shares the user's session** with `SOAPAgent`, `ClinicalExtractionAgent`,
  etc. — `session_manager.get_session(user_id)` returns the same session, so
  history persists across `/consult` → `/assessment` → `/soap`.
- **Feeds `/consultation-workflow`** — that composite route only mounts when
  *all three* of `consult`, `assessment`, and `clinical_extraction` are
  `enabled: true`; otherwise it degrades to the individual endpoints.

## Testing

- `test/test_agent_gating.py` — `/agent/consult` is present iff
  `agents.consult.enabled == true`.
- `test/test_registries.py` — `consult` prompt registered → `CONSULT_PROMPT`.
- `test/test_llm_litellm.py` — model fallback chain behind consult
  (`clinical_model → gemini → ollama`).
