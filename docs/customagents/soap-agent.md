# SOAPAgent (clinical note generation)

`SOAPAgent` turns a doctor-patient conversation into a structured
**SOAP** note (Subjective, Objective, Assessment, Plan) ready for EHR
filing. It is the report agent for `AgentType.SOAP`.

## Files

| File | Role |
|---|---|
| `customagents/reportagent/soapagent.py` | `SOAPAgent` class + JSON parsing |
| `customagents/reportagent/soapprompt.py` | `SOAP_PROMPT` — the 470-line master prompt (user rewrite) |
| `customagents/reportagent/soap.config.yaml` | per-agent YAML config |
| `api/routers/soap.py` | `POST /agent/soap` route |
| `api/routers/schemas.py` | `ConversationRequest`, `SOAPResponse` (incl. `SOAPNote`) |
| `utils/soapparser.py` | (read-only reference) alternate parser |

## Configuration

`customagents/reportagent/soap.config.yaml`:

```yaml
name: soap
model: clinical_model   # openai/gpt-4.1 via ModelRegistry
prompt: soap           # logical name -> SOAP_PROMPT in PromptRegistry
tools:
  - memory             # the ONLY registered tool the SOAP agent can call
```

`model: clinical_model` and `prompt: soap` resolve through the
`ModelRegistry` and `PromptRegistry` at factory time (see
**docs/customagents.md**). The endpoint only mounts when `agents.soap`
has `enabled: true`.

## Runtime flow — `POST /api/agent/soap`

`api/routers/soap.py`:

```python
async def generate_soap_api(request, data: ConversationRequest):
    session = await request.app.state.session_manager.get_session(user_id, config)
    agent   = AgentFactory.create(AgentType.SOAP, config, session)
    result  = await agent.generate(data.conversation)
    return result
```

`ConversationRequest.conversation` is a `List[str]` — a list of
diarized/ordered utterances from the consultation transcript.

`SOAPAgent.generate()` then:

1. Initializes the session context manager if needed.
2. Sets the agent name to `SOAPScribeAgent` for MLflow tracking.
3. Clears the context and installs **only** the SOAP system prompt.
4. Joins the conversation list into a single transcript and sends one
   user message:
   ```
   "Please generate a structured SOAP note from the following conversation:\n{transcript}"
   ```
5. Runs the agentic loop, accumulating the streamed `TEXT_COMPLETE`
   payload.
6. Parses the LLM output as JSON via `_parse_soap_json` (strips markdown code
   fences, falls back to `{"error": "JSON_PARSING_FAILED", …}`
   on unparseable output).

## Response contract

`POST /agent/soap` returns a `SOAPResponse` — a **nested, permissive**
schema (`model_config = {"extra": "allow"}`, every field has a default)
ported from the canonical contract in `test/demo.py`:

```
SOAPResponse
├── document_type                 # "SOAP_NOTE"
├── document_version              # "1.0"
├── patient                       # { patient_id, display_name, … }
├── encounter                     # { encounter_id, documentation_status, … }
├── documentation_metadata        # source types, clinician review flag
├── soap_note
│   ├── subjective  { chief_complaint, history_present_illness, … }
│   ├── objective   { vital_signs, physical_exam, measurements, … }
│   ├── assessment  { problem_list, dx_considerations, … }
│   └── plan        { medications, orders, referrals, follow_up, … }
└── clinical_actions              # extracted prescriptions / orders
└── review_flags
└── completeness
```

Because every field defaults and unknown keys are kept, the schema
validates **even when sections come back empty or specialty-extended** —
this is exactly what fixed the earlier 500 (`Field required` ×5) that the
old flat schema caused against the rewritten nested prompt.

`_parse_soap_json` is the last safety net: if the LLM emits prose instead
of JSON (or markdown), the endpoint still returns a structured body
(`error: JSON_PARSING_FAILED, raw_output: …`) instead of a 500 or an
empty 200.

## What the prompt expects vs. what is sent today  (known gap)

The master prompt (`soapprompt.py`, section `2. Input Context`) documents
that it "may receive some or all of the following inputs":

| Prompt input | Currently sent by `generate()`? |
|---|---|
| `consultation_transcript` | ✅ yes (as the joined `conversation`) |
| `patient_profile` | ❌ no |
| `encounter_metadata` | ❌ no |
| `vital_signs` | ❌ no |

`generate()` builds its user message from `conversation` only, so the
`patient`, `encounter`, and `vital_signs` sections of the output rely on
the LLM inferring them from dialogue — they frequently come back null
or placeholder, even though the response schema accepts them.

**The fix:** extend `generate()` (and `ConversationRequest`) to forward
the richer inputs, e.g.

```python
class ConversationRequest(BaseModel):
    conversation: list[str]
    patient_profile: dict | None = None
    encounter_metadata: dict | None = None
    consultation_transcript: list[str] | None = None
    vital_signs: list[dict] | None = None
```

and build the user message from those fields before joining the
transcript. Until then the SOAP note is generated, but the structured
clinical context fields are best-effort rather than sourced.

## Observability

- Every call opens an MLflow run titled `Generate SOAP Note`
  (`start_mlflow_run` / `end_mlflow_run`) — visible when
  `MLFLOW_ENABLED=true`.
- The agent name `SOAPScribeAgent` is set on the session for run
  tagging.
- The response is the parsed dict (not the raw streamed tokens), so the
  route returns clean JSON rather than an SSE stream.

## Testing

- `test/test_agent_gating.py` — `soap` appears in the OpenAPI schema
  only when `enabled: true`; flipping it off removes the route.
- `test/test_registries.py` — `soap` prompt is registered and resolves
  to the `SOAP_PROMPT` content.
- `test/test_llm_litellm.py` — model fallback chain that the SOAP agent
  rides on (`clinical_model → gemini → ollama`) if `gpt-4.1` is
  unavailable.
