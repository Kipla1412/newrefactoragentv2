# AssessmentAgent (medical assessment report)

`AssessmentAgent` produces a structured **medical assessment report** from a
consultation transcript. It is the `AgentType.ASSESSMENT` report agent and the
middle stage of `/consultation-workflow` (between consult and clinical
extraction).

## Files

| File | Role |
|---|---|
| `customagents/reportagent/assesment.py` | `AssessmentAgent` class |
| `customagents/reportagent/assessmentprompt.py` | `ASSESSMENT_PROMPT` |
| `customagents/reportagent/assessment.config.yaml` | per-agent YAML config |
| `api/routers/assessment.py` | `POST /api/agent/assessment` route |

> The module is `assesment.py` (one `s`) while the class and config use the
> correct spelling. The config file is `assessment.config.yaml`; the loader
> keys it by `name: assessment`.

## Configuration

```yaml
name: assessment
model: clinical_model   # openai/gpt-4.1 via ModelRegistry
prompt: assessment      # logical name -> ASSESSMENT_PROMPT
tools:
  - memory
```

## Runtime flow — `POST /api/agent/assessment`

Non-streaming request/response (unlike `/consult`).

`AssessmentAgent.generate(conversation: list[str])`:

1. Initializes the session; sets `agent_name = "AssessmentReporter"`.
2. **Clears the context** and installs only `ASSESSMENT_PROMPT`, so the report
   is generated from the transcript rather than accumulated chat history.
3. Sends one user message:
   `"Generate a medical assessment report based on this transcript:\n{transcript}"`
4. Consumes the agentic loop (`TEXT_DELTA` accumulated, `TEXT_COMPLETE`
   replaces) and returns parsed JSON.
5. `_parse_report()` strips ```json fences; on failure it returns
   `{"error": "JSON Parsing Failed", "raw_content": <first 500 chars>}` so the
   client still gets a debuggable body instead of a 500.

## Relationship to other agents

- Its output feeds `ClinicalExtractionAgent.generate(soap_note, assessment_report)`.
- Mounted only when `agents.assessment.enabled == true`.
- `/consultation-workflow` requires **soap + assessment + clinical_extraction**
  all enabled; otherwise use the individual endpoints.

## Testing

- `test/test_agent_gating.py` — `/api/agent/assessment` presence follows the
  `assessment` enabled flag.
- `test/test_registries.py` — `assessment` prompt resolves.
