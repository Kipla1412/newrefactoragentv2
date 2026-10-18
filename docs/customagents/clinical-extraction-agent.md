# ClinicalExtractionAgent (FHIR concept extraction)

`ClinicalExtractionAgent` turns already-produced clinical documents (a SOAP
note, optionally an assessment report) into **structured FHIR-style
concepts**. It is the `AgentType.CLINICAL_EXTRACTION` agent and the final
stage of `/consultation-workflow`.

## Files

| File | Role |
|---|---|
| `customagents/reportagent/clinicalextraction.py` | `ClinicalExtractionAgent` class |
| `customagents/reportagent/clinicalprompt.py` | `CLINICAL_EXTRACTION_PROMPT` |
| `customagents/reportagent/clinical_extraction.config.yaml` | per-agent YAML config |
| `api/routers/clinical.py` | `POST /api/agent/clinical-extraction` route |

## Configuration

```yaml
name: clinical_extraction
model: clinical_model
prompt: clinical_extraction
tools:
  - memory
```

## Runtime flow — `POST /api/agent/clinical-extraction`

`generate(soap_note: dict, assessment_report: dict | None = None)`:

1. Initializes the session; sets `agent_name = "ClinicalExtractionAgent"`.
2. Clears the context and installs `CLINICAL_EXTRACTION_PROMPT`.
3. Builds a single JSON document as the user message:
   ```json
   { "soap": { … }, "assessment": { … } }
   ```
   `assessment` is included only when provided. Serialized with
   `json.dumps(..., indent=2, ensure_ascii=False)`.
4. Runs the agentic loop and parses with `_parse_output()`.

### Output contract

`_parse_output()` strips markdown fences and returns **exactly** these four
keys, each defaulting to `[]`, so consumers get a stable shape even from a
partial model response:

```json
{
  "conditions": [],
  "observations": [],
  "medicationRequests": [],
  "serviceRequests": []
}
```

This is a **narrowing** parser — any extra keys the model emits are dropped.
On unparseable output it returns
`{"error": "JSON_PARSING_FAILED", "details": …, "raw_output": <first 1000 chars>}`.

## Relationship to other agents

- Consumes the SOAP agent's output shape (see **soap-agent.md**).
- Last of the three agents required by `/consultation-workflow`.
- Because it accepts `soap_note` / `assessment_report` as **dicts**, callers can
  chain `/soap` → `/assessment` → `/clinical-extraction` and pass the previous
  JSON bodies straight through.

## Testing

- `test/test_agent_gating.py` — `/api/agent/clinical-extraction` gated by
  `clinical_extraction`, and `/api/agent/consultation-workflow` gated by all
  three report agents.
- `test/test_registries.py` — `clinical_extraction` prompt resolves.
