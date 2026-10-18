# DoctorAssistantAgent (question recommendations)

`DoctorAssistantAgent` reviews a consultation conversation and suggests
follow-up questions for the clinician — the "magic button" feature. It is the
`AgentType.DOC` agent.

## Files

| File | Role |
|---|---|
| `customagents/doctorassistent/docagent.py` | `DoctorAssistantAgent` class |
| `customagents/doctorassistent/docprompt.py` | `DOC_PROMPT` |
| `customagents/doctorassistent/config.yaml` | per-agent YAML config |
| `api/routers/doc.py` | `POST /api/agent/doctoragent` route |

## Configuration

```yaml
name: doc
model: clinical_model
prompt: doc
tools:
  - memory
```

## Two entry points

The class exposes both a non-streaming and a streaming path over the same
prompt and loop.

### `recommend_questions(conversation: list[str], skill: str | None = None) -> dict` — non-streaming

This is what the `/doctoragent` route uses. `skill` optionally names a skill
whose instructions are activated for that call only — see
**[../skills.md](../skills.md)**.

1. Initializes the session; sets `agent_name = "DoctorAssistant"`
2. **Clears the context** and installs only `DOC_PROMPT`, so the
   recommendation is based purely on the supplied conversation.
3. Sends one user message:
   `"Review this conversation and suggest 3 questions:\n{history_block}"`
4. **Consumes the stream internally** — accumulates `TEXT_DELTA`, replaces on
   `TEXT_COMPLETE` — then returns `parse_questions(response_text)`.
5. On failure returns `{"questions": [], "error": str(e)}`; `parse_questions`
   strips markdown fences and falls back to
   `{"questions": [], "error": "Invalid JSON format received", "raw": <100 chars>}`.

Because it drains the generator, the HTTP response is a single JSON body, not
SSE — appropriate for a button that just needs a list.

### `run(message: str = "", skill: str | None = None)` — streaming

Same shape as the other streaming agents: installs `DOC_PROMPT` as the system
prompt, runs `_agentic_loop()`, tags events with the `doc` agent type, and
forwards `AgentEvent`s. Use it if the UI wants to show the assistant typing.

`skill` optionally activates a skill through the reusable `Agent.use_skill()`
seam for that request only — see **[../skills.md](../skills.md)**. Unlike
`recommend_questions`, `run()` keeps the existing conversation history instead
of clearing it.

## Relationship to other agents

- Typically called with the transcript of a `/consult` session, to prompt the
  doctor for the next question.
- Independent of the `/consultation-workflow` chain (consult → soap →
  assessment → extraction).

## Testing

- `test/test_agent_gating.py` — `/api/agent/doctoragent` presence follows the
  `doc` enabled flag.
- `test/test_registries.py` — `doc` prompt resolves.
