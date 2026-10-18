# IntakeAgent (pre-visit intake)

`IntakeAgent` gathers patient information **before** a visit through an
interactive, streaming conversation. It is the `AgentType.INTAKE` agent and
lives in `customagents/previsitagent/`.

## Files

| File | Role |
|---|---|
| `customagents/previsitagent/intakeagent.py` | `IntakeAgent` class |
| `customagents/previsitagent/intakeprompt.py` | `INTAKE_PROMPT` |
| `customagents/previsitagent/config.yaml` | per-agent YAML config |
| `api/routers/consult.py` | `POST /api/agent/intake` route (shared with consult) |

## Configuration

```yaml
name: intake
model: clinical_model
prompt: intake
tools:
  - memory
```

## Runtime flow — `POST /api/agent/intake` (streaming)

Same NDJSON streaming shape as `/consult`. `consult.py` exposes both `/consult`
and `/intake`, and the router is mounted when **either** agent is enabled.

`IntakeAgent.run(message)` mirrors the consult agent:

1. Initializes the session; sets `agent_name = "IntakeAssistant"`.
2. Builds the full system prompt (`INTAKE_PROMPT` + system prompt + user
   memory) via `get_system_prompt(..., user_memory=...)`.
3. **Greets once per session** — `"Hello, I'm your medical intake assistant.
   I'll gather some information before your visit."` — then returns early if
   the message was empty.
4. Streams the agentic loop, tagging events with the `intake` agent type.
5. **End-of-intake detection** — matches the reply (lowercased, substring)
   against `END_PATTERNS`:
   ```
   "intake is complete", "intake has been completed",
   "thank you. your intake", "information will be available for your doctor"
   ```
   On a match it emits `AgentEvent.status_end`.

## Difference from the voice intake agent

`IntakeAgent` (this one) is **text**. `VoiceIntakeAgent` is the speech-driven
counterpart — same greet-once and end-of-conversation pattern, but its
`END_PATTERNS` are **regexes** matched against punctuation-stripped text. See
**voice-intake-agent.md**.

## Testing

- `test/test_agent_gating.py` — the consult router (serving both
  `/api/agent/consult` and `/api/agent/intake`) mounts when *either* agent is
  enabled.
- `test/test_registries.py` — `intake` prompt resolves.
