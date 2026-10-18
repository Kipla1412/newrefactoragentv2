# VoiceIntakeAgent (spoken pre-visit intake)

`VoiceIntakeAgent` is the **speech-driven** counterpart of `IntakeAgent`: it
takes a transcript of the patient speaking and drives the pre-visit intake
conversation. It is the `AgentType.VOICE_INTAKE` agent.

## Files

| File | Role |
|---|---|
| `customagents/voiceagent/voiceintake.py` | `VoiceIntakeAgent` class |
| `customagents/voiceagent/voiceintakeprompt.py` | `VOICE_INTAKE_PROMPT` |
| `customagents/voiceagent/voice_intake.config.yaml` | per-agent YAML config |
| `customagents/voiceagent/voiceagent.py` | shared helpers (`build_patient_context_section`, `translate_text`, …) |
| WS router (`/ws`) | mounted when `voice_intake` **or** `voice_consult` is enabled |

## Configuration

```yaml
name: voice_intake
model: clinical_model
prompt: voice_intake
tools:
  - memory
```

## Runtime flow

`run(transcript: str)` — unlike the text agents, the input is already a
transcript string produced by the STT layer.

1. Initializes the session; sets `agent_name = VoiceIntakeAgent`.
2. **Patient context injection** — if `session.metadata["patient_context"]`
   exists, `build_patient_context_section(...)` is appended to the role prompt
   so the agent knows who it is speaking with.
3. `get_system_prompt(config, role_prompt=...)` then
   `add_user_message(transcript)`.
4. Streams `_agentic_loop()`, tagging events with the `voice_intake` agent
   type and accumulating `TEXT_DELTA`.
5. **End-of-conversation detection** — the accumulated reply is
   **punctuation-stripped and lowercased**
   (`re.sub(r"[^a-z0-9]+", " ", text.lower())`) and matched against **regex**
   `END_PATTERNS`:
   ```
   r"intake (?:is|has been) (?:now )?complete"
   r"complet(?:e|es|ed) (?:your|the) intake"
   r"information will be available for your doctor"
   ```
   On a match it emits `AgentEvent.status_end`.
6. `start_mlflow_run(transcript)` / `end_mlflow_run()`.

## Why regex patterns

The text agents (`ConsultingAgent`, `IntakeAgent`) use plain substring
`END_PATTERNS`. The voice agents normalize the text first and use regexes, so
STT punctuation and casing differences ("Intake is complete!" vs
"intake is complete") still match.

## Testing

- `test/test_agent_gating.py` — the `/ws` router mounts when *either*
  `voice_intake` or `voice_consult` is enabled.
- `test/test_registries.py` — `voice_intake` prompt resolves.
