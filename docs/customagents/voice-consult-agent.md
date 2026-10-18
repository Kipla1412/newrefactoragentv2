# VoiceConsultAgent (spoken clinical consultation)

`VoiceConsultAgent` is the **speech-driven** counterpart of `ConsultingAgent`:
it takes a transcript of the spoken consultation and returns clinical guidance.
It is the `AgentType.VOICE_CONSULT` agent.

## Files

| File | Role |
|---|---|
| `customagents/voiceagent/voiceconsult.py` | `VoiceConsultAgent` class |
| `customagents/voiceagent/voiceconsultprompt.py` | `VOICE_CONSULT_PROMPT` |
| `customagents/voiceagent/voice_consult.config.yaml` | per-agent YAML config |
| `customagents/voiceagent/voiceagent.py` | shared helpers (`build_patient_context_section`, `translate_text`, `LANGUAGE_MAP`) |
| WS router (`/ws`) | mounted when `voice_consult` **or** `voice_intake` is enabled |

## Configuration

```yaml
name: voice_consult
model: clinical_model
prompt: voice_consult
tools:
  - memory
```

## Runtime flow

`run(transcript: str)`, identical in shape to `VoiceIntakeAgent`:

1. Initializes the session; sets `agent_name = VoiceConsultAgent`.
2. Appends `build_patient_context_section(session.metadata["patient_context"])`
   to the role prompt when present.
3. `get_system_prompt(...)`, then `add_user_message(transcript)`.
4. Streams `_agentic_loop()`, tagging events with the `voice_consult` agent
   type and accumulating `TEXT_DELTA`.
5. **End-of-conversation detection** — punctuation-stripped, lowercased text
   matched against **regex** `END_PATTERNS`:
   ```
   r"consultation (?:is|has been) (?:now )?complete"
   r"complet(?:e|es|ed) (?:our|the) consultation"
   r"conclud(?:e|es|ed) (?:our|the) consultation"
   r"in person evaluation"
   r"see a doctor"
   r"emergency department"
   r"follow up instructions"
   ```
   On a match it emits `AgentEvent.status_end`.
6. `start_mlflow_run(transcript)` / `end_mlflow_run()`.

## Difference from ConsultingAgent

| | `ConsultingAgent` (text) | `VoiceConsultAgent` |
|---|---|---|
| Input | `message: str` per turn | a whole `transcript` |
| Greeting | emits a one-time greeting | none (the voice UI greets) |
| End detection | substring `END_PATTERNS` | regex on normalized text |
| Patient context | — | injected from session metadata |

## Testing

- `test/test_agent_gating.py` — the `/ws` router mounts when *either* voice
  agent is enabled.
- `test/test_registries.py` — `voice_consult` prompt resolves.
