# DiarizeAgent (speaker-attributed transcription)

`DiarizeAgent` accepts a live audio stream, transcribes it, and returns a
**fully diarized transcript with human-readable speaker labels**
(`Doctor:` / `Patient:`). It is the `AgentType.DIARIZE` agent — the only agent
that does not call an LLM for its primary job (it briefly uses one for
speaker-role mapping).

## Files

| File | Role |
|---|---|
| `customagents/diarizeagent/diarizeagent.py` | `DiarizeAgent` class |
| `customagents/diarizeagent/diarizeprompt.py` | `DIARIZE_ROLE_MAP_PROMPT` |
| `customagents/diarizeagent/config.yaml` | per-agent YAML config |
| WS diarize route | mounted at `/ws` when `diarize` is enabled |

Supporting modules (not part of `customagents/`):

- `speechtospeech/hybrid/orchestrator.py` — `DiarizationOrchestrator`
- `speechtospeech/hybrid/stt_provider.py` — `HybridSTTProvider`, `_relabel_segments`
- `speechtospeech/webvad.py` — `VoiceActivityDetector`

## Configuration

```yaml
name: diarize
model: clinical_model
prompt: diarize       # role-mapping prompt (DIARIZE_ROLE_MAP_PROMPT)
tools:
  - memory
```

## Runtime flow

```python
async for event in agent.run(
    audio_stream,               # AsyncGenerator[bytes] — raw PCM16 chunks
    silence_timeout=15.0,
    max_session_duration=600.0,
    num_speakers=2,
    on_partial=callback,        # optional: forward partial transcripts (e.g. WS)
):
    ...
```

1. Emits `agent_start` and sets `agent_name`.
2. Builds a `HybridSTTProvider` (Sarvam STT, `language_code="unknown"`) and a
   `VoiceActivityDetector` (`threshold=0.05`, `silence_duration=silence_timeout`),
   then a `DiarizationOrchestrator` wiring them together.
3. Connects the provider and runs the orchestrator inside an MLflow span
   (`trace_agent_run("hybrid_diarization")`). The session auto-ends on
   `silence_timeout` of silence or `max_session_duration`.
4. **Speaker labelling** — `_label_segments(segments, provider)`:
   - No segments → falls back to the provider's streaming transcript.
   - Otherwise builds `SPEAKER_<id>: text` lines, calls the LLM with
     `DIARIZE_ROLE_MAP_PROMPT`, and parses `doctor_speaker_id` /
     `patient_speaker_id` from the JSON reply.
   - **Fallback chain:** LLM role mapping fails or returns nothing → the
     deterministic heuristic `_relabel_segments()` (character-count based).
5. **Translation** — the labelled transcript is passed through
   `translate_text(..., target_language="en")`, so regional-language
   consultations arrive in English. A translation failure is logged and the
   original transcript is used.
6. Emits `AgentEvent.text_complete(final_transcript)` and records the final
   transcript in the context manager.
7. MLflow logs metrics (`session_duration_seconds`, `transcript_length_chars`,
   `num_speakers`) and params (`silence_timeout`, `max_session_duration`).

Errors yield `agent_error` **and re-raise**, so the WS route sees the failure.

## Design notes

- Speaker labels are the human-readable roles `Doctor` / `Patient` — never bare
  `SPEAKER_00` / `0:` — with `Speaker_<id>` as the last-resort label.
- The LLM is used only for the small role-mapping decision; the heuristic is
  the safety net, so a transcript is still produced if the model is down.

## Testing

- `test/test_agent_gating.py` — the diarize WS router mounts only when
  `diarize` is enabled.
