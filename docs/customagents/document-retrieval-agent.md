# DocumentRetrievalAgent (RAG over indexed medical documents)

`DocumentRetrievalAgent` answers questions grounded in the **indexed medical
document collection**. It is the `AgentType.DOCUMENT_RETRIEVAL` agent.

## Files

| File | Role |
|---|---|
| `customagents/documentagent/documentagent.py` | `DocumentRetrievalAgent` class |
| `customagents/documentagent/documentprompt.py` | `DOCUMENT_RETRIEVAL_PROMPT` |
| `customagents/documentagent/config.yaml` | per-agent YAML config |
| `api/routers/document.py` | `POST /api/agent/document` route |

## Configuration

```yaml
name: document_retrieval
model: clinical_model
prompt: document_retrieval
# No tools key: DocumentRetrievalAgent sets its own allowed_tool_names
# (jina_embedding, medical_document_search) in its constructor.
```

This is the second agent that deliberately **omits** `tools:` — and for a
stricter reason than `ehr`. The constructor hard-codes its tool set:

```python
self.force_tool_choice = True          # retrieval is mandatory
self.allowed_tool_names = ["jina_embedding", "medical_document_search"]
```

- `force_tool_choice = True` means the model **must** search the documents
  before answering — it can never emit a bare LLM answer.
- Restricting to just the two retrieval tools prevents the model from routing
  the question through unrelated tools (e.g. `memory`).

## Runtime flow — `POST /api/agent/document` (streaming)

`run(message)`:

1. Raises `ValueError("Document retrieval message cannot be empty.")` on an
   empty/blank message (before touching the session).
2. Initializes the session; sets `agent_name = "DocumentRetrievalAgent"`.
3. **Injects the retrieval scope** from session metadata into the system
   prompt, so the agent uses real identifiers instead of guessing placeholders:
   ```
   ## Retrieval Scope (authorized by the application)
   patient_id: 10008          # or "(none provided)"
   file_id: abc-123           # or "(none provided)"
   ```
   Read from `session.metadata["patient_id"]` / `["file_id"]`.
4. Builds the full prompt via `get_system_prompt(..., user_memory=...)`, adds
   the question, and runs `_agentic_loop()`, tagging events with the agent type.
5. `start_mlflow_run(message)` / `end_mlflow_run()`.

## Relationship to other agents

- Distinct from `PDFChatAgent`: this one searches the **persistent, indexed**
  collection; `PDFChatAgent` chats over a **single uploaded PDF** held in
  session state.

## Testing

- `test/test_agent_gating.py` — `/api/agent/document` presence follows the
  `document_retrieval` enabled flag.
- `test/test_registries.py` — the factory does **not** override this agent's
  own `allowed_tool_names` when `tools:` is omitted.
