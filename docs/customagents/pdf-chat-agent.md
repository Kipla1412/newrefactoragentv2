# PDFChatAgent (chat with an uploaded PDF)

`PDFChatAgent` answers questions about a **single uploaded PDF**, whose text is
extracted once and kept in the session. It is the `AgentType.PDF_CHAT` agent.

## Files

| File | Role |
|---|---|
| `customagents/pdfagent/pdfagent.py` | `PDFChatAgent` class |
| `customagents/pdfagent/pdfprompt.py` | `PDF_CHAT_PROMPT` |
| `customagents/pdfagent/config.yaml` | per-agent YAML config |
| `api/routers/pdf.py` | `POST /api/agent/pdf-chat` route (upload + chat) |
| `utils/pdf_extractor.py` | `PDFDocumentExtractor` — text layer + page rendering |
| `utils/pdf_ocr.py` | `PDFOCR` — vision-model OCR for scanned pages |

## Configuration

```yaml
name: pdf_chat
model: clinical_model
prompt: pdf_chat
tools:
  - memory
```

## Runtime flow — `POST /api/agent/pdf-chat`

Multipart form request (`message: str = Form(...)`, `file: UploadFile | None = File(None)`),
responses are NDJSON streams (`application/x-ndjson`) with an `X-Session-ID`
header.

### Upload (`file` present)

1. Read the bytes and extract the text layer with
   `PDFDocumentExtractor.extract_text(pdf_bytes)`.
2. **If the text layer is empty** (a scanned/image PDF), render pages and run
   `PDFOCR(session.client, model=config.vision_model_name).extract_text(images)`.
   The OCR path is the only consumer of `config.vision_model_name`, which
   resolves to the `vision` logical model (`openai/gpt-4o-mini`) — see
   **docs/litellm.md**.
3. Store the result on the session: `metadata["pdf_context"]`,
   `metadata["pdf_filename"]`. Re-uploading replaces it; follow-up turns can
   omit the file entirely.

No `pdf_context` → HTTP 400 `"No PDF context available. Upload a PDF file."`;
an extraction `ValueError` → HTTP 400 with its message.

### Chat turn

`AgentFactory.create(AgentType.PDF_CHAT, config, session)` then `agent.run(message)`.

`PDFChatAgent.run(message)`:

1. Errors with `agent_error` on an empty message.
2. Errors with `agent_error` if no `pdf_context` is in session metadata.
3. **Injects the PDF into the system prompt, regenerated every turn** — not
   into conversation history:
   ```
   <PDF_CHAT_PROMPT>

   ==================================================
   ACTIVE PDF DOCUMENT (source of truth)
   ==================================================

   <extracted pdf text>
   ```
   This keeps the document reusable across questions without duplicating it
   in the history on every turn.
4. Adds the user message and runs `_agentic_loop()`, tagging events with the
   agent type; `start_mlflow_run(message)` / `end_mlflow_run()` wrap it.

## Difference from DocumentRetrievalAgent

- `PDFChatAgent` — one uploaded document, held in session state, injected into
  the prompt.
- `DocumentRetrievalAgent` — the persistent indexed collection, reached through
  forced retrieval tools. See **document-retrieval-agent.md**.

## Testing

- `test/test_agent_gating.py` — `/api/agent/pdf-chat` presence follows the
  `pdf_chat` enabled flag.
- `test/test_registries.py` — `pdf_chat` prompt resolves.
- `test/test_llm_litellm.py` — `vision` resolves to `openai/gpt-4o-mini` and
  receives the OpenAI `API_KEY` / `BASE_URL`.
