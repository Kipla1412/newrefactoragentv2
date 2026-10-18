# Logging

One reusable logging layer for the whole framework: configured once at an
entrypoint, obtained the same way from every module.

## The API

```python
from utils.logging_config import get_logger

logger = get_logger(__name__)
```

That is the entire convention. Two rules make it reusable:

- **Library modules never configure logging.** They only call `get_logger`.
  `setup_logging()` is called by entrypoints (`api/main.py`, `main.py`), so the
  package stays importable from a CLI, a server, or a test runner without
  duplicate handlers or import-time side effects.
- **`get_logger` configures on first use** if nobody else has, so a module
  imported outside an entrypoint still produces output instead of silently
  discarding records. `setup_logging` is idempotent — the first call wins.

```python
setup_logging(level=None, fmt=None, log_file=None, force=False)
```

`force=True` reconfigures (used by tests). Handlers are replaced, never
stacked.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `LOG_LEVEL` | Root level: `DEBUG`/`INFO`/`WARNING`/`ERROR` | `INFO` |
| `LOG_FORMAT` | `text` or `json` | `text` |
| `LOG_FILE` | Optional extra file sink | *(none)* |
| `LOG_THIRD_PARTY_LEVEL` | Level for noisy third-party loggers | `WARNING` |
| `LOG_PROMPTS` | Log message content and transcripts | `false` |

### Text (default)

```
2026-10-16 13:21:39 | INFO     | api.main | Session initialized
2026-10-16 13:21:39 | WARNING  | client.llm_client | Previous model failed; falling back to gemini/gemini-2.0-flash
```

Chosen as the default because this framework is usually run locally under
`uvicorn` and read in a terminal.

### JSON

```json
{"timestamp": "2026-10-16 13:21:39", "level": "INFO", "logger": "agent.session", "message": "Session initialized", "session_id": "8f2c…", "tools": 12}
```

Opt in with `LOG_FORMAT=json` when shipping into a collector.

### Structured fields

Pass a mapping under the reserved `fields` key — it is flattened into the JSON
payload so collectors can query it directly:

```python
logger.info("Tool call", extra={"fields": {"tool": "wiki_search", "success": True}})
```

The core keys (`timestamp`, `level`, `logger`, `message`) always win, so a
field can never spoof them.

## Safety

This is a clinical framework, so two rules are enforced by design:

**Secrets never reach a log line.** `redact()` masks any key that looks like a
credential (`api_key`, `authorization`, `*token*`, `*secret*`, `*password*`):

```python
from utils.logging_config import redact
logger.debug("Call kwargs", extra={"fields": redact(call_kwargs)})
```

**Patient data is not logged by default.** Message content is gated behind
`LOG_PROMPTS`, which is off:

```python
from utils.logging_config import describe_messages, prompts_enabled

# Always safe: structure and sizes, never content.
logger.debug("LLM request", extra={"fields": describe_messages(messages)})   # {message_count, roles, content_chars}

if prompts_enabled():                                                       # opt-in
    logger.debug("LLM request content", extra={"fields": {"messages": messages}})
```

The same gate covers voice transcripts in `api/wsrouters/` and
`customagents/voiceagent/`, and the `[PROCESS]`/`[TRANSCRIPT]` paths.
Turning `LOG_PROMPTS=true` puts patient dialogue in your logs — deliberately.

## Third-party noise

LiteLLM logs every completion at INFO, and `httpx`, `urllib3`, `neo4j`,
`opensearch`, `mlflow`, `boto3` are similarly chatty. They are pinned to
`LOG_THIRD_PARTY_LEVEL` (default `WARNING`) so framework records stay readable.
Raise it to `DEBUG` when you need to see library internals.

## Where it is wired

| Entrypoint | Call |
|---|---|
| `api/main.py` | `setup_logging()` at import, before the app is built |
| `main.py` | `setup_logging()` at the start of the `main()` command |

Both are idempotent, so starting the API and the CLI in one process is safe.

## What was converted

The framework previously had a single `logging.basicConfig` in `api/main.py`
and ~87 `print()` calls — diagnostics that no level, filter, or test could
control. Diagnostics now go through the logger, with levels chosen by meaning
(`warning` in `except` blocks, `info` for lifecycle, `debug` for per-turn
internals):

| Area | Examples |
|---|---|
| `agent/agent.py` | approval requests, tool call start/finish, max-turns error, skill activation |
| `agent/session.py` | session initialization (with tool count), confirmation handshake, MLflow warnings |
| `client/llm_client.py` | LLM request metadata, model fallback warnings, all-models-failed error |
| `safety/approval.py` | confirmation request and resolution |
| `tools/registry.py`, `tools/mcp/*` | approval decisions, MCP connection and tool-call failures |
| `hooks/hook_system.py`, `api/auth.py`, `api/main.py` | hook failures, JWT decode failures |
| `utils/mlflow_tracker.py` | tracking URI, autolog, initialization failures |
| `customagents/voiceagent/`, `speechtospeech/`, `api/wsrouters/` | provider lifecycle, language detection, translation, session start/close (content gated) |

Program output is intentionally left on stdout: `scripts/reindex_clinical_documents.py`
progress, the interactive console in `test/checkmedical.py`, and the TUI in
`main.py` are user-facing output, not diagnostics.

## Testing

`test/test_logging.py` (50 tests) covers logger acquisition and one-time
configuration, idempotent/forced setup, level and format handling, JSON output
including exceptions and unserializable values, structured fields and
core-key protection, the file sink, third-party pinning, `redact`,
`describe_messages`, the `LOG_PROMPTS` gate, and three guards:

- **no `print()` remains** in the converted core modules (AST check per file)
- every converted module **exposes a module-level `logger`**
- **no framework module bypasses `get_logger`** — `utils/logging_config.py` is
  the only file allowed to touch stdlib `logging`

Those guards are what keep this reusable: a new `print`, or a new
`logging.getLogger` call, fails the suite.
