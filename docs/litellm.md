# LiteLLM provider layer

LiteLLM is the single provider abstraction underneath the framework. **Agents
never import it.** Every LLM call goes through `LLMClient`
(`client/llm_client.py`), which wraps `litellm.acompletion` and normalizes the
result back into the framework's own event contract.

```
Agent  ──>  Session  ──>  LLMClient  ──>  litellm.acompletion  ──>  provider
                            │
                            ├── ModelRegistry      (logical name → LiteLLM string)
                            ├── ProviderCredentialResolver  (which keys/base URL)
                            └── ModelRouter        (task type → logical name)
```

## Model resolution

Agents are configured with **logical** model names, never LiteLLM strings:

```yaml
# customagents/soap/config.yaml
model: clinical_model
```

```yaml
# agents.yaml
models:
  clinical_model:
    provider: openai
    name: gpt-4.1
    litellm_model: openai/gpt-4.1
    temperature: 0.1
```

`LLMClient._resolve_model()`:

1. If the name is registered in `ModelRegistry` → `registry.resolve_litellm(name)`
   returns the `litellm_model` (or the `provider`/`name` pair).
2. Otherwise the string is passed through **unchanged** — so a raw
   `openai/gpt-4o` or `ollama/llama3.2` works without a `models:` entry.

The same rule applies to `VISION_MODEL_NAME`: `vision` (registered) resolves to
`openai/gpt-4o-mini`, while `gpt-4o-mini` is passed through raw.

## Credentials — one strategy per provider class

`client/providers.py` implements `ProviderCredentialResolver`, a
strategy/factory keyed by provider prefix (SRP + OCP — adding a provider means
adding a strategy, not editing a branch chain, and it is injected into
`LLMClient`, so DIP holds).

| Strategy | Applies to | Sends |
|---|---|---|
| `OpenAIGatewayCredentials` | `openai`, `openrouter`, `azure`, `deepseek`, `groq`, `together_ai`, `xai`, `fireworks_ai`, `mistral` | `api_key` = `API_KEY`, `api_base` = `BASE_URL` |
| `LocalServerCredentials` | `ollama` | `api_base` from `OLLAMA_API_BASE`, else `OLLAMA_BASE_URL` (whitespace trimmed). **Never** the OpenAI key/base |
| env fallback (no explicit strategy) | `anthropic`, `gemini`, … | nothing — LiteLLM reads its own env vars (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) |

Two rules that matter in practice:

- **Ollama never receives the OpenAI key or base URL.** Forwarding them would
  point the local client at the wrong server and leak a key it does not expect.
  `OLLAMA_API_BASE` takes priority over `OLLAMA_BASE_URL` when both are set.
- If neither Ollama variable is set, no `api_base` is forwarded and LiteLLM's
  default (`http://localhost:11434`) applies.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `API_KEY` | key for OpenAI-compatible providers | — |
| `BASE_URL` | base URL for OpenAI-compatible providers | — |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | provider-native keys read by LiteLLM | — |
| `OLLAMA_API_BASE` / `OLLAMA_BASE_URL` | local Ollama server | LiteLLM default |
| `VISION_MODEL_NAME` | model for PDF OCR; logical name or raw string | runtime model |
| `LLM_ROUTING_ENABLED` | enable `ModelRouter` | `false` |
| `LLM_FALLBACK_MODELS` | ordered fallback chain | empty (no fallback) |
| `LLM_TIMEOUT` | per-call timeout (seconds) | `60` |
| `LLM_NUM_RETRIES` | retries **per model** | `3` |

> `load_dotenv` does **not** override variables already present in the process
> environment. A stale `export VISION_MODEL_NAME=…` in your shell silently wins
> over `.env`.

## Streaming

`LLMClient.chat_completion()` returns an async generator of `StreamEvent`
objects from `client/response.py` (`StreamEventType`: text deltas, tool calls,
message complete). That contract is frozen — the rewrite onto
`litellm.acompletion` did not change it, so every agent kept working unchanged.
`stream=False` still yields a single `MESSAGE_COMPLETE` event, which is what
the buffered agents (`generate`, `recommend_questions`, `PDFOCR`) consume.

## Retry and fallback

For a call to model `M`, `LLMClient._candidate_models(M)` returns
`[M] + [fallbacks…]`, and the client walks that list:

1. Try the candidate; on a **retryable** error (rate limit, connection,
   timeout, 5xx) retry up to `LLM_NUM_RETRIES` times with exponential backoff.
2. On a **structural** error (auth, bad request, not found) move to the next
   candidate **immediately** — retrying an invalid key is pointless.
3. If every candidate fails → one `ERROR` event:
   `"All models failed. Last error: …"`.

Nothing is a fallback unless it is listed in `LLM_FALLBACK_MODELS`; the
`models:` block in `agents.yaml` only *defines* models. Values may be logical
names or raw LiteLLM strings.

Verified behaviour with `LLM_FALLBACK_MODELS=gemini,ollama`:

```
clinical_model  -> ['openai/gpt-4.1', 'gemini/gemini-2.0-flash', 'ollama/llama3.2']
reasoning       -> ['anthropic/claude-3-5-sonnet-20241022', 'gemini/…', 'ollama/llama3.2']

# primary + gemini forced to fail:
models attempted : ['openai/gpt-4.1' ×4, 'gemini/gemini-2.0-flash' ×4, 'ollama/llama3.2']
text returned    : fallback worked
```

At runtime, LiteLLM logs one `completion() model= …; provider = …` line per
attempt — **more than one line for a single request means a fallback fired**.

## Routing (opt-in)

`client/model_router.py` holds `ModelRouter`, which maps a *task type* to a
logical model:

| Task type | Routes to |
|---|---|
| `simple` | `fast` |
| `clinical` | `reasoning` |
| `retrieval` | `fast` |

The router is **pure logic with zero LiteLLM or provider imports**, and it is
**gated on `LLM_ROUTING_ENABLED`**:

```python
if (model is None
        and self.model_router is not None
        and self.config.llm_routing_enabled):
    model = self.model_router.route(messages=messages)
```

That gate matters: with routing off (the default) an agent's configured model
is **never** silently overridden. Without the flag check, a SOAP request
configured for `clinical_model` (gpt-4.1) was rerouted to `reasoning`
(Anthropic), which had no key — the call failed and the permissive response
schema returned HTTP 200 with an empty note. The flag is the fix, and
`test_client_does_not_route_when_routing_disabled` is the regression test.

## Adding a provider or model

1. **Model only** — add a `models:` entry in `agents.yaml` and reference it
   from a `customagents/*/config.yaml`. No code change.
2. **New provider that speaks the OpenAI API** — it is already covered by
   `OpenAIGatewayCredentials` if its name is in `OPENAI_GATEWAY_PROVIDERS`;
   otherwise add the name to that set.
3. **New local server** — add an entry to `LOCAL_SERVER_BASE_ENV` mapping the
   provider prefix to its base-URL env var(s).
4. **Anything else** — implement `ProviderCredentials.call_kwargs(model, config)`
   and register it in `default_resolver()`.

## Tests

`test/test_llm_litellm.py` (29 tests) covers: model resolution and
pass-through, provider credential strategies (including that Ollama never sees
the OpenAI key/base, and the `OLLAMA_BASE_URL` fallback with priority and
trimming), streaming events, the retry-then-fallback chain, all-models-failed
errors, routing behaviour, and the routing-disabled regression.
