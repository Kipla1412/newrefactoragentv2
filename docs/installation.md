# Installation

Get the framework running from a clean checkout. Takes about five minutes plus
dependency download time.

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.12+** | `pyproject.toml` declares `requires-python = ">=3.12"`. Check with `python3 --version`. |
| **git** | To clone the repository. |
| **uv** *(recommended)* | `curl -LsSf https://astral.sh/uv/install.sh \| sh` — a lockfile (`uv.lock`) is committed, so this gives reproducible installs. |
| **An LLM API key** | Any provider LiteLLM supports. `API_KEY` for OpenAI-compatible gateways. |

Optional, only for specific features:

| Service | Needed for |
|---|---|
| FHIR server (`FHIR_BASE_URL`) | `ehr` and `demographics` agents (they call `fhir__*` MCP tools) |
| Sarvam key (`SARVAM_API_KEY`) | voice agents, diarization |
| Ollama | local models / offline fallback |
| OpenSearch | document retrieval (`document_retrieval`) |
| Neo4j | the medical knowledge-graph agent |
| MLflow | run tracking (off by default) |

## 1. Clone and enter the project

```bash
git clone <repo-url> refactorcode
cd refactorcode
```

> **Run every command from this directory.** The application loads `agents.yaml`
> and `customagents/` relative to the **current working directory**, so starting
> the server from anywhere else fails with
> `Agent '<name>' is enabled but has no config.yaml`.

## 2. Install dependencies

Pick one path.

### With uv (recommended)

```bash
uv sync
```

Creates `.venv/` and installs the exact versions in `uv.lock`. Run everything
afterwards with `uv run <command>`.

### With pip

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` and `pyproject.toml` list the same 26 packages, so either
works.

### With Docker

A `Dockerfile` is included (Python 3.12-slim, `uvicorn` on port 8000):

```bash
docker build -t clinical-agent .
docker run --rm -p 8000:8000 --env-file .env clinical-agent
```

## 3. Configure the environment

```bash
cp example.env .env
```

Then edit `.env`. The file is loaded automatically from the project root — you
do not need to export anything.

### Minimum to get a response

```env
# The gateway key (OpenAI, Azure AI Foundry, OpenRouter, Groq, ...)
API_KEY=sk-...

# Optional: point at a non-OpenAI OpenAI-compatible gateway
# BASE_URL=https://api.openai.com/v1
```

That is enough for the text agents, because `agents.yaml` maps
`clinical_model → openai/gpt-4.1` by default.

### Optional, grouped by feature

```env
# --- Other providers (native keys read by LiteLLM) -------------------------
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
OLLAMA_API_BASE=http://localhost:11434     # or OLLAMA_BASE_URL

# --- Fallback chain, tried in order when the primary fails -----------------
LLM_FALLBACK_MODELS=gemini,ollama
LLM_ROUTING_ENABLED=false
LLM_TIMEOUT=60
LLM_NUM_RETRIES=3

# --- Vision model for PDF OCR (logical name or raw model string) -----------
VISION_MODEL_NAME=vision

# --- FHIR over MCP: enables the ehr and demographics agents ----------------
FHIR_BASE_URL=http://localhost:8002/mcp/

# --- Auth (has a default; override for your own IAM) ------------------------
IAM_JWKS_URL=https://your-iam/api/auth/jwks
IAM_ISSUER=https://your-iam

# --- Voice (STT/TTS) --------------------------------------------------------
SARVAM_API_KEY=...
STT_PROVIDER=sarvam
TTS_PROVIDER=sarvam

# --- Observability ----------------------------------------------------------
MLFLOW_ENABLED=false
LOG_LEVEL=INFO                 # DEBUG | INFO | WARNING | ERROR
LOG_FORMAT=text                # text | json
LOG_THIRD_PARTY_LEVEL=WARNING
LOG_PROMPTS=false              # true puts patient content in your logs
```

`agents.yaml` and `customagents/` are already in the repository — no setup
needed.

## 4. Verify the install

```bash
uv run python -c "import api.main; print('ok')"
```

This also validates the configuration: an unknown model, prompt, skill, or
agent class fails here with a specific message rather than at request time.

Check which agents loaded:

```bash
uv run python -c "
import api.main as m
print(len(m.app.openapi()['paths']), 'routes')
print(sorted(m.app.openapi()['paths']))
"
```

## 5. Run it

### HTTP API

```bash
uv run uvicorn api.main:app --reload --port 8000
```

- Swagger UI: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health> → `{"status":"ok"}`

Or with Docker: `docker run --rm -p 8000:8000 --env-file .env clinical-agent`.

### CLI

```bash
uv run python main.py "Summarise this repository"
uv run python main.py            # interactive
```

## 6. Run the tests

```bash
uv run --with pytest --with pytest-asyncio pytest test/ -q
```

Expect `305 passed`. No network or API key is required — the tests stub the LLM
client.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Agent '<name>' is enabled but has no config.yaml` | Not running from the project root. `cd` to the directory containing `agents.yaml`. |
| `Agent '<name>' references unknown skill '<skill>'` | A `config.yaml` names a skill with no `SKILL.md` under `skills/library/`. |
| `ModuleNotFoundError: api` | Dependencies not installed, or the venv is not active. Re-run step 2. |
| `No API key found. Set API_KEY environment variable` | `.env` missing or `API_KEY` unset. |
| `[MCP CONNECTION ERROR] Client failed to connect` | `FHIR_BASE_URL` is set but no server is listening. Harmless for non-FHIR agents; unset it or start the FHIR server. |
| Requests return an empty body | The model call failed and a permissive response schema returned defaults. Check the logs for the `LiteLLM completion() model=…` line — if the provider has no key, set the right one or add `LLM_FALLBACK_MODELS`. |
| `403` on an `/api/agent/*` route | Missing `consultagent:chat` permission on your token. |
| `409`/port in use | Another process on 8000: `uv run uvicorn api.main:app --port 8001`. |

## Next steps

| If you want to... | Read |
|---|---|
| Understand how a request flows | [README → How a request flows](../README.md#how-a-request-flows) |
| Add a new agent | [customagents/new-agent.md](customagents/new-agent.md) |
| Configure models, fallbacks, routing | [litellm.md](litellm.md) |
| Add or edit skills | [skills.md](skills.md) |
| Set log levels, JSON output, or PHI behaviour | [logging.md](logging.md) |
