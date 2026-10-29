# AI Agent

An AI agent that can execute tasks using tools and manage conversations.

> **New here?** Start with **[docs/installation.md](docs/installation.md)** —
> prerequisites, dependency install, `.env` setup, and running the API or CLI.

## How a request flows

```
Client
  │  POST /api/age1nt/demographics   { "message": "Summarise patient 10008" }
  ▼
FastAPI router ── require_permission ──► app.state.session_manager
  │                                          │
  │  clear context (one report per request)  ▼
  │                                        Session
  │                            ToolRegistry · MCPManager · LLMClient
  ▼
AgentFactory.create(AgentType.DEMOGRAPHICS)
  │
  ├─ agents.yaml      → is the agent enabled?
  ├─ config.yaml      → model · prompt · skill · tools
  ├─ ModelRegistry    → logical name  →  LiteLLM model string
  ├─ PromptRegistry   → prompt content
  └─ SkillRegistry    → the agent's default skill
  ▼
DemographicsAgent.run(message)
  │
  ├─ 1. ContextManager.set_system_prompt( role prompt
  │                                        + "# Active Skill"
  │                                        + SKILL.md instructions )
  ├─ 2. ContextManager.add_user_message(message)
  └─ 3. _agentic_loop() ─────────────────────────────────┐
                                                         │
        ┌────────────────────────────────────────────────┘
        ▼
   LLMClient.chat_completion(messages, tool_schemas)
        │  retry (3x) → fallback chain: primary → gemini → ollama
        └─► LiteLLM ──► provider (OpenAI / Anthropic / Gemini / Ollama)
                 │
                 ├── tool call ─► ToolRegistry ─► MCPTool ─► FHIR server
                 │                                  (caller's token)
                 │                result appended to context, loop again
                 │
                 └── no more tool calls ─► final text
        ▼
NDJSON stream   { tool_start | tool_result | text_delta | text_complete | error }
```

The skill is deactivated when the request finishes, so it never leaks into the
next one.

## Features

### Core Functionality

- Interactive and single-run modes
- Streaming text responses
- Multi-turn conversations with tool calling
- Configurable model settings and temperature

### Built-in Tools

- File operations: read, write, edit files
- Directory operations: list directories, search with glob patterns
- Text search: grep for pattern matching
- Shell execution: run shell commands
- Web access: search and fetch web content
- Memory: store and retrieve information
- Todo: manage task lists

### Context Management

- Automatic context compression when approaching token limits
- Tool output pruning to manage context size
- Token usage tracking

### Safety and Approval

- Multiple approval policies: on-request, auto, never, yolo
- Dangerous command detection and blocking
- Path-based safety checks
- User confirmation prompts for mutating operations

### Session Management

- Save and resume sessions
- Create checkpoints
- Persistent session storage

### MCP Integration

- Connect to Model Context Protocol servers
- Use tools from MCP servers
- Support for stdio and HTTP/SSE transports

### Subagents

- Specialized subagents for specific tasks
- Built-in subagents: codebase investigator, code reviewer
- Configurable subagent definitions with custom tools and limits

### Loop Detection

- Detects repeating actions
- Prevents infinite loops in agent execution

### Hooks System

- Execute scripts before/after agent runs
- Execute scripts before/after tool calls
- Error handling hooks
- Custom commands and scripts

### Configuration

- Configurable working directory
- Tool allowlisting
- Developer and user instructions
- Shell environment policies
- MCP server configuration

### User Interface

- Terminal UI with formatted output
- Command interface: /help, /config, /tools, /mcp, /stats, /save, /resume, /checkpoint, /restore
- Real-time tool call visualization

## Agent architecture

### Config-driven agents

- Two-level configuration: `agents.yaml` at the project root (models and `enabled` flags) plus `customagents/<agent>/config.yaml` (model, prompt, skill, tools).
- `AgentFactory` is the only code that knows the concrete classes; everything else refers to agents by name.
- Every reference — model, prompt, skill, tool, class — is validated at startup, so a typo fails loudly instead of at request time.
- An agent with no behaviour beyond "prompt + tools + skill" needs no new Python: set `agent_class: generic` in its config.

### Skills

- Reusable instruction sets: a directory containing a `SKILL.md` (YAML frontmatter plus a markdown body) under `skills/library/`.
- Loaded on demand and injected into the **existing** system message through `ContextManager` — no second context manager and no second prompt path.
- An agent declares a default skill in its `config.yaml`; a caller can override or disable it per request.
- Scoped per request — the instructions are removed when the call ends.
- Bundled: `clinical-followup-questions`, `ehr-fhir-retrieval`, `patient-demographics-summary`.

### LLM provider layer

- Agents never call a provider directly: every request goes Agent → Session → `LLMClient` → LiteLLM.
- Logical model names resolve through `ModelRegistry` (`clinical_model`, `fast`, `reasoning`, `vision`, `gemini`, `ollama`).
- Per-model retries plus an ordered fallback chain from `LLM_FALLBACK_MODELS`; fallbacks are logged with the model that actually ran.
- Optional task routing (`simple` / `clinical` / `retrieval`) behind `LLM_ROUTING_ENABLED`, off by default.
- Provider-aware credentials, including a separate path for local Ollama so it never receives the OpenAI key.

### Logging

- One reusable layer in `utils/logging_config.py`; modules call `get_logger(__name__)` and never configure logging themselves.
- `LOG_LEVEL`, `LOG_FORMAT` (`text` / `json`), `LOG_FILE`, `LOG_THIRD_PARTY_LEVEL`.
- Patient content is off by default — `LOG_PROMPTS=true` is required to log prompts or transcripts.

### Clinical agents

Fourteen agents, each with its own endpoint or WebSocket route and its own configuration:

`consult` · `intake` · `doc` · `soap` · `assessment` · `clinical_extraction` · `ehr` ·
`demographics` · `medical` · `document_retrieval` · `pdf_chat` · `voice_intake` ·
`voice_consult` · `diarize`

FHIR access goes through MCP tools named `fhir__*`, authorized with the caller's
token.

## Documentation

Detailed notes live in [`docs/`](docs/README.md):

| Document | Covers |
|---|---|
| [docs/installation.md](docs/installation.md) | Install, configure, and run — the starting point for new users |
| [docs/customagents.md](docs/customagents.md) | The config-driven agent system |
| [docs/customagents/new-agent.md](docs/customagents/new-agent.md) | Build guide for a new agent |
| [docs/customagents/](docs/customagents/README.md) | One page per agent, with a route map |
| [docs/litellm.md](docs/litellm.md) | Model resolution, credentials, retry/fallback, routing |
| [docs/skills.md](docs/skills.md) | Skills and the ContextManager integration |
| [docs/logging.md](docs/logging.md) | The logging layer |
