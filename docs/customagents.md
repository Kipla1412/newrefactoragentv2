# CustomAgents — config-driven agent system

`customagents/` is the **composition root** for the clinical agents. It is the
*only* layer that knows the concrete agent classes; everything else in the
framework talks to agents by their **type name**. It turns two levels of YAML
configuration into ready-to-run `Agent` instances, resolving each piece
through the right registry (model → `ModelRegistry`, prompt →
`PromptRegistry`, tools → the existing `ToolRegistry` — which is never
replaced).

## Two-level configuration

### 1. Application config — `agents.yaml` (project root)

Defines **what models exist** and **which agents are enabled**.

```yaml
models:
  clinical_model:         # used by most agents by default
    provider: openai
    name: gpt-4.1
    litellm_model: openai/gpt-4.1
    temperature: 0.1

  fast:
    provider: openai
    name: gpt-4o-mini
    litellm_model: openai/gpt-4o-mini

  reasoning:
    provider: anthropic
    name: claude-3-5-sonnet-20241022
    litellm_model: anthropic/claude-3-5-sonnet-20241022

  gemini:     # fallback
    ...
  ollama:     # fallback / local
    ...
  vision:     # GPT vision model, used for PDF OCR by config.vision_model_name
    provider: openai
    name: gpt-4o-mini
    litellm_model: openai/gpt-4o-mini

agents:
  doc:                 { enabled: true }
  soap:                { enabled: true }
  assessment:          { enabled: true }
  consult:             { enabled: true }
  intake:              { enabled: true }
  ehr:                 { enabled: true }
  clinical_extraction: { enabled: true }
  voice_intake:        { enabled: true }
  voice_consult:       { enabled: true }
  diarize:             { enabled: true }
  medical:             { enabled: true }
  document_retrieval:  { enabled: true }
  pdf_chat:            { enabled: true }
  # An agent can carry framework settings, e.g.
  # consult: { enabled: true, max_turns: 8, temperature: 0.2 }
```

`models:` are **logical definitions** — names that `ModelRegistry`
resolves to a full LiteLLM string. They are *not* fallbacks themselves;
only the names listed in `LLM_FALLBACK_MODELS` become the retry chain
(see **docs/litellm.md**).

### 2. Per-agent config — `customagents/<agent>/config.yaml`

The 13 agent folders each keep their own config. Only `name` is required;
`model`, `prompt`, and `tools` all have sensible defaults.

```yaml
name: medical          # logical agent type (matches agents:<name> above)
model: clinical_model  # a logical model defined under models:
prompt: medical        # a logical prompt registered in PromptRegistry
tools:                 # validated against the existing ToolRegistry
  - memory
  - wiki_search
  - neo4j_query
```

**Tool-key semantics** (added by design, so every agent is explicit):
- `tools: []` → the agent runs with **no tools**.
- `tools:` **omitted** → the agent is **unrestricted** (all tools).

Two agents intentionally omit `tools:` because they own their tool set at
runtime rather than from YAML:

| Agent | `config.yaml` | Why tools come from code |
|---|---|---|
| `documentagent` (`document_retrieval`) | no `tools` key | `DocumentRetrievalAgent` sets `allowed_tool_names = [jina_embedding, medical_document_search]` in its constructor |
| `ehragent` (`ehr`) | no `tools` key | the FHIR MCP tools are registered dynamically by the MCP Manager at runtime |

### Per-agent config inventory (13 agents)

| Folder | config file | `name` | `model` | `prompt` | `tools` |
|---|---|---|---|---|---|
| `consultagent` | `config.yaml` | `consult` | clinical_model | consult | memory |
| `diarizeagent` | `config.yaml` | `diarize` | clinical_model | diarize | memory |
| `doctorassistent` | `config.yaml` | `doc` | clinical_model | doc | memory |
| `documentagent` | `config.yaml` | `document_retrieval` | clinical_model | document_retrieval | _(omitted)_ |
| `ehragent` | `config.yaml` | `ehr` | clinical_model | ehr | _(omitted)_ |
| `medicalagent` | `config.yaml` | `medical` | clinical_model | medical | memory, wiki_search, neo4j_query |
| `pdfagent` | `config.yaml` | `pdf_chat` | clinical_model | pdf_chat | memory |
| `previsitagent` | `config.yaml` | `intake` | clinical_model | intake | memory |
| `reportagent` | `soap.config.yaml` | `soap` | clinical_model | soap | memory |
| `reportagent` | `assessment.config.yaml` | `assessment` | clinical_model | assessment | memory |
| `reportagent` | `clinical_extraction.config.yaml` | `clinical_extraction` | clinical_model | clinical_extraction | memory |
| `voiceagent` | `voice_intake.config.yaml` | `voice_intake` | clinical_model | voice_intake | memory |
| `voiceagent` | `voice_consult.config.yaml` | `voice_consult` | clinical_model | voice_consult | memory |

> The `reportagent` configs use the `*.config.yaml` suffix (no other
> configs live in that folder), so they are discovered by the same
> `name:`-keyed loader as the others.

## The `enabled` flag is enforced (not decorative)

`enabled` is checked in **two** places, so a false value truly disables an
agent rather than just hiding it:

1. **Route mounting** — `api/main.py` mounts each `/api/agent/<name>` router
   only when its required agents are enabled, and the
   `/consultation-workflow` composite only when `soap`, `assessment`, and
   `clinical_extraction` are *all* enabled. Disable `soap` and both
   `/api/agent/soap` and `/api/agent/consultation-workflow` disappear from
   the OpenAPI schema.
2. **Factory construction** — `AgentFactory.create(...)` raises:
   ```
   ConfigError: Agent '<name>' is disabled in agents.yaml
   ```
   before any tool/model/prompt work happens.

Only **enabled** agents' `config.yaml` files are even discovered at
bootstrap, so a disabled agent can't accidentally load.

## Tool resolution — the ToolRegistry is never replaced

The YAML `tools:` list is a **filter**, not an alternative registry:

1. `customagents/agent_config_loader.py` reads each `config.yaml`.
2. `AgentFactory.components().tools.resolve(...)` validates every name
   against the **existing** `tools/registry.py` (imports the real classes).
3. Unknown names raise a clear error instead of silently dropping them.
4. Tools are applied as `agent.allowed_tool_names` — the agent's own
   instance attribute — never by mutating the shared
   `session.config.allowed_tools`. This is actually stricter than the
   sibling `refactoragent` repo, where the factory mutated shared state.

`customagents/composition.py` also understands the **`mcp` pseudo-tool**:
the literal name `mcp` is expanded into whatever the MCP Manager registered
for that agent at runtime, rather than being treated as a tool class.

The ToolRegistry files (`tools/registry.py`, `tools/base.py`,
`tools/mcp/`) are **intentionally untouched** by this system.

## `AgentFactory` — the composition root

`customagents/factory.py` is a thin orchestrator. Its `create` walk mirrors
the dependency flow, one delegated step per concern:

```python
components = cls.components()
agent_class    = components.types.get(name)        # AgentTypeRegistry
agent_yaml, mr = components.configs.provide(name)  # 1. config + enabled check
agent_config   = components.models.resolve(...)    # 2. logical -> concrete
agent_config   = components.settings.apply(...)    #    + AgentSettings (max_turns, temperature)
prompt         = components.prompts.resolve(...)   # 3. logical prompt -> content
agent          = agent_class(agent_config, session=session)  # 4. construct
tools          = components.tools.resolve(...)     # 5. ToolRegistry
components.overrides.apply(agent, prompt, tools)  # 6. attach
```

### Public API

- `AgentFactory.components()` → the frozen `AgentComponents` bag holding all
  collaborators. Swap it wholesale for tests/custom registries with
  `AgentFactory.configure(custom_components)`.
- `AgentFactory.create(agent_type, config, session=None, resolved=None)`
  — backward-compatible signature; `resolved` is for pre-resolved overrides.
- `AgentFactory.known_agent_names()` — the set of types currently wired in.

### SOLID collaborators (`customagents/composition.py`)

`AgentComponents` groups six single-responsibility collaborators so the
factory never grows:

| Collaborator | Single responsibility |
|---|---|
| `AgentTypeRegistry` | name → agent class (and runtime registration) |
| `AgentConfigProvider` | load + validate per-agent `config.yaml`; enforce `enabled` |
| `ModelResolver` | logical model name → concrete `Config` via `ModelRegistry` |
| `PromptResolver` | logical prompt name → `PromptRegistry` content |
| `ToolResolver` | YAML list → real tools, incl. `mcp` expansion, via the ToolRegistry |
| `AgentSettingsApplier` | apply `max_turns` / `temperature` from `agents.yaml` to a copy |
| `AgentOverrideApplier` | attach prompt + tools to the constructed agent |

A frozen `AgentComponents` instance is built once and shared app-wide.

## Bootstrap — config is loaded once

`config/application_bootstrap.py` (`ApplicationBootstrap` +
`bootstrap_application()`) is the one place configuration is loaded and
registries are built, then published on `app.state` at startup:

| `app.state` key | Built where | Used by |
|---|---|---|
| `config` | `ApplicationBootstrap` | whole app |
| `agent_config` | `agent_config_loader` (dict name→`AgentYaml`) | router mounting |
| `model_registry` | `ModelRegistry` | `LLMClient`, `AgentFactory` |
| `prompt_registry` | `PromptRegistry.from_builtin()` | `PromptResolver` |
| `session_manager` | `SessionManager` | request handlers (`request.app.state.session_manager`) |

Because routes and the OpenAPI schema are generated from the live config,
the API surface literally changes when you flip an `enabled:` flag — no
restart of the agent code is needed beyond a server reload.

## How to add a new agent

1. Add `models:` / `agents:` entries in `agents.yaml` (and enable it).
2. Create `customagents/<agent>/config.yaml` with `name/model/prompt/tools`.
3. Drop the agent class + prompt in the folder.
4. Register the class:
   ```python
   from customagents.factory import AgentFactory
   AgentFactory.components().types.register("myname", MyAgent)
   ```
   (or add it to `DEFAULT_AGENT_CLASSES` in `factory.py`.)
5. Add the route in `api/main.py` via the `ROUTER_AGENT_REQUIREMENTS` table.

See **docs/litellm.md** for how the model side of all this is served.
