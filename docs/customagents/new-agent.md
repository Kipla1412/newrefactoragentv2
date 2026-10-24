# Building a new agent

A step-by-step checklist for adding an agent to this framework. Follow it in
order — every step is either code, configuration, or registration, and skipping
one produces a specific startup error (listed in **Failure modes** below).

If the agent has no behaviour beyond prompt + model + tools + skill, read
**Fastest path** first: it needs no new Python at all.

Background on how agents are wired: **[../customagents.md](../customagents.md)**.

## The one string that must match

An agent's identity is a single lowercase string that has to agree in **four**
places:

```
"my_agent" ─┬─ AgentType.MY_AGENT = "my_agent"      agent/events.py
            ├─ agents.yaml:  my_agent: { enabled: true }
            ├─ config.yaml:  name: my_agent        customagents/myagent/config.yaml
            └─ DEFAULT_AGENT_CLASSES["my_agent"]   customagents/factory.py
```

`AgentFactory.create(AgentType.MY_AGENT, …)` lowercases the enum and looks that
string up in the type registry, so any mismatch fails with
`ValueError: Unknown agent type: …`.

## Fastest path: a config-only agent

If the agent is just "a prompt, a model, some tools, maybe a skill" — no custom
control flow — you need **no new Python at all**. Any number of agents share one
registered class and each declares it with `agent_class`:

```yaml
# customagents/triage_nurse/config.yaml
name: triage_nurse
agent_class: generic                    # the shared class (see GenericAgent)
model: clinical_model
prompt: triage                          # still needs a PromptRegistry entry
skill: triage-questions
agent_name: TriageNurseAgent            # MLflow tag / error prefix
tools:
  - memory
```

```yaml
# agents.yaml
agents:
  triage_nurse: { enabled: true }
```

That skips steps **1, 3, and 6** — the `AgentType` member, the agent class, and
the class registration already exist (`AgentType.GENERIC` and `"generic"`).
Step 7 still applies if you set `prompt:`, and steps 5, 8, 9 are unchanged.

Because the name is not registered as a class, the loader accepts it when
`agent_class` names a registered one — and rejects it (with a clear message) when
it does not:

| Config | Result |
|---|---|
| `agent_class: generic` | accepted; `GenericAgent` builds it |
| no `agent_class` | `Agent '<name>' is enabled but not registered in AgentFactory, and its config.yaml sets no 'agent_class'` |
| `agent_class: not_a_class` | `Agent '<name>' references unknown agent_class 'not_a_class'` |

Use the full path below when the agent needs behaviour of its own: a bespoke
`run()`, a forced tool choice, output parsing, audio, or TTS. `GenericAgent`
deliberately holds no prompt text and no domain logic.

Implementation: `customagents/genericagent/genericagent.py`.

## Checklist

| # | Where | What | Required? |
|---|---|---|---|
| 1 | `agent/events.py` | Add an `AgentType` member | yes |
| 2 | `customagents/<agent>/<prompt>.py` | The prompt constant | yes |
| 3 | `customagents/<agent>/<agent>.py` | The agent class | yes |
| 4 | `customagents/<agent>/config.yaml` | `name` / `model` / `prompt` / `tools` | for config-driven use |
| 5 | `agents.yaml` | Enable it, and add a `models:` entry if needed | for config-driven use |
| 6 | `customagents/factory.py` | Register the class in `DEFAULT_AGENT_CLASSES` | yes |
| 7 | `registries/prompt_registry.py` | Register the prompt by logical name | if `prompt:` is set |
| 8 | `api/routers/<router>.py` + `api/main.py` | Expose an HTTP endpoint | optional |
| 9 | `test/test_<agent>.py` | Tests | strongly recommended |

**"For config-driven use"** means: the agent gets its model/prompt/skill/tools
from `config.yaml`, is validated at startup, and can be gated with
`enabled:`. An agent that is neither in `agents.yaml` nor in a `config.yaml`
still constructs through the factory, but it runs on the prompt and defaults
its class hard-codes, is never validated, and cannot be disabled — so list it
unless that is deliberate.

## Step by step

### 1. The agent type

`agent/events.py`:

```python
class AgentType(str, Enum):
    DOC = "doc"
    ...
    MY_AGENT = "my_agent"      # ← add
```

### 2. The prompt

`customagents/myagent/myprompt.py`:

```python
MY_PROMPT = """
You are a <role>.
<instructions>
"""
```

Keep it a plain module-level constant. `PromptRegistry` imports it by name —
the prompt content is never duplicated into the registry.

### 3. The agent class

`customagents/myagent/myagent.py`:

```python
from __future__ import annotations

from typing import AsyncGenerator

from agent.agent import Agent
from agent.events import AgentEvent, AgentType
from prompts.system import get_system_prompt

from .myprompt import MY_PROMPT


class MyAgent(Agent):
    """One-line description of what this agent does."""

    def __init__(self, config, session=None, skill_registry=None):
        super().__init__(
            config,
            MY_PROMPT,
            AgentType.MY_AGENT,
            skill_registry=skill_registry,
        )
        if session:
            self.session = session

    async def run(
        self, message: str = "", skill: str | None = None
    ) -> AsyncGenerator[AgentEvent, None]:
        """Stream the agent's response as AgentEvents.

        Args:
            message: The user's request.
            skill: Optional skill name; falls back to this agent's
                configured default.

        Yields:
            AgentEvent objects produced by the agentic loop.
        """
        if not self.session.context_manager:
            await self.session.initialize()

        self.session.agent_name = "MyAgent"

        self.session.context_manager.set_system_prompt(
            get_system_prompt(
                config=self.config,
                role_prompt=self.system_prompt,
                user_memory=getattr(self, "memory", None),
            )
        )

        if message:
            self.session.context_manager.add_user_message(message)

        self.session.start_mlflow_run(message)

        try:
            with self.use_skill(self.active_skill(skill)):
                async for event in self._agentic_loop():
                    if hasattr(event, "data") and event.data.get("agent") is None:
                        event.data["agent"] = self.agent_type
                    yield event
        except Exception as e:
            yield AgentEvent.agent_error(error=f"MyAgent error: {e}")
        finally:
            self.session.end_mlflow_run()

    async def __aenter__(self):
        if not self.session.context_manager:
            await self.session.initialize()
        return self
```

Non-negotiable conventions, all visible above:

- **`__init__(self, config, session=None, skill_registry=None)`** — the factory
  calls `agent_class(agent_config, session=session)`, so every extra parameter
  after `session` must have a default.
- **Never build your own `Session`.** `Agent.__init__` creates one; assigning
  `self.session = session` reuses the caller's shared session (history, tools,
  MCP connections).
- **Always set the system prompt** before the loop. Skipping it means no system
  message reaches the model at all.
- **Always `start_mlflow_run` / `end_mlflow_run`** around the loop, in a
  `finally`.
- **Wrap the loop in `self.use_skill(self.active_skill(skill))`** if the agent
  should support skills. `active_skill` applies the configured default; `None`
  means no skill.
- **Reuse `_agentic_loop()`** — the loop handles tool calls, approvals, retries,
  compaction, and pruning. Do not write a second loop.

#### Two response shapes

| Style | Method | Used by | HTTP result |
|---|---|---|---|
| Streaming | `run(...)` yielding `AgentEvent`s | most agents | NDJSON stream |
| Buffered | `generate(...)` / `recommend_questions(...)` draining the loop internally and returning a dict | report agents, doc agent | single JSON body |

Pick per agent; the doc agent implements **both** from the same prompt.

If the agent must run with a fixed tool set that YAML cannot express, set it in
the constructor and leave `tools:` out of `config.yaml` — see
`DocumentRetrievalAgent` (`force_tool_choice = True`,
`allowed_tool_names = [...]`) and `EHRAgent` (MCP tools registered at runtime).

### 4. The agent config

`customagents/myagent/config.yaml`:

```yaml
name: my_agent
model: clinical_model        # logical model name from agents.yaml
prompt: my_agent             # logical prompt name in PromptRegistry
# skill: my-skill            # optional — see ../skills.md
tools:                       # validated against the existing ToolRegistry
  - memory
```

Tool-key semantics: `tools: []` means **no tools**; omitting the key leaves the
agent **unrestricted**.

### 5. Enable it

`agents.yaml` (project root):

```yaml
agents:
  my_agent: { enabled: true }
```

Add a `models:` entry only if you need a model that does not exist yet:

```yaml
models:
  my_model:
    provider: openai
    name: gpt-4.1
    litellm_model: openai/gpt-4.1
    temperature: 0.1
```

### 6. Register the class

`customagents/factory.py`:

```python
from customagents.myagent.myagent import MyAgent

DEFAULT_AGENT_CLASSES: dict[str, type] = {
    ...
    "my_agent": MyAgent,
}
```

This is the single place concretions are referenced. It also feeds
`AgentFactory.known_agent_names()`, which is what startup validation checks
`agents.yaml` against.

For a plugin-style agent you can alternatively register at runtime instead of
editing the mapping:

```python
from customagents.factory import AgentFactory
AgentFactory.components().types.register("my_agent", MyAgent)
```

### 7. Register the prompt

`registries/prompt_registry.py` → `from_builtin()`:

```python
from customagents.myagent.myprompt import MY_PROMPT

return cls({
    ...
    "my_agent": MY_PROMPT,
})
```

Required whenever `config.yaml` sets `prompt:` — the logical name must resolve.

### 8. Expose an HTTP endpoint (optional)

`api/routers/myagent.py`, following an existing router such as `api/routers/ehr.py`:

```python
import json

from fastapi import APIRouter, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.events import AgentType
from api.auth import require_permission
from customagents.factory import AgentFactory

router = APIRouter(prefix="/agent")


class MyAgentRequest(BaseModel):
    message: str
    skill: str | None = None


@router.post(
    "/myagent",
    summary="My agent",
    dependencies=[Depends(require_permission("consultagent", "chat"))],
)
async def myagent_api(request: Request, data: MyAgentRequest):
    config = request.app.state.config
    session = await request.app.state.session_manager.get_session(
        request.state.user["sub"], config
    )
    agent = AgentFactory.create(AgentType.MY_AGENT, config, session)

    async def event_stream():
        async for event in agent.run(data.message, skill=data.skill):
            yield json.dumps(jsonable_encoder(event)) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
```

Use a buffered method instead of `StreamingResponse` if the agent returns a
single JSON body. Either way, take the session from
`request.app.state.session_manager` — never construct one per request.

Then register it in `api/main.py`:

```python
ROUTER_AGENT_REQUIREMENTS: list[tuple[object, tuple[str, ...]]] = [
    ...,
    (myagent_router, ("my_agent",)),
]
```

Mounting, the OpenAPI schema, and the `enabled` gate are all driven off that
table — there is nothing else to wire. A router listed with multiple names (for
example `("soap", "assessment", "clinical_extraction")`) mounts only when **all**
of them are enabled.

### 9. Tests

`test/test_<agent>.py`. The suite's conventions:

- `test/conftest.py` provides a `config` fixture (`Config(cwd=Path.cwd())`).
- Stub the LLM by replacing `session.client.chat_completion` with an async
  generator yielding `StreamEvent(type=StreamEventType.TEXT_DELTA, …)` then a
  `MESSAGE_COMPLETE`. See `test/test_skills.py` for the pattern.
- Assert routes through the OpenAPI schema, not `app.routes` (the included
  routers are lazy, so `app.routes` is empty).
- Enabled state changes the API surface, so a gating test belongs in
  `test/test_agent_gating.py`.

## Failure modes

Startup validation reports missing references one at a time — enable the agent
first and let the error tell you what is left.

| Symptom | Cause |
|---|---|
| `ValueError: Unknown agent type: my_agent` | missing from `DEFAULT_AGENT_CLASSES` (step 6) or a name mismatch |
| `Agent 'my_agent' is enabled but has no config.yaml` | step 4 |
| `Agent 'my_agent' is enabled but not registered in AgentFactory` | step 6, or a name mismatch |
| `Agent 'my_agent' references unknown model 'x'` | model missing from `agents.yaml` `models:` |
| `Agent 'my_agent' references unknown prompt 'x'` | step 7, or a name mismatch |
| `Agent 'my_agent' references unknown skill 'x'` | skill missing from `skills/library/` |
| `Agent 'my_agent' is disabled in agents.yaml` | `enabled: false` — that is the gate working |
| `ConfigError: Invalid agent config in …` | a `config.yaml` field is misspelled; pydantic rejects extras |
| Model replies with no system prompt | the entry point never called `set_system_prompt` |
| Tools the agent should not have are callable | `tools:` not set, and the constructor does not restrict `allowed_tool_names` |

## Rules

- **Do not** create a second `SessionManager`, `ContextManager`, `AgentFactory`,
  `ToolRegistry`, or LLM client. Every agent shares them.
- **Do not** create a second agentic loop — use `_agentic_loop()`.
- **Do not** put model, prompt, or tool *content* in `config.yaml`. It holds
  logical references only; the content lives in its registry.
- **Do not** let an agent call LiteLLM or a provider directly — go through
  `session.client` (see **[../litellm.md](../litellm.md)**).
- **Do not** enable a new agent in a workflow composite unless it belongs there;
  check `ROUTER_AGENT_REQUIREMENTS` in `api/main.py`.
- **Keep** the constructor signature compatible, or `AgentFactory` breaks for
  every deployment.
