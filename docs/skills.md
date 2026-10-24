# Skills

A **skill** is a reusable block of instructions the model can be given for a
single request. Skills are prompt content — never executable code — and they
live in their own subsystem, separate from agents, tools, and prompts.

```
skills/
├── __init__.py          # public exports
├── errors.py            # SkillError / SkillValidationError / SkillNotFoundError
├── models.py            # SkillMetadata, Skill (frozen dataclasses)
├── frontmatter.py       # SKILL.md parsing + metadata validation
├── registry.py          # discovery + on-demand loading
└── library/
    ├── clinical-followup-questions/
    │   └── SKILL.md     # bundled skill — text follow-up questions
    └── ehr-fhir-retrieval/
        └── SKILL.md     # bundled skill — FHIR/MCP data retrieval
```

## The `SKILL.md` format

YAML frontmatter delimited by `---`, then the instruction body:

```markdown
---
name: clinical-followup-questions
description: Generate exactly three clinically relevant follow-up questions…
---

You are a clinical decision support assistant for doctors.
…
```

Rules enforced by `skills/frontmatter.py`:

| Rule | Failure |
|---|---|
| File starts with a `---` line | `SkillValidationError` (missing frontmatter) |
| The opening block is closed by `---` | `SkillValidationError` (unclosed) |
| Frontmatter parses as YAML | `SkillValidationError` (invalid YAML) |
| It is a mapping | `SkillValidationError` |
| `name` is a non-empty string | `SkillValidationError` |
| `description` is a non-empty string | `SkillValidationError` |
| No two skills share a `name` | `SkillValidationError` |

Only the **first** closing `---` ends the frontmatter, so `---` separators
inside the instruction body are preserved (the bundled example uses them).
Blank lines before the opening `---` are tolerated.

## Discovery vs. loading

`SkillRegistry` keeps the two phases apart:

| Phase | Method | Reads |
|---|---|---|
| Discovery | `discover()` / `list_skills()` / `get_metadata(name)` | frontmatter only (cached) |
| Loading | `load(name)` / `load_instructions(name)` | frontmatter **and** body, fresh from disk |

- A **missing root** is not an error — it yields no skills.
- A directory **without** `SKILL.md` is simply not a skill (ignored).
- A **malformed** `SKILL.md` raises rather than being skipped, so a broken
  skill cannot be silently dropped.
- `refresh()` clears the discovery cache after adding or editing skills.
- `load()` re-reads the file, so editing instructions needs no restart.
- Unknown names raise `SkillNotFoundError` listing what *is* available.

`default_skill_registry()` returns a process-wide registry over the bundled
`skills/library/` directory.

## Skill lifecycle

```
User Request
  → Custom Agent            agent.run(message, skill="…")
  → Agent.active_skill()    caller's name, else the agent's configured skill
  → Agent.use_skill()       reusable seam on the base Agent
  → SkillRegistry.load()    reads SKILL.md, validates metadata, returns Skill
  → ContextManager          use_skill(skill.instructions)
  → ContextManager.get_messages()   system message = prompt + "# Active Skill"
  → Agent._agentic_loop()   unchanged — the normal loop runs
  → LLMClient → LiteLLM     receives the skill instructions
  → use_skill exit          previous state restored (try/finally)
```

Skill activation never introduces a second loop, a second context, or a second
prompt path: it is one extra section on the **existing** system message.

## How an agent knows which skill to use

An agent declares its skill in its own `config.yaml`, next to `model`, `prompt`,
and `tools`:

```yaml
# customagents/ehragent/config.yaml
name: ehr
model: clinical_model
prompt: ehr
skill: ehr-fhir-retrieval
```

`AgentFactory` resolves `skill:` through the `SkillRegistry` and applies it to
the agent as `agent.default_skill`. `Agent.active_skill()` then decides per
call:

| Caller passes | Result |
|---|---|
| nothing (`None`) | the agent's configured skill |
| a name (`"other"`) | that skill — the caller overrides the default |
| `""` | **no skill** — an explicit opt-out for one call |
| nothing, and no configured skill | no skill (previous behaviour) |

So the request body no longer has to name a skill: `POST /api/agent/mcpagent`
with just `{"message": "…"}` activates `ehr-fhir-retrieval` because the EHR
agent declares it. Passing `"skill": "…"` still overrides, and `"skill": ""`
disables it for that request.

**Validated at startup.** An agent naming a skill that does not exist fails
`AgentConfigLoader.load()` with
`Agent '<name>' references unknown skill '<skill>'` — the same fail-early
treatment as unknown models and prompts, so a typo never becomes a silent
no-op at runtime.

**Why this over LLM-based selection?** It is deterministic, costs no extra
call, keeps one source of truth alongside the agent's other registry
references, and leaves room to add model-driven selection later on top.

## ContextManager integration

Four small additions to `context/manager.py`:

| Member | Purpose |
|---|---|
| `set_skill_instructions(text)` | activate, replace, or clear (`None`/blank clears) |
| `clear_skill_instructions()` | remove the active skill only |
| `skill_instructions` (property) | inspect what is active |
| `use_skill(text)` | scoped activation — restores the previous state on exit |
| `build_system_prompt()` | the system content, prompt + skill section |

`get_messages()` now calls `build_system_prompt()`, which appends the skill
section to the configured system prompt:

```
<system prompt>

# Active Skill

<skill instructions>
```

**Backward compatibility.** With no skill active, `build_system_prompt()`
returns exactly `self._system_prompt`: same message count, same ordering, same
single leading system turn, and no system message at all when no prompt is set.
Conversation history, tool messages, pruning, compaction, and token accounting
are untouched.

**Why the system message and not a separate one?** Appending keeps one system
turn at the head of the conversation, leaving history ordering intact, and the
instructions appear in exactly one place.

## Reusable seam on the base `Agent`

Skill activation lives on **`Agent`**, not on individual agents, so every agent
gets it the same way:

```python
@contextmanager
def use_skill(self, skill: str | None) -> Iterator[Skill | None]:
    if not skill:
        yield None
        return
    if not self.session.context_manager:
        raise AgentError("Session must be initialized before activating a skill …")
    loaded = self.skill_registry.load(skill)                 # skills subsystem
    with self.session.context_manager.use_skill(loaded.instructions):
        yield loaded                                         # scoped to this call
```

- `self.skill_registry` is set in `Agent.__init__` and defaults to
  `default_skill_registry()`; tests inject a temp directory. Because it is an
  optional constructor kwarg, `AgentFactory`'s
  `agent_class(agent_config, session=session)` is unchanged.
- An agent opts in by wrapping part of its execution:
  `with self.use_skill(self.active_skill(skill)): async for event in self._agentic_loop(): …`
  — `active_skill` applies the configured default (see above).
- Uninitialized sessions fail loudly rather than silently dropping the skill.

`EHRAgent` is wired this way (`run(message, skill=None)`); `DoctorAssistantAgent`
uses the same inherited pattern from `recommend_questions(conversation, skill=…)`
and `run(message, skill=…)`. Neither agent carries its own copy of the logic.

## Scoping and isolation

The `ContextManager` belongs to a `Session`, and sessions are shared per user,
so skill state must not outlive the request. Both `Agent.use_skill()` and
`ContextManager.use_skill()` are `@contextmanager`s that restore the previous
value in a `finally` block, so instructions are removed even when the call
raises. Tests cover the block exiting normally, on exception, restoring an
outer skill, two contexts staying independent, and — at the agent level — that
a follow-up request without a skill does not see the previous one's
instructions.

## Safety

- Skill contents are **prompt text**. Nothing imports, evaluates, or executes
  a skill file.
- Loading a skill grants **no tool permissions**. A skill that mentions a tool
  does not enable it; tools still come only from the existing `ToolRegistry`.
  Tested: neither the tool registry nor `allowed_tools` changes.
- Required metadata is validated at discovery time, so a broken skill fails
  loudly.
- Model output for structured skills is validated in application code — see
  `DoctorAssistantAgent.validate_questions()`, which enforces exactly three
  non-empty questions and rejects anything else.

## Example: `EHRAgent`

`customagents/ehragent/config.yaml` declares the skill:

```yaml
name: ehr
model: clinical_model
prompt: ehr
skill: ehr-fhir-retrieval
```

`customagents/ehragent/ehragent.py` activates it:

```python
agent = AgentFactory.create(AgentType.EHR, config, session)
assert agent.default_skill == "ehr-fhir-retrieval"   # applied from config

# The agent knows its skill — no argument needed.
async for event in agent.run("Show medications for patient 12345"):
    ...

# A caller can still override for one request…
async for event in agent.run("Show medications for patient 12345",
                             skill="clinical-followup-questions"):
    ...

# …or opt out entirely for one request.
async for event in agent.run("Show medications for patient 12345", skill=""):
    ...

# Tests inject their own registry:
agent = EHRAgent(config, session, skill_registry=SkillRegistry(tmp_path))
```

Over HTTP, `POST /api/agent/mcpagent` needs nothing extra:

```json
{ "message": "Show medications for patient 12345", "session_id": "session-123" }
```

`skill` is optional and, when supplied, overrides the configured default;
`"skill": ""` disables it for that request.

### The other consumer: `DoctorAssistantAgent`

```python
agent = AgentFactory.create(AgentType.DOC, config, session)
assert agent.default_skill == "clinical-followup-questions"

await agent.recommend_questions(conversation)                       # uses its skill
await agent.recommend_questions(conversation, skill="")             # no skill
await agent.recommend_questions(conversation, skill="other")        # override
```

Over HTTP, `POST /api/agent/doctoragent` accepts `skill`. `DoctorAgentRequest`
extends the shared `ConversationRequest`, so the SOAP endpoint's contract is
unchanged.

## Bundled skills

| Skill | Declared by | Source of instructions | Preserves |
|---|---|---|---|
| `clinical-followup-questions` | `doc` | `DOC_PROMPT` | exactly three clinically relevant follow-up questions, no repetition of given or previously asked information, valid JSON only, the `{"questions": [...]}` schema, no diagnosis/treatment/explanation/extra text |
| `ehr-fhir-retrieval` | `ehr` | `EHR_AGENT_PROMPT` | MCP/FHIR tool usage, grounding every statement in tool output, never fabricating or hallucinating patient data or lab values, concise structured summaries, no diagnosis or intake workflows |
| `patient-demographics-summary` | `demographics` | written for the skill (the agent's prompt is a minimal role statement) | the Patient/RelatedPerson/Coverage field workflow, a mandatory "missing or unavailable" section, no inventing or inferring data, no assuming an emergency contact or active coverage, identity verification before summarising |

The first two mirror the prompt the agent already uses, so those agents'
behaviour is unchanged apart from the instructions now arriving as an explicitly
activated `# Active Skill` section. `patient-demographics-summary` is new
content: its agent's prompt is deliberately a minimal role statement, and the
workflow lives in the skill so it can be revised without touching code.

Any other agent can opt in the same way — add `skill:` to its
`customagents/<agent>/config.yaml`. Agents without the key behave exactly as
before: no skill unless the caller names one.

## Testing

`test/test_skills.py` (75 tests):

| Area | Coverage |
|---|---|
| Discovery / loading | both bundled skills found; bodies preserve their requirements; sorted listing; missing root; non-skill directories ignored; caching and `refresh()` |
| Valid / invalid files | 9 malformed `SKILL.md` cases; frontmatter splitting; blank leading lines |
| ContextManager | activation, replacement, blank/`None` clearing, clearing restores messages exactly |
| LLM messages | skill text present in the system message; history intact and ordered behind it |
| Backward compatibility | no skill → messages byte-identical to the old behaviour; no prompt and no skill → no system message; `clear()` still preserves the prompt |
| Isolation | `use_skill` scoping, exception safety, nested restore, two contexts independent |
| Reusable seam | `Agent.skill_registry` defaulted; `Agent.use_skill` activates any registered skill; `None` is a no-op; uninitialized session raises |
| Default skill resolution | `active_skill`: configured default, caller override, `""` opt-out; schema accepts an optional `skill`; loader rejects an unknown skill and accepts a valid one; the override applier sets `default_skill`; shipped `doc`/`ehr` configs declare theirs |
| End-to-end (EHR) | `run(skill=…)` and config-driven `run()` — captured LLM messages contain the skill text and keep `Healthcare MCP Agent`; the bundled `ehr-fhir-retrieval` resolves by name; no skill → no section; follow-up call clean; unknown skill → `agent_error` and no LLM call |
| End-to-end (doc) | `recommend_questions(...)` and streaming `run(...)`, with and without a skill — same checks through the inherited seam; `run()` sets the system prompt (regression) |
| Output validation | exactly three questions; wrong count, non-list, non-string, blank; markdown fences; invalid JSON |
| Safety | loading a skill changes neither the tool registry nor `allowed_tools` |
