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
  → Agent.use_skill(skill)  reusable seam on the base Agent
  → SkillRegistry.load()    reads SKILL.md, validates metadata, returns Skill
  → ContextManager          use_skill(skill.instructions)
  → ContextManager.get_messages()   system message = prompt + "# Active Skill"
  → Agent._agentic_loop()   unchanged — the normal loop runs
  → LLMClient → LiteLLM     receives the skill instructions
  → use_skill exit          previous state restored (try/finally)
```

Skill activation never introduces a second loop, a second context, or a second
prompt path: it is one extra section on the **existing** system message.

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
  `with self.use_skill(skill): async for event in self._agentic_loop(): …`
- Uninitialized sessions fail loudly rather than silently dropping the skill.

`EHRAgent` is wired this way (`run(message, skill=None)`); `DoctorAssistantAgent`
uses the same inherited block from `recommend_questions(conversation, skill=…)`.
Neither agent carries its own copy of the logic.

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

`customagents/ehragent/ehragent.py` — the first agent on the reusable seam:

```python
agent = AgentFactory.create(AgentType.EHR, config, session)

# Without a skill — unchanged behaviour.
async for event in agent.run("Show medications for patient 12345"):
    ...

# With a skill by name — instructions reach the LLM for this request only.
async for event in agent.run("Show medications for patient 12345",
                             skill="ehr-fhir-retrieval"):
    ...

# The optional tests use the same seam with a temp registry:
agent = EHRAgent(config, session, skill_registry=SkillRegistry(tmp_path))
```

Over HTTP, `POST /api/agent/mcpagent` accepts the optional field:

```json
{ "message": "Show medications for patient 12345",
  "session_id": "session-123",
  "skill": "ehr-fhir-retrieval" }
```

### The other consumer: `DoctorAssistantAgent`

```python
agent = AgentFactory.create(AgentType.DOC, config, session)

await agent.recommend_questions(conversation)                       # unchanged
await agent.recommend_questions(conversation,
                                skill="clinical-followup-questions")
```

Over HTTP, `POST /api/agent/doctoragent` accepts `skill`. `DoctorAgentRequest`
extends the shared `ConversationRequest`, so the SOAP endpoint's contract is
unchanged.

## Bundled skills

| Skill | Source of instructions | Preserves |
|---|---|---|
| `clinical-followup-questions` | `DOC_PROMPT` | exactly three clinically relevant follow-up questions, no repetition of given or previously asked information, valid JSON only, the `{"questions": [...]}` schema, no diagnosis/treatment/explanation/extra text |
| `ehr-fhir-retrieval` | `EHR_AGENT_PROMPT` | MCP/FHIR tool usage, grounding every statement in tool output, never fabricating or hallucinating patient data or lab values, concise structured summaries, no diagnosis or intake workflows |

Both are copies of the prompt the agent already uses, so the agents' behaviour
is unchanged when no skill is requested.

## Testing

`test/test_skills.py` (63 tests):

| Area | Coverage |
|---|---|
| Discovery / loading | both bundled skills found; bodies preserve their requirements; sorted listing; missing root; non-skill directories ignored; caching and `refresh()` |
| Valid / invalid files | 9 malformed `SKILL.md` cases; frontmatter splitting; blank leading lines |
| ContextManager | activation, replacement, blank/`None` clearing, clearing restores messages exactly |
| LLM messages | skill text present in the system message; history intact and ordered behind it |
| Backward compatibility | no skill → messages byte-identical to the old behaviour; no prompt and no skill → no system message; `clear()` still preserves the prompt |
| Isolation | `use_skill` scoping, exception safety, nested restore, two contexts independent |
| Reusable seam | `Agent.skill_registry` defaulted; `Agent.use_skill` activates any registered skill; `None` is a no-op; uninitialized session raises |
| End-to-end (EHR) | `run(skill=…)` — captured LLM messages contain the skill text and keep `Healthcare MCP Agent`; the bundled `ehr-fhir-retrieval` resolves by name; no skill → no section; follow-up call clean; unknown skill → `agent_error` and no LLM call |
| End-to-end (doc) | `recommend_questions(skill=…)` and streaming `run(skill=…)` — same checks through the inherited seam; `run()` sets the system prompt (regression) |
| Output validation | exactly three questions; wrong count, non-list, non-string, blank; markdown fences; invalid JSON |
| Safety | loading a skill changes neither the tool registry nor `allowed_tools` |
