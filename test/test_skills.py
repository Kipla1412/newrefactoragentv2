"""Tests for the skills subsystem and its ContextManager integration.

Coverage:

- discovery and on-demand loading
- valid and invalid ``SKILL.md`` files
- activation / replacement / clearing on the ContextManager
- skill instructions actually appearing in the messages sent to the LLM
- preservation of the system prompt, history, and ordering
- isolation between contexts and between requests
- backward compatibility when no skill is used
- clinical follow-up output validation
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from agent.events import AgentEventType
from client.response import StreamEvent, StreamEventType, TextDelta, TokenUsage
from config.config import Config
from context.manager import ContextManager
from skills import (
    SKILL_FILE_NAME,
    SkillNotFoundError,
    SkillRegistry,
    SkillValidationError,
    default_skill_registry,
)
from skills.frontmatter import split_frontmatter

EXAMPLE_SKILL = "clinical-followup-questions"
EHR_SKILL = "ehr-fhir-retrieval"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_SKILL = """---
name: sample
description: A sample skill.
---

Do the sample thing.
"""


def write_skill(root: Path, dirname: str, content: str) -> Path:
    """Create ``<root>/<dirname>/SKILL.md`` and return its path."""
    skill_dir = root / dirname
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / SKILL_FILE_NAME
    path.write_text(content, encoding="utf-8")
    return path


def make_context(config: Config) -> ContextManager:
    """Build a ContextManager with nothing but configuration attached."""
    return ContextManager(config, user_memory=None, tools=[])


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------


def test_discovers_bundled_example_skill():
    registry = default_skill_registry()

    assert registry.has(EXAMPLE_SKILL)

    metadata = registry.get_metadata(EXAMPLE_SKILL)
    assert metadata.name == EXAMPLE_SKILL
    assert metadata.description.strip()
    assert metadata.path.name == SKILL_FILE_NAME
    assert metadata.path.is_file()


def test_load_returns_instructions_preserving_clinical_requirements():
    skill = default_skill_registry().load(EXAMPLE_SKILL)

    assert skill.name == EXAMPLE_SKILL
    instructions = skill.instructions
    assert "Generate EXACTLY 3 follow-up questions" in instructions
    assert "Do NOT repeat information already present in the conversation" in instructions
    assert "Do NOT repeat previously asked questions" in instructions
    assert "Output ONLY valid JSON" in instructions
    assert "Do NOT provide diagnosis, treatment, explanation, or reasoning" in instructions
    assert '"questions"' in instructions
    # Instruction-body separators survive frontmatter splitting.
    assert "\n---\n" in instructions


def test_list_skills_is_sorted_by_name():
    names = [metadata.name for metadata in default_skill_registry().list_skills()]

    assert names == sorted(names)
    assert EXAMPLE_SKILL in names
    assert EHR_SKILL in names


def test_discovers_bundled_ehr_skill_preserving_requirements():
    registry = default_skill_registry()

    assert registry.has(EHR_SKILL)

    metadata = registry.get_metadata(EHR_SKILL)
    assert metadata.description.strip()

    instructions = registry.load_instructions(EHR_SKILL)
    assert "Healthcare MCP Agent" in instructions
    assert "Never fabricate patient information" in instructions
    assert "Never hallucinate medical records" in instructions
    assert "Never generate fake lab values" in instructions
    assert "Always ground responses in actual tool outputs" in instructions


def test_load_instructions_matches_load():
    registry = default_skill_registry()

    assert registry.load_instructions(EXAMPLE_SKILL) == registry.load(EXAMPLE_SKILL).instructions


def test_skill_metadata_and_skill_are_json_serializable():
    skill = default_skill_registry().load(EXAMPLE_SKILL)

    assert skill.metadata.to_dict() == {
        "name": EXAMPLE_SKILL,
        "description": skill.description,
    }
    assert skill.to_dict()["instructions"] == skill.instructions


def test_missing_root_yields_no_skills(tmp_path):
    registry = SkillRegistry(tmp_path / "does-not-exist")

    assert registry.discover() == {}
    assert registry.list_skills() == []
    assert registry.has(EXAMPLE_SKILL) is False


def test_directory_without_skill_file_is_ignored(tmp_path):
    (tmp_path / "not-a-skill").mkdir()
    (tmp_path / "not-a-skill" / "notes.txt").write_text("nope", encoding="utf-8")
    write_skill(tmp_path, "real", VALID_SKILL)

    assert list(SkillRegistry(tmp_path).discover()) == ["sample"]


def test_unknown_skill_raises_and_lists_available(tmp_path):
    write_skill(tmp_path, "real", VALID_SKILL)

    with pytest.raises(SkillNotFoundError) as exc:
        SkillRegistry(tmp_path).load("missing-skill")

    assert "missing-skill" in str(exc.value)
    assert "sample" in str(exc.value)


def test_load_after_file_deleted_raises_not_found(tmp_path):
    registry = SkillRegistry(tmp_path)
    path = write_skill(tmp_path, "sample", VALID_SKILL)

    assert registry.load("sample").instructions == "Do the sample thing."

    path.unlink()

    with pytest.raises(SkillNotFoundError):
        registry.load("sample")


def test_discovery_is_cached_until_refreshed(tmp_path):
    registry = SkillRegistry(tmp_path)
    assert registry.discover() == {}

    write_skill(tmp_path, "sample", VALID_SKILL)

    assert registry.discover() == {}          # cached
    assert list(registry.refresh()) == ["sample"]  # forced re-scan


def test_duplicate_skill_names_are_rejected(tmp_path):
    write_skill(tmp_path, "one", VALID_SKILL)
    write_skill(tmp_path, "two", VALID_SKILL)

    with pytest.raises(SkillValidationError) as exc:
        SkillRegistry(tmp_path).discover()

    assert "Duplicate" in str(exc.value)


# ---------------------------------------------------------------------------
# Malformed SKILL.md files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("no frontmatter at all", id="missing-frontmatter"),
        pytest.param("---\nname: x\ndescription: d\n", id="unclosed-frontmatter"),
        pytest.param("---\ndescription: only a description\n---\nbody", id="missing-name"),
        pytest.param("---\nname: x\n---\nbody", id="missing-description"),
        pytest.param("---\nname: ''\ndescription: d\n---\nbody", id="empty-name"),
        pytest.param("---\nname: x\ndescription: '   '\n---\nbody", id="blank-description"),
        pytest.param("---\nname: [1, 2]\ndescription: d\n---\nbody", id="non-string-name"),
        pytest.param("---\nname: x\ndescription: d\n\tbad: [unclosed\n---\nbody", id="bad-yaml"),
        pytest.param("---\n- just\n- a\n- list\n---\nbody", id="frontmatter-not-a-mapping"),
    ],
)
def test_invalid_skill_files_raise_validation_error(tmp_path, content):
    write_skill(tmp_path, "broken", content)

    with pytest.raises(SkillValidationError):
        SkillRegistry(tmp_path).discover()


def test_frontmatter_split_preserves_body_separators():
    text = "---\nname: n\ndescription: d\n---\n\nA\n\n---\n\nB\n"

    frontmatter, body = split_frontmatter(text)

    assert frontmatter.strip() == "name: n\ndescription: d"
    assert body == "A\n\n---\n\nB"


def test_blank_lines_before_frontmatter_are_tolerated():
    frontmatter, body = split_frontmatter("\n\n---\nname: n\ndescription: d\n---\nbody")

    assert frontmatter.strip() == "name: n\ndescription: d"
    assert body == "body"


# ---------------------------------------------------------------------------
# ContextManager: activation, replacement, clearing
# ---------------------------------------------------------------------------


def test_no_skill_keeps_existing_message_building_behaviour(config):
    context = make_context(config)
    context.set_system_prompt("SYSTEM")
    context.add_user_message("hello")

    messages = context.get_messages()

    assert messages == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "hello"},
    ]
    assert "Active Skill" not in messages[0]["content"]


def test_no_system_prompt_and_no_skill_produces_no_system_message(config):
    context = make_context(config)

    assert context.get_messages() == []
    assert context.build_system_prompt() is None


def test_skill_instructions_are_included_in_llm_messages(config):
    context = make_context(config)
    context.set_system_prompt("SYSTEM")
    context.add_user_message("hello")

    context.set_skill_instructions("SKILL BODY")
    messages = context.get_messages()

    # One system message + untouched history: no extra system turns.
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "SYSTEM" in messages[0]["content"]
    assert "SKILL BODY" in messages[0]["content"]
    assert messages[0]["content"].startswith("SYSTEM")
    assert messages[1] == {"role": "user", "content": "hello"}


def test_skill_without_system_prompt_still_reaches_the_llm(config):
    context = make_context(config)

    context.set_skill_instructions("SKILL BODY")
    messages = context.get_messages()

    assert messages[0]["role"] == "system"
    assert "SKILL BODY" in messages[0]["content"]


def test_clearing_skill_restores_messages_exactly(config):
    context = make_context(config)
    context.set_system_prompt("SYSTEM")
    context.add_user_message("hello")
    context.add_assistant_message("hi there")

    before = context.get_messages()

    context.set_skill_instructions("SKILL BODY")
    assert "SKILL BODY" in context.get_messages()[0]["content"]

    context.clear_skill_instructions()

    assert context.skill_instructions is None
    assert context.get_messages() == before


def test_set_skill_instructions_replaces_and_blank_clears(config):
    context = make_context(config)
    context.set_system_prompt("SYSTEM")

    context.set_skill_instructions("FIRST")
    context.set_skill_instructions("SECOND")
    content = context.get_messages()[0]["content"]
    assert "SECOND" in content
    assert "FIRST" not in content

    context.set_skill_instructions("   ")
    assert context.skill_instructions is None
    assert context.get_messages()[0]["content"] == "SYSTEM"

    context.set_skill_instructions("THIRD")
    context.set_skill_instructions(None)
    assert context.skill_instructions is None


def test_clear_preserves_system_prompt_backward_compatibly(config):
    context = make_context(config)
    context.set_system_prompt("SYSTEM")
    context.add_user_message("hello")

    context.clear()

    assert context.get_messages() == [{"role": "system", "content": "SYSTEM"}]


# ---------------------------------------------------------------------------
# Scoping and isolation
# ---------------------------------------------------------------------------


def test_use_skill_is_scoped_to_the_block(config):
    context = make_context(config)
    context.set_system_prompt("SYSTEM")

    assert context.skill_instructions is None

    with context.use_skill("SCOPED"):
        assert "SCOPED" in context.get_messages()[0]["content"]

    assert context.skill_instructions is None
    assert "SCOPED" not in context.get_messages()[0]["content"]


def test_use_skill_restores_previous_instructions(config):
    context = make_context(config)
    context.set_system_prompt("SYSTEM")
    context.set_skill_instructions("OUTER")

    with context.use_skill("INNER"):
        assert "INNER" in context.get_messages()[0]["content"]
        assert "OUTER" not in context.get_messages()[0]["content"]

    assert context.skill_instructions == "OUTER"


def test_use_skill_restores_even_when_the_block_raises(config):
    context = make_context(config)
    context.set_system_prompt("SYSTEM")

    with pytest.raises(RuntimeError):
        with context.use_skill("SCOPED"):
            raise RuntimeError("boom")

    assert context.skill_instructions is None
    assert "SCOPED" not in context.get_messages()[0]["content"]


def test_skill_instructions_are_isolated_between_contexts(config):
    one = make_context(config)
    two = make_context(config)
    one.set_system_prompt("ONE")
    two.set_system_prompt("TWO")

    one.set_skill_instructions("SKILL ONE")

    assert "SKILL ONE" in one.get_messages()[0]["content"]
    assert two.get_messages() == [{"role": "system", "content": "TWO"}]


# ---------------------------------------------------------------------------
# Clinical follow-up output validation
# ---------------------------------------------------------------------------


def _validate(payload):
    from customagents.doctorassistent.docagent import DoctorAssistantAgent

    return DoctorAssistantAgent.validate_questions(payload)


def test_validate_questions_accepts_exactly_three():
    assert _validate({"questions": ["a?", "b?", "c?"]}) == ["a?", "b?", "c?"]


def test_validate_questions_strips_whitespace():
    assert _validate({"questions": ["  a?  ", "b?", "c?"]}) == ["a?", "b?", "c?"]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="no-questions-key"),
        pytest.param({"questions": "not a list"}, id="questions-not-a-list"),
        pytest.param({"questions": ["a?", "b?"]}, id="too-few"),
        pytest.param({"questions": ["a?", "b?", "c?", "d?"]}, id="too-many"),
        pytest.param({"questions": ["a?", "", "c?"]}, id="blank-question"),
        pytest.param({"questions": ["a?", 2, "c?"]}, id="non-string-question"),
        pytest.param({"questions": ["a?", ["b?"], "c?"]}, id="nested-question"),
        pytest.param(["a?", "b?", "c?"], id="payload-not-an-object"),
    ],
)
def test_validate_questions_rejects_malformed_payloads(payload):
    with pytest.raises(ValueError):
        _validate(payload)


def test_parse_questions_handles_markdown_fences_and_reports_errors():
    from customagents.doctorassistent.docagent import DoctorAssistantAgent

    ok = DoctorAssistantAgent.parse_questions('```json\n{"questions": ["a?", "b?", "c?"]}\n```')
    assert ok == {"questions": ["a?", "b?", "c?"]}

    bad_json = DoctorAssistantAgent.parse_questions("not json at all")
    assert bad_json["questions"] == []
    assert "Invalid JSON" in bad_json["error"]

    wrong_count = DoctorAssistantAgent.parse_questions('{"questions": ["a?", "b?"]}')
    assert wrong_count["questions"] == []
    assert "Expected exactly 3" in wrong_count["error"]


# ---------------------------------------------------------------------------
# Agent runtime integration (the mandatory end-to-end check)
# ---------------------------------------------------------------------------


SKILL_WITH_MARKER = """---
name: sample
description: A sample skill.
---

SKILL-BODY-MARKER
"""


def _stub_session(config, payload: str):
    """Build a Session whose LLM is stubbed, capturing the messages it sees."""
    from agent.session import Session

    session = Session(config)
    session.context_manager = ContextManager(config, user_memory=None, tools=[])

    captured: list[list[dict]] = []

    async def fake_chat(messages, model=None, tools=None, tool_choice=None, **kwargs):
        captured.append(messages)
        yield StreamEvent(
            type=StreamEventType.TEXT_DELTA,
            text_delta=TextDelta(content=payload),
        )
        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason="stop",
            usage=TokenUsage(),
        )

    session.client.chat_completion = fake_chat
    return session, captured


def _build_agent(config, tmp_path, payload: str, agent_class):
    """Build an agent with a temp skill registry and a stubbed LLM."""
    registry = SkillRegistry(tmp_path)
    write_skill(tmp_path, "sample", SKILL_WITH_MARKER)

    session, captured = _stub_session(config, payload)
    agent = agent_class(config, session, skill_registry=registry)
    return agent, session, captured


def _doc_agent(config, tmp_path, payload: str):
    from customagents.doctorassistent.docagent import DoctorAssistantAgent

    return _build_agent(config, tmp_path, payload, DoctorAssistantAgent)


def _ehr_agent(config, tmp_path, payload: str):
    from customagents.ehragent.ehragent import EHRAgent

    return _build_agent(config, tmp_path, payload, EHRAgent)


def _drive_ehr(agent, message: str, skill: str | None = None):
    async def _run():
        return [event async for event in agent.run(message, skill=skill)]

    return asyncio.run(_run())


QUESTIONS_JSON = '{"questions": ["a?", "b?", "c?"]}'
EHR_REPLY = "Patient Summary:\n- Name: John Doe"


# --- reusable seam on the base Agent ---------------------------------------


def test_base_agent_exposes_a_skill_registry(config):
    from agent.agent import Agent
    from agent.events import AgentType

    agent = Agent(config, "ROLE", AgentType.EHR)

    # Injected-by-default, so every agent can use skills the same way.
    assert agent.skill_registry is not None
    assert agent.skill_registry.has(EXAMPLE_SKILL)


def test_base_agent_use_skill_activates_any_registered_skill(config):
    from agent.agent import Agent
    from agent.events import AgentType

    session, _captured = _stub_session(config, "unused")
    session.context_manager.set_system_prompt("ROLE")

    agent = Agent(config, "ROLE", AgentType.EHR)
    agent.session = session

    with agent.use_skill(EXAMPLE_SKILL) as loaded:
        assert loaded.name == EXAMPLE_SKILL
        assert "Generate EXACTLY 3 follow-up questions" in loaded.instructions
        assert session.context_manager.skill_instructions == loaded.instructions

    assert session.context_manager.skill_instructions is None


def test_base_agent_use_skill_with_none_is_a_noop(config):
    from agent.agent import Agent
    from agent.events import AgentType

    session, _captured = _stub_session(config, "unused")
    agent = Agent(config, "ROLE", AgentType.EHR)
    agent.session = session

    with agent.use_skill(None) as loaded:
        assert loaded is None

    assert session.context_manager.skill_instructions is None


def test_use_skill_requires_an_initialized_session(config):
    from agent.agent import Agent
    from agent.events import AgentType
    from utils.errors import AgentError

    agent = Agent(config, "ROLE", AgentType.EHR)
    agent.session.context_manager = None

    with pytest.raises(AgentError):
        with agent.use_skill(EXAMPLE_SKILL):
            pass


# --- Doc agent (delegates to the shared seam) --------------------------------


def test_doc_agent_skill_instructions_reach_the_llm(config, tmp_path):
    agent, session, captured = _doc_agent(config, tmp_path, QUESTIONS_JSON)

    result = asyncio.run(agent.recommend_questions(["Doctor: hello"], skill="sample"))

    assert result == {"questions": ["a?", "b?", "c?"]}
    assert captured, "the LLM was never called"

    messages = captured[0]
    assert messages[0]["role"] == "system"
    # The skill text is in the messages handed to the provider.
    assert "SKILL-BODY-MARKER" in messages[0]["content"]
    # ...without replacing the agent's own system prompt.
    assert "clinical decision support assistant" in messages[0]["content"].lower()
    # ...and history is intact, behind the single system message.
    assert messages[-1]["role"] == "user"
    assert "Doctor: hello" in messages[-1]["content"]

    # Scoped to the request: nothing is left behind.
    assert session.context_manager.skill_instructions is None
    assert "SKILL-BODY-MARKER" not in session.context_manager.build_system_prompt()


def test_doc_agent_without_a_skill_does_not_inject_instructions(config, tmp_path):
    agent, session, captured = _doc_agent(config, tmp_path, QUESTIONS_JSON)

    result = asyncio.run(agent.recommend_questions(["Doctor: hello"]))

    assert result == {"questions": ["a?", "b?", "c?"]}

    content = captured[0][0]["content"]
    assert "SKILL-BODY-MARKER" not in content
    assert "# Active Skill" not in content
    # The system message is exactly the agent's own prompt.
    assert content.lstrip().startswith("You are a clinical decision support")
    assert len(captured[0]) == 2  # system + user, no extra turns


def test_doc_agent_skill_does_not_leak_into_a_followup_request(config, tmp_path):
    agent, _session, captured = _doc_agent(config, tmp_path, QUESTIONS_JSON)

    asyncio.run(agent.recommend_questions(["Doctor: hello"], skill="sample"))
    assert "SKILL-BODY-MARKER" in captured[0][0]["content"]

    asyncio.run(agent.recommend_questions(["Doctor: again"]))

    assert "SKILL-BODY-MARKER" not in captured[1][0]["content"]


def test_doc_agent_unknown_skill_returns_an_error(config, tmp_path):
    agent, _session, captured = _doc_agent(config, tmp_path, QUESTIONS_JSON)

    result = asyncio.run(agent.recommend_questions(["Doctor: hello"], skill="nope"))

    assert result["questions"] == []
    assert "nope" in result["error"]
    assert captured == []  # never reached the LLM


def test_doc_agent_rejects_wrong_question_count(config, tmp_path):
    agent, _session, _captured = _doc_agent(
        config, tmp_path, '{"questions": ["a?", "b?"]}'
    )

    result = asyncio.run(agent.recommend_questions(["Doctor: hello"], skill="sample"))

    assert result["questions"] == []
    assert "Expected exactly 3" in result["error"]


# --- Doc agent: streaming run() path -----------------------------------------


def _drive_doc_run(agent, message: str, skill: str | None = None):
    async def _run():
        return [event async for event in agent.run(message, skill=skill)]

    return asyncio.run(_run())


def test_doc_agent_run_sets_the_system_prompt(config, tmp_path):
    """Regression: run() used to leave the session with no system prompt."""
    agent, _session, captured = _doc_agent(config, tmp_path, "not json")

    _drive_doc_run(agent, "hello")

    assert captured, "the LLM was never called"
    messages = captured[0]
    assert messages[0]["role"] == "system"
    assert "clinical decision support assistant" in messages[0]["content"].lower()
    assert messages[-1]["content"] == "hello"


def test_doc_agent_run_activates_a_skill(config, tmp_path):
    agent, session, captured = _doc_agent(config, tmp_path, "not json")

    _drive_doc_run(agent, "hello", skill="sample")

    content = captured[0][0]["content"]
    assert "SKILL-BODY-MARKER" in content
    assert "clinical decision support assistant" in content.lower()
    assert session.context_manager.skill_instructions is None


def test_doc_agent_run_without_a_skill_has_no_skill_section(config, tmp_path):
    agent, _session, captured = _doc_agent(config, tmp_path, "not json")

    _drive_doc_run(agent, "hello")

    content = captured[0][0]["content"]
    assert "SKILL-BODY-MARKER" not in content
    assert "# Active Skill" not in content
    assert content.lstrip().startswith("You are a clinical decision support")


# --- EHR agent (first consumer of the reusable seam) -------------------------


def test_ehr_agent_skill_instructions_reach_the_llm(config, tmp_path):
    agent, session, captured = _ehr_agent(config, tmp_path, EHR_REPLY)

    events = _drive_ehr(agent, "Show medications for patient 12345", skill="sample")

    assert events, "the agent produced no events"
    assert captured, "the LLM was never called"

    messages = captured[0]
    assert messages[0]["role"] == "system"
    assert "SKILL-BODY-MARKER" in messages[0]["content"]
    # The EHR agent's own prompt is preserved alongside the skill.
    assert "Healthcare MCP Agent" in messages[0]["content"]
    assert messages[-1]["content"] == "Show medications for patient 12345"

    # Scoped to the request.
    assert session.context_manager.skill_instructions is None
    assert "SKILL-BODY-MARKER" not in session.context_manager.build_system_prompt()


def test_ehr_agent_loads_the_bundled_ehr_skill(config, tmp_path):
    """No temp registry: the real bundled library resolves by name."""
    from customagents.ehragent.ehragent import EHRAgent

    session, captured = _stub_session(config, EHR_REPLY)
    agent = EHRAgent(config, session)

    _drive_ehr(agent, "Show allergies for patient 12345", skill=EHR_SKILL)

    content = captured[0][0]["content"]
    assert "Never hallucinate medical records" in content
    assert session.context_manager.skill_instructions is None


def test_ehr_agent_without_a_skill_has_no_skill_section(config, tmp_path):
    agent, _session, captured = _ehr_agent(config, tmp_path, EHR_REPLY)

    _drive_ehr(agent, "Show medications for patient 12345")

    content = captured[0][0]["content"]
    assert "SKILL-BODY-MARKER" not in content
    assert "# Active Skill" not in content
    assert "Healthcare MCP Agent" in content


def test_ehr_agent_skill_does_not_leak_into_a_followup_request(config, tmp_path):
    agent, _session, captured = _ehr_agent(config, tmp_path, EHR_REPLY)

    _drive_ehr(agent, "Show medications", skill="sample")
    assert "SKILL-BODY-MARKER" in captured[0][0]["content"]

    _drive_ehr(agent, "Show allergies")

    assert "SKILL-BODY-MARKER" not in captured[1][0]["content"]


def test_ehr_agent_unknown_skill_surfaces_as_an_agent_error(config, tmp_path):
    agent, _session, captured = _ehr_agent(config, tmp_path, EHR_REPLY)

    events = _drive_ehr(agent, "Show medications", skill="nope")

    errors = [
        event for event in events
        if event.type == AgentEventType.AGENT_ERROR
    ]
    assert errors, "expected an agent_error event"
    assert "nope" in errors[0].data["error"]
    assert captured == []  # never reached the LLM


def test_skill_grant_is_not_a_tool_grant(config, tmp_path):
    agent, session, _captured = _ehr_agent(config, tmp_path, EHR_REPLY)

    tools_before = len(session.tool_registry.get_tools())
    allowed_before = config.allowed_tools

    _drive_ehr(agent, "Show medications", skill="sample")

    assert len(session.tool_registry.get_tools()) == tools_before
    assert session.config.allowed_tools == allowed_before


# ---------------------------------------------------------------------------
# Default skill configured per agent — this is how an agent "knows" its skill
# ---------------------------------------------------------------------------


OTHER_SKILL = """---
name: other
description: Another skill.
---

OTHER-MARKER
"""


def test_agent_yaml_accepts_an_optional_skill_reference():
    from config.schemas import AgentYaml

    assert AgentYaml(name="doc", skill=EXAMPLE_SKILL).skill == EXAMPLE_SKILL
    # Optional: existing configs without a skill keep working.
    assert AgentYaml(name="doc").skill is None


def test_shipped_doc_and_ehr_agents_declare_a_skill():
    from config.agent_config_loader import AgentConfigLoader

    loader = AgentConfigLoader(Config(), base_dir=PROJECT_ROOT)

    assert loader.load_agent("doc").skill == EXAMPLE_SKILL
    assert loader.load_agent("ehr").skill == EHR_SKILL


def _write_project(tmp_path: Path, agents_yaml: dict, agent_configs: dict) -> None:
    """Write a minimal two-level config tree for loader validation tests."""
    (tmp_path / "customagents").mkdir(parents=True, exist_ok=True)
    (tmp_path / "agents.yaml").write_text(yaml.safe_dump(agents_yaml), encoding="utf-8")
    for name, data in agent_configs.items():
        directory = tmp_path / "customagents" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def test_loader_rejects_an_unknown_skill(tmp_path):
    from config.agent_config_loader import AgentConfigLoader
    from utils.errors import ConfigError

    _write_project(
        tmp_path,
        {"models": {"clinical_model": {"name": "gpt-4.1"}},
         "agents": {"doc": {"enabled": True}}},
        {"doc": {"name": "doc", "model": "clinical_model", "prompt": "doc",
                 "skill": "does-not-exist"}},
    )

    with pytest.raises(ConfigError) as exc:
        AgentConfigLoader(Config(), base_dir=tmp_path, known_agents={"doc"}).load()

    assert "does-not-exist" in str(exc.value)


def test_loader_accepts_a_valid_skill_reference(tmp_path):
    from config.agent_config_loader import AgentConfigLoader

    _write_project(
        tmp_path,
        {"models": {"clinical_model": {"name": "gpt-4.1"}},
         "agents": {"doc": {"enabled": True}}},
        {"doc": {"name": "doc", "model": "clinical_model", "prompt": "doc",
                 "skill": EXAMPLE_SKILL}},
    )

    resolved = AgentConfigLoader(
        Config(), base_dir=tmp_path, known_agents={"doc"}
    ).load()

    assert resolved.enabled_agents["doc"].skill == EXAMPLE_SKILL


def test_override_applier_applies_the_configured_skill():
    from customagents.composition import AgentOverrideApplier

    class _StubAgent:
        system_prompt = "original"
        configured_tools = None
        allowed_tool_names = None

    agent = _StubAgent()
    AgentOverrideApplier().apply(agent, None, None, EHR_SKILL)

    assert agent.default_skill == EHR_SKILL


def test_override_applier_leaves_the_skill_unset_when_absent():
    from customagents.composition import AgentOverrideApplier

    class _StubAgent:
        system_prompt = "original"
        configured_tools = None
        allowed_tool_names = None

    agent = _StubAgent()
    AgentOverrideApplier().apply(agent, None, None, None)

    assert getattr(agent, "default_skill", None) is None


def test_active_skill_resolution(config):
    from agent.agent import Agent
    from agent.events import AgentType

    agent = Agent(config, "ROLE", AgentType.DOC)

    # No skill configured: only an explicit name activates one.
    assert agent.active_skill(None) is None
    assert agent.active_skill("explicit") == "explicit"

    agent.default_skill = EXAMPLE_SKILL

    assert agent.active_skill(None) == EXAMPLE_SKILL   # the configured default
    assert agent.active_skill("other") == "other"      # caller override wins
    assert agent.active_skill("") is None              # explicit opt-out


def test_doc_agent_uses_its_configured_skill_without_being_asked(config, tmp_path):
    agent, session, captured = _doc_agent(config, tmp_path, QUESTIONS_JSON)
    agent.default_skill = "sample"  # what the factory applies from config.yaml

    result = asyncio.run(agent.recommend_questions(["Doctor: hello"]))

    assert result == {"questions": ["a?", "b?", "c?"]}
    assert "SKILL-BODY-MARKER" in captured[0][0]["content"]
    assert session.context_manager.skill_instructions is None


def test_doc_agent_configured_skill_applies_to_run_too(config, tmp_path):
    agent, _session, captured = _doc_agent(config, tmp_path, "not json")
    agent.default_skill = "sample"

    _drive_doc_run(agent, "hello")

    assert "SKILL-BODY-MARKER" in captured[0][0]["content"]


def test_explicit_skill_overrides_the_configured_default(config, tmp_path):
    write_skill(tmp_path, "other", OTHER_SKILL)
    agent, _session, captured = _doc_agent(config, tmp_path, QUESTIONS_JSON)
    agent.default_skill = "sample"

    asyncio.run(agent.recommend_questions(["Doctor: hello"], skill="other"))

    content = captured[0][0]["content"]
    assert "OTHER-MARKER" in content
    assert "SKILL-BODY-MARKER" not in content


def test_empty_skill_string_opts_out_of_the_configured_default(config, tmp_path):
    agent, _session, captured = _doc_agent(config, tmp_path, QUESTIONS_JSON)
    agent.default_skill = "sample"

    asyncio.run(agent.recommend_questions(["Doctor: hello"], skill=""))

    content = captured[0][0]["content"]
    assert "SKILL-BODY-MARKER" not in content
    assert "# Active Skill" not in content


def test_ehr_agent_uses_its_configured_skill_without_being_asked(config, tmp_path):
    agent, session, captured = _ehr_agent(config, tmp_path, EHR_REPLY)
    agent.default_skill = "sample"

    _drive_ehr(agent, "Show medications for patient 12345")

    assert "SKILL-BODY-MARKER" in captured[0][0]["content"]
    assert "Healthcare MCP Agent" in captured[0][0]["content"]
    assert session.context_manager.skill_instructions is None
