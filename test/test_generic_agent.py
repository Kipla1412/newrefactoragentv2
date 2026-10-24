"""Tests for the configuration-driven agent (GenericAgent).

The point of this class is that an agent of the shape "prompt + model + tools +
skill" needs **no new Python**: several agents share one class and declare it
with ``agent_class: generic``. These tests prove that end to end, and check the
loader/registry plumbing that makes it safe.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from agent.events import AgentEventType, AgentType
from client.response import StreamEvent, StreamEventType, TextDelta, TokenUsage
from config.agent_config_loader import ResolvedAgentConfig
from config.config import Config, ModelConfig
from config.schemas import AgentYaml
from context.manager import ContextManager
from customagents.composition import AgentOverrideApplier
from customagents.genericagent.genericagent import GenericAgent
from registries.model_registry import ModelRegistry
from registries.prompt_registry import PromptRegistry
from utils.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _reset_factory_components():
    """Keep any component swap from leaking into other tests."""
    from customagents.factory import AgentFactory

    AgentFactory.reset_components()
    yield
    AgentFactory.reset_components()


def _stub_session(config, payload: str = "handled"):
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


def _drive(agent, message: str = "hello", skill: str | None = None):
    async def _run():
        return [event async for event in agent.run(message, skill=skill)]

    return asyncio.run(_run())


def _resolved(agent_yaml: AgentYaml, models: dict | None = None) -> ResolvedAgentConfig:
    """Build a pre-resolved config, so the factory never reads YAML."""
    return ResolvedAgentConfig(
        enabled_agents={agent_yaml.name: agent_yaml},
        model_registry=ModelRegistry(models or {}),
        prompt_registry=PromptRegistry.from_builtin(),
    )


# ---------------------------------------------------------------------------
# Registration and schema
# ---------------------------------------------------------------------------


def test_generic_agent_is_registered():
    from customagents.factory import AgentFactory

    assert "generic" in AgentFactory.known_agent_names()


def test_agent_type_has_generic():
    assert AgentType.GENERIC.value == "generic"


def test_existing_agents_are_still_registered():
    from customagents.factory import AgentFactory

    names = AgentFactory.known_agent_names()

    for expected in ("doc", "soap", "ehr", "consult", "diarize", "pdf_chat"):
        assert expected in names


def test_agent_yaml_accepts_agent_class_and_name():
    parsed = AgentYaml(
        name="demographics",
        agent_class="generic",
        agent_name="DemographicsAgent",
    )

    assert parsed.agent_class == "generic"
    assert parsed.agent_name == "DemographicsAgent"
    # Both optional: existing configs are unaffected.
    assert AgentYaml(name="doc").agent_class is None
    assert AgentYaml(name="doc").agent_name is None


def test_override_applier_applies_the_display_name():
    class _StubAgent:
        system_prompt = "original"
        configured_tools = None
        allowed_tool_names = None

    agent = _StubAgent()
    AgentOverrideApplier().apply(agent, None, None, None, "DemographicsAgent")

    assert agent.configured_agent_name == "DemographicsAgent"


def test_override_applier_leaves_the_name_unset_when_absent():
    class _StubAgent:
        system_prompt = "original"
        configured_tools = None
        allowed_tool_names = None

    agent = _StubAgent()
    AgentOverrideApplier().apply(agent, None, None)

    assert getattr(agent, "configured_agent_name", None) is None


# ---------------------------------------------------------------------------
# Built entirely from configuration — no per-agent Python
# ---------------------------------------------------------------------------


def test_config_driven_agent_is_built_from_config_alone(config):
    """A name with no registered class resolves through agent_class."""
    from customagents.factory import AgentFactory

    # "triage_nurse" is deliberately not registered as a class.
    agent_yaml = AgentYaml(
        name="triage_nurse",
        agent_class="generic",
        model="clinical_model",
        prompt="doc",
        skill="clinical-followup-questions",
        agent_name="TriageNurseAgent",
    )
    resolved = _resolved(
        agent_yaml,
        {"clinical_model": ModelConfig(
            provider="openai", name="gpt-4o-mini", litellm_model="openai/gpt-4o-mini"
        )},
    )

    agent = AgentFactory.create("triage_nurse", config, None, resolved=resolved)

    assert isinstance(agent, GenericAgent)
    # model applied from config
    assert agent.config.model_name == "gpt-4o-mini"
    # prompt applied from the registry
    assert "clinical decision support assistant" in agent.system_prompt.lower()
    # skill and display name applied
    assert agent.default_skill == "clinical-followup-questions"
    assert agent.configured_agent_name == "TriageNurseAgent"
    assert agent.display_name == "TriageNurseAgent"


def test_registered_name_wins_over_agent_class(config):
    """A name with a registered class never falls through to agent_class."""
    from customagents.factory import AgentFactory

    resolved = _resolved(AgentYaml(name="doc", agent_class="generic", prompt="doc"))

    agent = AgentFactory.create("doc", config, None, resolved=resolved)

    assert not isinstance(agent, GenericAgent)


def test_unknown_name_without_agent_class_still_fails(config):
    from customagents.factory import AgentFactory

    with pytest.raises(ValueError):
        AgentFactory.create("no_such_agent", config, None)


def test_agent_class_pointing_at_an_unregistered_class_fails(config):
    from customagents.factory import AgentFactory

    resolved = _resolved(AgentYaml(name="triage_nurse", agent_class="nope"))

    with pytest.raises(ValueError):
        AgentFactory.create("triage_nurse", config, None, resolved=resolved)


# ---------------------------------------------------------------------------
# Runtime behaviour
# ---------------------------------------------------------------------------


def _configured_agent(config, system_prompt="You are a demographics assistant."):
    """A GenericAgent wired the way the factory would wire it."""
    session, captured = _stub_session(config)
    agent = GenericAgent(config, session)
    agent.system_prompt = system_prompt
    return agent, session, captured


def test_generic_agent_runs_the_standard_loop(config):
    agent, session, captured = _configured_agent(config)

    events = _drive(agent, "Summarise patient 10008")

    assert captured, "the LLM was never called"
    messages = captured[0]
    assert messages[0]["role"] == "system"
    assert "demographics assistant" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Summarise patient 10008"}

    # Uses the shared agentic loop, and tags events with its own type.
    assert any(
        getattr(event.data.get("agent"), "value", None) == AgentType.GENERIC.value
        for event in events
        if isinstance(event.data, dict)
    )
    assert session.agent_name == "GenericAgent"


def test_generic_agent_uses_its_configured_display_name(config):
    agent, session, _captured = _configured_agent(config)
    agent.configured_agent_name = "DemographicsAgent"

    assert agent.display_name == "DemographicsAgent"

    _drive(agent, "hello")

    assert session.agent_name == "DemographicsAgent"


def test_generic_agent_reports_a_missing_prompt_without_calling_the_llm(config):
    agent, _session, captured = _configured_agent(config, system_prompt="")

    events = _drive(agent, "hello")

    errors = [e for e in events if e.type == AgentEventType.AGENT_ERROR]
    assert errors, "expected an agent_error event"
    assert "no prompt configured" in errors[0].data["error"]
    assert captured == [], "an unconfigured agent must not reach the model"


def test_generic_agent_activates_its_configured_skill(config):
    agent, session, captured = _configured_agent(config)
    agent.default_skill = "ehr-fhir-retrieval"

    _drive(agent, "summarise patient 1")

    content = captured[0][0]["content"]
    assert "Never hallucinate medical records" in content      # skill instructions
    assert "demographics assistant" in content                 # its own prompt
    assert "# Active Skill" in content

    # Scoped to the request.
    assert session.context_manager.skill_instructions is None


def test_generic_agent_without_a_skill_has_no_skill_section(config):
    agent, _session, captured = _configured_agent(config)

    _drive(agent, "hello")

    content = captured[0][0]["content"]
    assert "# Active Skill" not in content


def test_generic_agent_reports_unknown_skills(config):
    agent, _session, captured = _configured_agent(config)
    agent.default_skill = "does-not-exist"

    events = _drive(agent, "hello")

    errors = [e for e in events if e.type == AgentEventType.AGENT_ERROR]
    assert errors
    assert "does-not-exist" in errors[0].data["error"]
    assert captured == []


# ---------------------------------------------------------------------------
# Loader validation for config-driven agents
# ---------------------------------------------------------------------------


def _write_project(tmp_path: Path, agents_yaml: dict, agent_configs: dict) -> None:
    (tmp_path / "customagents").mkdir(parents=True, exist_ok=True)
    (tmp_path / "agents.yaml").write_text(yaml.safe_dump(agents_yaml), encoding="utf-8")
    for name, data in agent_configs.items():
        directory = tmp_path / "customagents" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


MODELS = {"clinical_model": {"name": "gpt-4.1"}}


def test_loader_accepts_a_config_driven_agent(tmp_path):
    from config.agent_config_loader import AgentConfigLoader

    _write_project(
        tmp_path,
        {"models": MODELS, "agents": {"demographics": {"enabled": True}}},
        {"demographics": {
            "name": "demographics",
            "agent_class": "generic",
            "model": "clinical_model",
            "prompt": "doc",
            "agent_name": "DemographicsAgent",
        }},
    )

    resolved = AgentConfigLoader(
        Config(), base_dir=tmp_path, known_agents={"generic"}
    ).load()

    agent_yaml = resolved.enabled_agents["demographics"]
    assert agent_yaml.agent_class == "generic"
    assert agent_yaml.agent_name == "DemographicsAgent"


def test_loader_rejects_a_config_driven_agent_without_agent_class(tmp_path):
    from config.agent_config_loader import AgentConfigLoader

    _write_project(
        tmp_path,
        {"models": MODELS, "agents": {"demographics": {"enabled": True}}},
        {"demographics": {
            "name": "demographics",
            "model": "clinical_model",
            "prompt": "doc",
        }},
    )

    with pytest.raises(ConfigError) as exc:
        AgentConfigLoader(Config(), base_dir=tmp_path, known_agents={"generic"}).load()

    assert "agent_class" in str(exc.value)


def test_loader_rejects_an_unknown_agent_class(tmp_path):
    from config.agent_config_loader import AgentConfigLoader

    _write_project(
        tmp_path,
        {"models": MODELS, "agents": {"demographics": {"enabled": True}}},
        {"demographics": {
            "name": "demographics",
            "agent_class": "not_a_class",
            "model": "clinical_model",
            "prompt": "doc",
        }},
    )

    with pytest.raises(ConfigError) as exc:
        AgentConfigLoader(Config(), base_dir=tmp_path, known_agents={"generic"}).load()

    assert "not_a_class" in str(exc.value)


def test_loader_still_accepts_normally_registered_agents(tmp_path):
    """Backward compatibility: agents registered under their own name."""
    from config.agent_config_loader import AgentConfigLoader

    _write_project(
        tmp_path,
        {"models": MODELS, "agents": {"medical": {"enabled": True}}},
        {"medical": {"name": "medical", "model": "clinical_model", "prompt": "medical"}},
    )

    resolved = AgentConfigLoader(
        Config(), base_dir=tmp_path, known_agents={"medical"}
    ).load()

    assert resolved.enabled_agents["medical"].agent_class is None


BUILTIN_AGENTS = (
    "doc", "soap", "assessment", "clinical_extraction", "consult", "intake",
    "ehr", "medical", "diarize", "voice_intake", "voice_consult",
    "document_retrieval", "pdf_chat", "demographics",
)


def test_shipped_agents_declare_no_agent_class():
    """Every shipped agent is registered under its own name.

    The ``agent_class`` escape hatch exists for configuration-only agents, but
    none ship that way today — this guards against one drifting into it
    silently.
    """
    from config.agent_config_loader import AgentConfigLoader

    loader = AgentConfigLoader(Config(), base_dir=PROJECT_ROOT)
    discovered = loader._discover_agent_config_files()

    assert len(discovered) == len(BUILTIN_AGENTS)
    for name in BUILTIN_AGENTS:
        assert discovered[name].agent_class is None, f"{name} should not set agent_class"
