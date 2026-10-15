"""Tests for the agent composition layer.

Each collaborator is exercised in isolation (SRP), the open/closed seams
are verified (registering a new agent / pseudo tool), and the dependency
injection seam is exercised (DIP).
"""

from pathlib import Path

import pytest

from config.agent_config_loader import ResolvedAgentConfig
from config.config import AgentsSection, Config, ModelConfig, ModelsSection
from config.schemas import AgentSettings, AgentYaml, ApplicationAgentConfig
from customagents.composition import (
    AgentComponents,
    AgentConfigProvider,
    AgentOverrideApplier,
    AgentSettingsApplier,
    AgentTypeRegistry,
    ModelResolver,
    PromptResolver,
    PseudoToolRegistry,
    ToolResolver,
)
from registries.model_registry import ModelRegistry
from registries.prompt_registry import PromptRegistry
from utils.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _reset_factory_components():
    """Keep a test's injected/registered components from leaking."""
    from customagents.factory import AgentFactory

    yield
    AgentFactory.reset_components()


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


class _FakeTool:
    def __init__(self, name, kind=None):
        self.name = name
        self.kind = kind


class _FakeKind:
    def __init__(self, value):
        self.value = value


class _FakeRegistry:
    def __init__(self, tools):
        self._tools = {tool.name: tool for tool in tools}

    def get(self, name):
        return self._tools.get(name)

    def get_tools(self):
        return list(self._tools.values())


class _DummyAgent:
    """Minimal agent used to prove new types can be registered (OCP)."""

    def __init__(self, config, session=None):
        self.config = config
        self.session = session
        self.system_prompt = "hardcoded"
        self.allowed_tool_names = None
        self.configured_tools = None


class _NullLoader:
    """Config loader stub that resolves nothing (no filesystem access)."""

    resolved = None

    def load(self):
        return None


def _components(classes, known_agents=None) -> AgentComponents:
    """Build a component bundle backed by a null config loader."""
    return AgentComponents(
        types=AgentTypeRegistry(classes),
        configs=AgentConfigProvider(
            known_agents=set(known_agents or classes),
            loader_factory=lambda _config, _known: _NullLoader(),
        ),
        models=ModelResolver(),
        prompts=PromptResolver(),
        tools=ToolResolver(),
        settings=AgentSettingsApplier(),
        overrides=AgentOverrideApplier(),
    )


# ---------------------------------------------------------------------------
# AgentTypeRegistry (SRP / OCP)
# ---------------------------------------------------------------------------


def test_registry_registers_and_resolves_without_touching_the_factory():
    registry = AgentTypeRegistry({"doc": _DummyAgent})

    registry.register("triage", _DummyAgent)  # OCP: no factory edit

    assert registry.get("triage") is _DummyAgent
    assert registry.get("TRIAGE") is _DummyAgent  # case-insensitive
    assert "triage" in registry
    assert registry.known_names() == {"doc", "triage"}
    assert registry.get("missing") is None


def test_factory_builds_a_newly_registered_agent():
    """A brand-new agent needs no change to AgentFactory."""
    from customagents.factory import AgentFactory

    AgentFactory.components().types.register("dummy", _DummyAgent)

    agent = AgentFactory.create("dummy", Config(), session=None)

    assert isinstance(agent, _DummyAgent)
    assert agent.system_prompt == "hardcoded"  # no config → default kept


# ---------------------------------------------------------------------------
# AgentComponents / DIP
# ---------------------------------------------------------------------------


def test_configure_swaps_the_collaborator_bundle():
    from customagents.factory import DEFAULT_AGENT_CLASSES, AgentFactory

    AgentFactory.configure(_components({"only": _DummyAgent}))

    assert AgentFactory.known_agent_names() == {"only"}
    assert isinstance(AgentFactory.create("only", Config(), session=None), _DummyAgent)

    AgentFactory.reset_components()
    assert AgentFactory.known_agent_names() == set(DEFAULT_AGENT_CLASSES)


# ---------------------------------------------------------------------------
# AgentConfigProvider (SRP / DIP)
# ---------------------------------------------------------------------------


def test_config_provider_rejects_disabled_agent():
    config = Config(
        agents=AgentsSection(definitions={
            "soap": ApplicationAgentConfig(enabled=False),
        }),
    )
    provider = AgentConfigProvider(known_agents={"soap"})

    with pytest.raises(ConfigError) as exc:
        provider.provide("soap", config, ResolvedAgentConfig())

    assert "soap" in str(exc.value)
    assert "disabled" in str(exc.value)


def test_config_provider_returns_none_for_unconfigured_agent():
    provider = AgentConfigProvider(known_agents={"soap"})
    agent_yaml, _model_registry = provider.provide(
        "soap", Config(), ResolvedAgentConfig()
    )

    # No application-level entry → the agent keeps its hardcoded defaults.
    assert agent_yaml is None


def test_config_provider_returns_resolved_config():
    expected = AgentYaml(name="soap", model="m", prompt="soap")
    provider = AgentConfigProvider(known_agents={"soap"})
    registry = ModelRegistry({"m": ModelConfig(name="x")})

    agent_yaml, model_registry = provider.provide(
        "soap",
        Config(),
        ResolvedAgentConfig(enabled_agents={"soap": expected}, model_registry=registry),
    )

    assert agent_yaml is expected
    assert model_registry is registry


# ---------------------------------------------------------------------------
# ModelResolver / AgentSettingsApplier (SRP)
# ---------------------------------------------------------------------------


def test_model_resolver_derives_a_config_without_mutating_the_original():
    config = Config(
        models=ModelsSection(definitions={
            "clinical": ModelConfig(provider="openai", name="gpt-4.1",
                                    litellm_model="openai/gpt-4.1", temperature=0.1),
        }),
    )
    original_name = config.model_name

    derived = ModelResolver().resolve(AgentYaml(name="a", model="clinical"), config)

    assert derived.model_name == "gpt-4.1"
    assert derived.model.temperature == 0.1
    assert config.model_name == original_name  # untouched


def test_model_resolver_is_a_noop_without_a_model():
    config = Config()
    assert ModelResolver().resolve(AgentYaml(name="a"), config) is config
    assert ModelResolver().resolve(None, config) is config


def test_settings_applier_is_a_noop_without_settings():
    config = Config()
    assert AgentSettingsApplier().apply(config, AgentYaml(name="a")) is config


def test_settings_applier_applies_on_a_copy():
    config = Config()
    original_turns = config.max_turns

    derived = AgentSettingsApplier().apply(
        config,
        AgentYaml(name="a", settings=AgentSettings(max_turns=7, temperature=0.2)),
    )

    assert derived.max_turns == 7
    assert derived.model.temperature == 0.2
    assert config.max_turns == original_turns


# ---------------------------------------------------------------------------
# PromptResolver (SRP / DIP)
# ---------------------------------------------------------------------------


def test_prompt_resolver_returns_none_without_a_prompt_entry():
    assert PromptResolver().resolve("doc", AgentYaml(name="doc"), Config()) is None


def test_prompt_resolver_uses_the_injected_registry():
    resolver = PromptResolver(
        registry_factory=lambda: PromptRegistry({"doc": "INJECTED PROMPT"})
    )

    prompt = resolver.resolve("doc", AgentYaml(name="doc", prompt="doc"), Config())

    assert prompt == "INJECTED PROMPT"


# ---------------------------------------------------------------------------
# ToolResolver / PseudoToolRegistry (SRP / OCP / DIP)
# ---------------------------------------------------------------------------


def _resolver(tools, pseudo=None):
    return ToolResolver(pseudo_tools=pseudo, registry_factory=lambda _c: _FakeRegistry(tools))


def test_tool_resolver_returns_none_when_tools_is_none():
    assert _resolver([]).resolve("a", AgentYaml(name="a"), Config()) is None


def test_tool_resolver_empty_list_means_no_tools():
    result = _resolver([]).resolve("a", AgentYaml(name="a", tools=[]), Config())
    assert result == []


def test_tool_resolver_raises_for_unknown_tool():
    with pytest.raises(ConfigError) as exc:
        _resolver([_FakeTool("memory")]).resolve(
            "medical", AgentYaml(name="medical", tools=["ghost"]), Config()
        )

    assert "medical" in str(exc.value)
    assert "ghost" in str(exc.value)


def test_tool_resolver_resolves_registered_tools():
    resolver = _resolver([_FakeTool("memory"), _FakeTool("wiki_search")])

    tools = resolver.resolve(
        "medical", AgentYaml(name="medical", tools=["memory", "wiki_search"]), Config()
    )

    assert [t.name for t in tools] == ["memory", "wiki_search"]


def test_pseudo_tool_registry_is_open_for_extension():
    """A new pseudo tool is added without editing ToolResolver (OCP)."""
    pseudo = PseudoToolRegistry()
    pseudo.register("everything", lambda registry: registry.get_tools())

    resolver = _resolver([_FakeTool("a"), _FakeTool("b")], pseudo=pseudo)
    tools = resolver.resolve(
        "x", AgentYaml(name="x", tools=["everything"]), Config()
    )

    assert [t.name for t in tools] == ["a", "b"]


def test_default_pseudo_tools_expand_mcp_kind():
    from customagents.composition import default_pseudo_tools

    registry = _FakeRegistry([
        _FakeTool("memory", _FakeKind("memory")),
        _FakeTool("fhir__get_patient", _FakeKind("mcp")),
    ])

    assert [t.name for t in default_pseudo_tools().expand("mcp", registry)] == [
        "fhir__get_patient"
    ]


# ---------------------------------------------------------------------------
# AgentOverrideApplier (SRP)
# ---------------------------------------------------------------------------


def test_override_applier_sets_prompt_and_tools():
    agent = _DummyAgent(Config())
    AgentOverrideApplier().apply(agent, "PROMPT", [_FakeTool("memory")])

    assert agent.system_prompt == "PROMPT"
    assert [t.name for t in agent.configured_tools] == ["memory"]
    assert agent.allowed_tool_names == ["memory"]


def test_override_applier_leaves_defaults_when_nothing_resolved():
    agent = _DummyAgent(Config())
    AgentOverrideApplier().apply(agent, None, None)

    assert agent.system_prompt == "hardcoded"
    assert agent.allowed_tool_names is None