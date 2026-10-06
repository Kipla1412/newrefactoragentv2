"""Tests for the configuration-driven agent selection.

Covers the ModelRegistry, PromptRegistry, the two-level agents.yaml +
per-agent config.yaml loading, and the AgentFactory composing them.
"""

from pathlib import Path

import pytest
import yaml

from config.config import Config, ModelConfig, ModelsSection
from config.loader import load_config
from config.schemas import AgentYaml
from registries.model_registry import ModelRegistry
from registries.prompt_registry import PromptRegistry
from tools.registry import create_default_registry
from utils.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def test_agents_yaml_loads_through_config_loader():
    config = load_config(cwd=PROJECT_ROOT)

    assert "clinical_model" in config.models.definitions
    assert "fast_model" in config.models.definitions
    assert config.models.definitions["clinical_model"].name == "gpt-4.1"
    assert config.models.definitions["clinical_model"].temperature == 0.1
    # Application-level section only tracks enabled flags.
    assert config.agents.definitions["medical"].enabled is True


def test_yaml_does_not_create_a_second_config_type():
    assert isinstance(load_config(cwd=PROJECT_ROOT), Config)


# ---------------------------------------------------------------------------
# Per-agent config discovery
# ---------------------------------------------------------------------------

def test_per_agent_config_files_discovered():
    from config.agent_config_loader import AgentConfigLoader
    from customagents.factory import AgentFactory

    config = load_config(cwd=PROJECT_ROOT)
    loader = AgentConfigLoader(
        config, base_dir=PROJECT_ROOT,
        known_agents=AgentFactory.known_agent_names(),
    )
    resolved = loader.load()

    # Every enabled agent resolved a config.yaml.
    assert set(resolved.enabled_agents) == set(config.agents.definitions)

    medical = resolved.enabled_agents["medical"]
    assert medical.model == "clinical_model"
    assert medical.prompt == "medical"
    assert medical.tools == ["memory", "wiki_search", "neo4j_query"]


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

def test_model_registry_resolves_configured_model():
    registry = ModelRegistry({
        "clinical_model": ModelConfig(name="gpt-4.1", temperature=0.1),
    })
    model = registry.get("clinical_model")
    assert model.name == "gpt-4.1"
    assert model.temperature == 0.1


def test_model_registry_unknown_model_raises():
    registry = ModelRegistry({})
    with pytest.raises(ConfigError) as exc:
        registry.get("unknown_model")
    assert "unknown_model" in str(exc.value)


def test_model_registry_apply_does_not_mutate_original():
    config = Config()
    original_name = config.model_name
    registry = ModelRegistry({
        "clinical_model": ModelConfig(name="gpt-4.1", temperature=0.1),
    })

    derived = registry.apply("clinical_model", config)

    assert derived.model_name == "gpt-4.1"
    assert derived.model.temperature == 0.1
    assert config.model_name == original_name


# ---------------------------------------------------------------------------
# PromptRegistry
# ---------------------------------------------------------------------------

def test_prompt_registry_builtin_contains_logical_names():
    registry = PromptRegistry.from_builtin()
    expected = {
        "doc", "soap", "assessment", "consult", "intake", "ehr",
        "clinical_extraction", "voice_intake", "voice_consult",
        "diarize", "medical", "document_retrieval", "pdf_chat",
    }
    assert expected.issubset(set(registry.prompts.keys()))


def test_prompt_registry_unknown_prompt_raises():
    registry = PromptRegistry({})
    with pytest.raises(ConfigError) as exc:
        registry.get("unknown_prompt")
    assert "unknown_prompt" in str(exc.value)


# ---------------------------------------------------------------------------
# AgentConfigLoader validation
# ---------------------------------------------------------------------------

def _write_project(tmp_path: Path, agents_yaml: dict, agent_configs: dict):
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "customagents").mkdir(parents=True)
    (tmp_path / "config" / "agents.yaml").write_text(yaml.safe_dump(agents_yaml))
    for name, data in agent_configs.items():
        d = tmp_path / "customagents" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.yaml").write_text(yaml.safe_dump(data))


def test_loader_rejects_unknown_model(tmp_path):
    from config.agent_config_loader import AgentConfigLoader

    _write_project(
        tmp_path,
        {"models": {"clinical_model": {"name": "gpt-4.1"}},
         "agents": {"medical": {"enabled": True}}},
        {"medical": {"name": "medical", "model": "unknown_model", "prompt": "medical"}},
    )
    loader = AgentConfigLoader(
        Config(), base_dir=tmp_path, known_agents={"medical"},
    )
    with pytest.raises(ConfigError) as exc:
        loader.load()
    assert "medical" in str(exc.value)
    assert "unknown_model" in str(exc.value)


def test_loader_rejects_unknown_prompt(tmp_path):
    from config.agent_config_loader import AgentConfigLoader

    _write_project(
        tmp_path,
        {"models": {"clinical_model": {"name": "gpt-4.1"}},
         "agents": {"medical": {"enabled": True}}},
        {"medical": {"name": "medical", "model": "clinical_model", "prompt": "nope"}},
    )
    loader = AgentConfigLoader(
        Config(), base_dir=tmp_path, known_agents={"medical"},
    )
    with pytest.raises(ConfigError) as exc:
        loader.load()
    assert "nope" in str(exc.value)


def test_loader_rejects_duplicate_agent_definition(tmp_path):
    from config.agent_config_loader import AgentConfigLoader

    _write_project(
        tmp_path,
        {"models": {}, "agents": {"soap": {"enabled": True}}},
        {"a": {"name": "soap"}, "b": {"name": "soap"}},
    )
    loader = AgentConfigLoader(
        Config(), base_dir=tmp_path, known_agents={"soap"},
    )
    with pytest.raises(ConfigError) as exc:
        loader.load()
    assert "soap" in str(exc.value)


def test_loader_requires_config_for_enabled_agent(tmp_path):
    from config.agent_config_loader import AgentConfigLoader

    _write_project(
        tmp_path,
        {"models": {}, "agents": {"soap": {"enabled": True}}},
        {},
    )
    loader = AgentConfigLoader(
        Config(), base_dir=tmp_path, known_agents={"soap"},
    )
    with pytest.raises(ConfigError) as exc:
        loader.load()
    assert "soap" in str(exc.value)
    assert "config.yaml" in str(exc.value)


def test_loader_skips_disabled_agents(tmp_path):
    from config.agent_config_loader import AgentConfigLoader

    _write_project(
        tmp_path,
        {"models": {}, "agents": {"soap": {"enabled": False}}},
        {},
    )
    loader = AgentConfigLoader(
        Config(), base_dir=tmp_path, known_agents={"soap"},
    )
    resolved = loader.load()
    assert resolved.enabled_agents == {}


# ---------------------------------------------------------------------------
# AgentFactory
# ---------------------------------------------------------------------------

def test_factory_uses_configured_model_prompt_and_tools():
    from agent.session import Session
    from customagents.factory import AgentFactory

    loaded = load_config(cwd=PROJECT_ROOT)
    session = Session(loaded)

    agent = AgentFactory.create("medical", loaded, session=session)

    assert agent.config.model_name == "gpt-4.1"
    assert agent.config.model.temperature == 0.1
    assert agent.system_prompt == PromptRegistry.from_builtin().get("medical")
    assert set(agent.allowed_tool_names) == {"memory", "wiki_search", "neo4j_query"}
    assert {t.name for t in agent.configured_tools} == {
        "memory", "wiki_search", "neo4j_query",
    }


def test_factory_does_not_override_agent_own_tool_restriction():
    from agent.session import Session
    from customagents.factory import AgentFactory

    loaded = load_config(cwd=PROJECT_ROOT)
    session = Session(loaded)
    agent = AgentFactory.create("document_retrieval", loaded, session=session)

    # document_retrieval has no `tools` key, so its constructor's own
    # restriction survives.
    assert agent.allowed_tool_names == [
        "jina_embedding", "medical_document_search",
    ]


def test_factory_empty_resolution_falls_back_to_defaults():
    from config.agent_config_loader import ResolvedAgentConfig
    from customagents.factory import AgentFactory

    # An empty resolution means no agent configuration at all.
    agent = AgentFactory.create(
        "consult", Config(), session=None, resolved=ResolvedAgentConfig(),
    )
    assert agent.system_prompt
    assert agent.allowed_tool_names is None


def test_factory_unknown_tool_raises_clearly():
    from customagents.factory import AgentFactory

    config = Config(
        models=ModelsSection(definitions={
            "clinical_model": ModelConfig(name="gpt-4.1"),
        }),
    )
    agent_yaml = AgentYaml(name="medical", tools=["does_not_exist"])
    with pytest.raises(ConfigError) as exc:
        AgentFactory._resolve_tools("medical", agent_yaml, config, session=None)
    assert "medical" in str(exc.value)
    assert "does_not_exist" in str(exc.value)


def test_factory_unknown_agent_type_raises(config):
    from customagents.factory import AgentFactory

    with pytest.raises(ValueError):
        AgentFactory.create("does_not_exist", config, session=None)


def test_configured_tools_resolve_through_existing_registry(config):
    registry = create_default_registry(config)
    assert registry.get("memory") is not None
    assert registry.get("wiki_search") is not None
    assert registry.get("neo4j_query") is not None