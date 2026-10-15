"""Tests for application-level agent gating.

Verifies that the ``enabled`` flags in ``agents.yaml`` actually gate the
application: the bootstrap reports the enabled set, routes are mounted
conditionally, and the factory refuses to build disabled agents.
"""

from pathlib import Path

import pytest
import yaml

from config.agent_config_loader import AgentConfigLoader, ResolvedAgentConfig
from config.config import AgentsSection, Config
from config.loader import load_config
from config.schemas import ApplicationAgentConfig
from utils.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_project(tmp_path: Path, agents: dict, agent_configs: dict):
    (tmp_path / "customagents").mkdir(parents=True)
    (tmp_path / "agents.yaml").write_text(
        yaml.safe_dump({"models": {}, "agents": agents})
    )
    for name, data in agent_configs.items():
        d = tmp_path / "customagents" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.yaml").write_text(yaml.safe_dump(data))


# ---------------------------------------------------------------------------
# Bootstrap / loader
# ---------------------------------------------------------------------------

def test_bootstrap_reports_enabled_agents():
    from config.application_bootstrap import RuntimeState, bootstrap_application
    from customagents.factory import AgentFactory

    state = bootstrap_application(
        base_dir=PROJECT_ROOT,
        known_agents=AgentFactory.known_agent_names(),
    )

    assert isinstance(state, RuntimeState)
    assert state.enabled_agents  # non-empty
    # Enabled set matches agents.yaml exactly.
    config = load_config(cwd=PROJECT_ROOT)
    assert set(state.enabled_agents) == {
        name for name, entry in config.agents.definitions.items() if entry.enabled
    }
    assert state.session_manager is not None
    assert state.model_registry is not None
    assert state.prompt_registry is not None


def test_loader_excludes_disabled_agents(tmp_path):
    _write_project(
        tmp_path,
        {"a": {"enabled": True}, "b": {"enabled": False}},
        {"a": {"name": "a"}, "b": {"name": "b"}},
    )
    loader = AgentConfigLoader(Config(), base_dir=tmp_path, known_agents={"a", "b"})
    resolved = loader.load()

    assert set(resolved.enabled_agents) == {"a"}


# ---------------------------------------------------------------------------
# Conditional route mounting
# ---------------------------------------------------------------------------

def _route_paths(app) -> set[str]:
    """Return the app's route paths via its OpenAPI schema.

    FastAPI may wrap included routers lazily (``_IncludedRouter``) so
    ``app.routes`` is not reliable across versions; the OpenAPI schema is.
    """
    from fastapi.testclient import TestClient

    return set(TestClient(app).get("/openapi.json").json()["paths"])


def test_mount_agent_routers_mounts_only_enabled():
    from fastapi import FastAPI

    from api.main import _mount_agent_routers

    app = FastAPI()
    _mount_agent_routers(
        app, {"doc", "soap", "assessment", "clinical_extraction"}
    )
    paths = _route_paths(app)

    assert "/api/agent/doctoragent" in paths
    assert "/api/agent/soap" in paths
    assert "/api/agent/assessment" in paths
    assert "/api/agent/clinical-extraction" in paths
    assert "/api/agent/consultation-workflow" in paths  # all three required agents on
    assert "/api/agent/mcpagent" not in paths           # ehr not enabled
    assert "/api/agent/document" not in paths           # document_retrieval not enabled


def test_composite_workflow_requires_all_three_agents():
    from fastapi import FastAPI

    from api.main import _mount_agent_routers

    app = FastAPI()
    _mount_agent_routers(app, {"soap", "assessment"})  # clinical_extraction missing
    assert "/api/agent/consultation-workflow" not in _route_paths(app)
    assert "/api/agent/soap" in _route_paths(app)


def test_create_app_gates_disabled_agents(tmp_path):
    """End-to-end: a disabled agent's routes are absent from the app."""
    from api.main import create_app
    from customagents.factory import AgentFactory
    from config.application_bootstrap import ApplicationBootstrap

    _write_project(
        tmp_path,
        {"doc": {"enabled": True}, "soap": {"enabled": False}},
        {"doc": {"name": "doc", "prompt": "doc"}, "soap": {"name": "soap", "prompt": "soap"}},
    )
    state = ApplicationBootstrap(
        base_dir=tmp_path, known_agents=AgentFactory.known_agent_names(),
    ).run()
    app = create_app(state)
    paths = _route_paths(app)

    assert state.enabled_agents == ("doc",)
    assert "/api/agent/doctoragent" in paths
    assert "/api/agent/soap" not in paths


# ---------------------------------------------------------------------------
# Factory enforcement
# ---------------------------------------------------------------------------

def test_factory_rejects_disabled_agent():
    from customagents.factory import AgentFactory

    config = Config(
        agents=AgentsSection(definitions={
            "soap": ApplicationAgentConfig(enabled=False),
        }),
    )
    with pytest.raises(ConfigError) as exc:
        AgentFactory.create(
            "soap", config, session=None, resolved=ResolvedAgentConfig(),
        )
    assert "soap" in str(exc.value)
    assert "disabled" in str(exc.value)


def test_factory_allows_agent_with_no_application_entry():
    from customagents.factory import AgentFactory

    # No application-level entry at all → backward-compatible defaults.
    agent = AgentFactory.create(
        "soap", Config(), session=None, resolved=ResolvedAgentConfig(),
    )
    assert agent.system_prompt
    assert agent.allowed_tool_names is None


def test_factory_creates_enabled_agent():
    from agent.session import Session
    from config.application_bootstrap import ApplicationBootstrap
    from customagents.factory import AgentFactory

    state = ApplicationBootstrap(
        base_dir=PROJECT_ROOT, known_agents=AgentFactory.known_agent_names(),
    ).run()
    session = Session(state.config)

    agent = AgentFactory.create(
        "doc", state.config, session=session, resolved=state.agent_config,
    )
    assert agent is not None


# ---------------------------------------------------------------------------
# Router requirements table
# ---------------------------------------------------------------------------

def test_router_requirements_cover_all_gated_agents():
    from api.main import ROUTER_AGENT_REQUIREMENTS

    required = {agent for _router, agents in ROUTER_AGENT_REQUIREMENTS for agent in agents}
    assert {"doc", "soap", "assessment", "clinical_extraction",
            "document_retrieval", "pdf_chat", "ehr"}.issubset(required)
    # The composite workflow requires exactly the three clinical agents.
    workflow_reqs = {
        tuple(sorted(agents)) for router, agents in ROUTER_AGENT_REQUIREMENTS
        if getattr(router, "prefix", None) == "/agent" and len(agents) == 3
    }
    assert ("assessment", "clinical_extraction", "soap") in workflow_reqs