"""Tests for the patient demographics agent.

Covers the integration end to end: skill loading, the agent resolving through
AgentFactory, ContextManager delivery of the skill instructions, FHIR tool
execution through the ToolRegistry, partial results when a tool fails, and the
instruction contract for identity / missing-data / output rules.

The tests are explicit about what they do and do not prove: model *behaviour*
cannot be tested without a live LLM, so the clinical rules are asserted against
the skill instructions that the model receives (see the "instruction contract"
section).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.events import AgentEventType, AgentType
from client.response import StreamEvent, StreamEventType, TextDelta, ToolCall, TokenUsage
from config.agent_config_loader import AgentConfigLoader, ResolvedAgentConfig
from config.config import Config
from config.schemas import AgentYaml
from context.manager import ContextManager
from customagents.demographicsagent.demographicsagent import DemographicsAgent
from customagents.demographicsagent.demographicsprompt import DEMOGRAPHICS_PROMPT
from registries.model_registry import ModelRegistry
from registries.prompt_registry import PromptRegistry
from skills import default_skill_registry
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILL = "patient-demographics-summary"
AGENT = "demographics"


@pytest.fixture(autouse=True)
def _reset_factory_components():
    from customagents.factory import AgentFactory

    AgentFactory.reset_components()
    yield
    AgentFactory.reset_components()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeFhirTool(Tool):
    """Stands in for an MCP tool registered as "<server>__<tool>"."""

    kind = ToolKind.MCP

    def __init__(self, config, name: str, payload: str = "", error: str | None = None):
        super().__init__(config)
        self.name = name
        self.description = "fake FHIR tool"
        self.calls: list[dict] = []
        self._payload = payload
        self._error = error

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": [],
        }

    def is_mutating(self, params) -> bool:
        # Mirrors MCPTool, which reports every call as mutating.
        return True

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        self.calls.append(invocation.params)
        if self._error:
            return ToolResult.error_result(self._error)
        return ToolResult.success_result(self._payload)


def _stub_session(config, llm):
    """Build a Session with a stubbed LLM, capturing every request."""
    from agent.session import Session

    session = Session(config)
    session.context_manager = ContextManager(config, user_memory=None, tools=[])
    captured: list[list[dict]] = []

    async def chat(messages, model=None, tools=None, tool_choice=None, **kwargs):
        captured.append(messages)
        async for event in llm(len(captured)):
            yield event

    session.client.chat_completion = chat
    return session, captured


def _text_only(payload: str):
    async def llm(_turn):
        yield StreamEvent(
            type=StreamEventType.TEXT_DELTA, text_delta=TextDelta(content=payload)
        )
        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason="stop",
            usage=TokenUsage(),
        )

    return llm


def _tool_then_text(tool_name: str, payload: str, tool_payload: str = "{}"):
    """First turn requests an FHIR tool; second turn returns the summary."""
    async def llm(turn):
        if turn == 1:
            yield StreamEvent(
                type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call=ToolCall(
                    call_id="call-1", name=tool_name, arguments={"id": "10008"}
                ),
            )
            yield StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                finish_reason="tool_calls",
                usage=TokenUsage(),
            )
        else:
            yield StreamEvent(
                type=StreamEventType.TEXT_DELTA, text_delta=TextDelta(content=payload)
            )
            yield StreamEvent(
                type=StreamEventType.MESSAGE_COMPLETE,
                finish_reason="stop",
                usage=TokenUsage(),
            )

    return llm


def _build(config, llm, *, skill: str | None = SKILL):
    """Build the demographics agent the way the factory does."""
    from customagents.factory import AgentFactory

    session, captured = _stub_session(config, llm)
    resolved = ResolvedAgentConfig(
        enabled_agents={
            AGENT: AgentYaml(
                name=AGENT,
                prompt="demographics",
                skill=skill,
            )
        },
        model_registry=ModelRegistry({}),
        prompt_registry=PromptRegistry.from_builtin(),
    )
    agent = AgentFactory.create(AGENT, config, session, resolved=resolved)
    return agent, session, captured


def _drive(agent, message="Summarise patient 10008", skill=None):
    async def _run():
        return [event async for event in agent.run(message, skill=skill)]

    return asyncio.run(_run())


def _skill_text() -> str:
    """Skill instructions with whitespace collapsed.

    Assertions run against this so they survive the skill's line wrapping.
    """
    return " ".join(default_skill_registry().load_instructions(SKILL).split())


# ---------------------------------------------------------------------------
# Skill loading
# ---------------------------------------------------------------------------


def test_demographics_skill_is_discovered():
    registry = default_skill_registry()

    assert registry.has(SKILL)

    metadata = registry.get_metadata(SKILL)
    assert metadata.name == SKILL
    assert metadata.description.strip()
    assert metadata.path.is_file()


def test_skill_frontmatter_has_no_extra_requirements():
    """Discovery already validates name/description; this pins the contract."""
    skill = default_skill_registry().load(SKILL)

    assert skill.instructions.strip()
    assert skill.name == SKILL


def test_skill_covers_all_three_resource_types():
    text = _skill_text()

    for resource in ("Patient", "RelatedPerson", "Coverage"):
        assert resource in text


def test_skill_requires_the_agreed_patient_fields():
    text = _skill_text()

    for field in (
        "name",
        "birthDate",
        "gender",
        "address",
        "telecom",
        "communication",
        "identifier",
        "extension",
    ):
        assert field in text


def test_skill_requires_related_person_fields():
    text = _skill_text()

    for field in ("relationship", "period"):
        assert field in text


# ---------------------------------------------------------------------------
# Instruction contract (what the model is told)
# ---------------------------------------------------------------------------


def test_skill_forbids_inventing_or_inferring_data():
    text = _skill_text()

    assert "Never invent, guess, infer" in text
    assert "traceable to retrieved data" in text


def test_skill_forbids_assuming_emergency_contact():
    text = _skill_text()

    assert "Do not label anyone an emergency contact" in text
    assert "a phone number alone is not" in text


def test_skill_forbids_assuming_active_coverage():
    text = _skill_text()

    assert "Do not state that coverage is active" in text
    assert "not, by itself, a statement of active coverage" in text


def test_skill_requires_patient_identity_verification():
    text = _skill_text()

    # Must stop on a mismatch rather than summarising the wrong person.
    assert "does not correspond to the requested identifier" in text


def test_skill_requires_a_missing_data_section_distinguishing_failure_modes():
    text = _skill_text()

    assert "Missing or Unavailable" in text
    assert "not present" in text
    assert "not retrieved" in text
    assert "retrieval failed" in text


def test_skill_requires_attempting_retrieval_before_reporting_no_data():
    text = _skill_text()

    assert "at least one retrieval attempt is mandatory" in text.lower()


def test_skill_handles_invalid_identifiers_without_falling_back_to_a_search():
    text = _skill_text()

    assert "Never fall back to a search" in text


def test_skill_states_privacy_rules():
    text = _skill_text()

    assert "PRIVACY AND AUTHORIZATION" in text
    assert "authorization" in text.lower()


# ---------------------------------------------------------------------------
# Agent configuration and construction
# ---------------------------------------------------------------------------


def test_demographics_agent_is_registered_as_its_own_class():
    from customagents.factory import AgentFactory

    assert AgentType.DEMOGRAPHICS.value == AGENT
    assert AGENT in AgentFactory.known_agent_names()
    assert AgentFactory.components().types.get(AGENT) is DemographicsAgent


def test_shipped_config_points_at_its_prompt_and_skill():
    loader = AgentConfigLoader(Config(), base_dir=PROJECT_ROOT)
    agent_yaml = loader.load_agent(AGENT)

    assert agent_yaml is not None
    # Registered under its own name, like the other agents.
    assert agent_yaml.agent_class is None
    assert agent_yaml.agent_name is None
    assert agent_yaml.prompt == "demographics"
    assert agent_yaml.skill == SKILL
    # Omitted on purpose: a static list would hide the runtime FHIR tools.
    assert agent_yaml.tools is None


def test_demographics_prompt_is_registered():
    assert PromptRegistry.from_builtin().has("demographics")


def test_agent_builds_through_the_factory(config):
    agent, _session, _captured = _build(config, _text_only("ok"))

    assert isinstance(agent, DemographicsAgent)
    assert agent.system_prompt == DEMOGRAPHICS_PROMPT
    assert agent.default_skill == SKILL
    assert agent.agent_type == AgentType.DEMOGRAPHICS


def test_demographics_route_is_mounted():
    import api.main as main

    paths = main.app.openapi()["paths"]

    assert "/api/agent/demographics" in paths
    schema = paths["/api/agent/demographics"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    name = schema.rsplit("/", 1)[-1]
    properties = main.app.openapi()["components"]["schemas"][name]["properties"]
    assert set(properties) >= {"message", "session_id", "skill"}


# ---------------------------------------------------------------------------
# ContextManager integration
# ---------------------------------------------------------------------------


def test_skill_instructions_reach_the_llm(config):
    agent, session, captured = _build(config, _text_only("summary"))

    _drive(agent)

    assert captured, "the LLM was never called"
    system = captured[0][0]
    assert system["role"] == "system"
    # The skill's workflow and the agent's role prompt both arrive.
    assert "RETRIEVAL WORKFLOW" in system["content"]
    assert "patient demographics assistant" in system["content"].lower()

    # Scoped to the request.
    assert session.context_manager.skill_instructions is None


def test_no_skill_means_no_skill_section(config):
    agent, _session, captured = _build(config, _text_only("summary"), skill=None)

    _drive(agent)

    content = captured[0][0]["content"]
    assert "RETRIEVAL WORKFLOW" not in content
    assert "# Active Skill" not in content
    # The role prompt still states the hard rules.
    assert "never invent" in content.lower()


def test_unregistered_skill_is_reported_not_ignored(config):
    agent, _session, captured = _build(config, _text_only("summary"), skill="nope")

    events = _drive(agent)

    errors = [e for e in events if e.type == AgentEventType.AGENT_ERROR]
    assert errors
    assert "nope" in errors[0].data["error"]
    assert captured == []


# ---------------------------------------------------------------------------
# FHIR tool execution through the ToolRegistry
# ---------------------------------------------------------------------------


def test_fhir_tool_is_executed_and_its_result_fed_back(config):
    tool = None
    agent, session, captured = _build(
        config, _tool_then_text("fhir__get_patient", "Patient: Jane Doe")
    )
    tool = _FakeFhirTool(config, "fhir__get_patient", payload='{"name": "Jane Doe"}')
    session.tool_registry.register_mcp(tool)

    events = _drive(agent)

    # The tool ran, with the arguments the model asked for.
    assert tool.calls == [{"id": "10008"}]

    # The loop reported it.
    tool_events = [e for e in events if e.type == AgentEventType.TOOL_CALL_COMPLETE]
    assert tool_events

    # Its output went back to the model on the next turn.
    assert len(captured) == 2
    tool_messages = [m for m in captured[1] if m.get("role") == "tool"]
    assert tool_messages, "the tool result was not fed back to the model"
    assert "Jane Doe" in tool_messages[-1]["content"]

    # And the agent produced its summary.
    assert any(e.type == AgentEventType.TEXT_COMPLETE for e in events)


def test_fhir_tool_failure_produces_partial_results_not_a_crash(config):
    agent, session, captured = _build(
        config, _tool_then_text("fhir__get_patient", "Patient: unknown")
    )
    tool = _FakeFhirTool(
        config, "fhir__get_patient", error="FHIR server unavailable"
    )
    session.tool_registry.register_mcp(tool)

    events = _drive(agent)

    # The failure surfaced to the model rather than aborting the turn.
    assert tool.calls
    tool_messages = [m for m in captured[1] if m.get("role") == "tool"]
    assert tool_messages
    assert "FHIR server unavailable" in tool_messages[-1]["content"]

    # A summary was still produced (partial results).
    assert any(e.type == AgentEventType.TEXT_COMPLETE for e in events)


def test_unknown_tool_does_not_break_the_turn(config):
    agent, _session, captured = _build(
        config, _tool_then_text("fhir__not_registered", "partial summary")
    )

    events = _drive(agent)

    tool_messages = [m for m in captured[1] if m.get("role") == "tool"]
    assert tool_messages
    assert "Unknown tool" in tool_messages[-1]["content"]
    assert any(e.type == AgentEventType.TEXT_COMPLETE for e in events)


def test_mcp_style_tool_calls_are_not_blocked_by_approval(config):
    """MCPTool reports every call as mutating; the policy still lets them run."""
    agent, session, _captured = _build(
        config, _tool_then_text("fhir__get_patient", "done")
    )
    tool = _FakeFhirTool(config, "fhir__get_patient", payload="{}")
    session.tool_registry.register_mcp(tool)

    _drive(agent)

    assert tool.calls == [{"id": "10008"}]


# ---------------------------------------------------------------------------
# Skill scoping across requests
# ---------------------------------------------------------------------------

def test_skill_does_not_leak_into_a_followup_request(config):
    agent, session, captured = _build(config, _text_only("summary"))

    _drive(agent)
    assert "RETRIEVAL WORKFLOW" in captured[0][0]["content"]

    _drive(agent, "and their coverage?")

    assert session.context_manager.skill_instructions is None
    assert "# Active Skill" in captured[1][0]["content"]  # default skill again


# ---------------------------------------------------------------------------
# Route: per-request isolation
# ---------------------------------------------------------------------------


def test_route_clears_context_before_every_request(monkeypatch):
    """A reused session_id must not carry the previous patient's data."""
    from types import SimpleNamespace

    import api.routers.demographics as mod

    class _FakeContext:
        def __init__(self):
            self.cleared = 0

        def clear(self):
            self.cleared += 1

    class _FakeMcp:
        def __init__(self):
            self.tokens = []
            self.registered = False

        async def initialize(self, auth_token=None):
            self.tokens.append(auth_token)

        def register_tools(self, registry):
            self.registered = True

    class _FakeSession:
        def __init__(self):
            self.context_manager = _FakeContext()
            self.mcp_manager = _FakeMcp()
            self.tool_registry = object()
            self.auth_token = None
            self.initialized = 0

        async def initialize(self):
            self.initialized += 1

    session = _FakeSession()
    seen: dict[str, int] = {}

    class _FakeAgent:
        async def run(self, message, skill=None):
            seen["cleared_before_run"] = session.context_manager.cleared
            return
            yield  # pragma: no cover - makes this an async generator

    class _SessionManager:
        async def get_session(self, user_id, config, session_id=None):
            return session

    class _App:
        state = SimpleNamespace(config=Config(), session_manager=_SessionManager())

    request = SimpleNamespace(
        app=_App(),
        state=SimpleNamespace(user={"sub": "user-1"}, token="bearer-token"),
    )

    monkeypatch.setattr(
        mod, "AgentFactory", SimpleNamespace(create=lambda *a, **k: _FakeAgent())
    )

    response = asyncio.run(
        mod.demographics_api(mod.DemographicsRequest(message="hi"), request)
    )

    # The agent only runs when the streaming response is consumed.
    async def _drain():
        async for _chunk in response.body_iterator:
            pass

    asyncio.run(_drain())

    # Context is reset, and reset before the agent runs.
    assert session.context_manager.cleared == 1
    assert seen["cleared_before_run"] == 1

    # The caller's token still reaches the MCP connection.
    assert session.auth_token == "bearer-token"
    assert session.mcp_manager.tokens == ["bearer-token"]
    assert session.mcp_manager.registered is True
