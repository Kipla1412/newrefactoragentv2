"""Regression tests for CLI agent construction.

``main.py`` used to build ``Agent(config, system_prompt)`` while ``Agent``
requires an ``agent_type``, so both CLI modes raised
``TypeError: __init__() missing 1 required positional argument: 'agent_type'``
on startup. The subagent tool had the same bug. These tests pin both call sites.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent.agent import Agent
from agent.events import AgentType
from config.config import Config
from tools.base import ToolInvocation
from tools.subagents import SubagentDefinition, SubagentTool


def test_cli_builds_a_generic_agent(config):
    from main import CLI

    cli = CLI(config)

    agent = cli._build_agent("You are a test assistant.")

    assert isinstance(agent, Agent)
    assert agent.agent_type == AgentType.GENERIC
    assert agent.system_prompt == "You are a test assistant."
    assert agent.session is not None


def test_cli_passes_the_confirmation_callback_through(config):
    from main import CLI

    cli = CLI(config)

    def sentinel(confirmation) -> bool:
        return True

    agent = cli._build_agent("prompt", confirmation_callback=sentinel)

    assert agent.session.approval_manager.confirmation_callback is sentinel


def test_importing_main_does_not_run_the_cli():
    """``main()`` must stay behind an ``__main__`` guard."""
    text = (Path(__file__).resolve().parent.parent / "main.py").read_text(
        encoding="utf-8"
    )

    assert 'if __name__ == "__main__":' in text
    assert text.rstrip().endswith("main()")


def test_subagent_tool_builds_an_agent_with_its_prompt(monkeypatch):
    """The subagent tool used to call ``Agent(subagent_config)`` — one argument."""
    captured: dict = {}

    class _StubAgent:
        def __init__(self, *args, **kwargs):
            captured["args"] = args

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def run(self, message):
            if False:  # pragma: no cover - keeps this an async generator
                yield

    # ``execute`` imports Agent lazily, so patching the module attribute works.
    monkeypatch.setattr("agent.agent.Agent", _StubAgent)

    tool = SubagentTool(
        config=Config(),
        definition=SubagentDefinition(
            name="probe", description="probe", goal_prompt="Probe the repo."
        ),
    )

    result = asyncio.run(
        tool.execute(ToolInvocation(params={"goal": "look around"}, cwd=Path(".")))
    )

    args = captured["args"]
    assert len(args) >= 3, f"Agent got only {len(args)} argument(s)"
    assert args[1], "no system prompt was passed"
    assert args[2] == AgentType.GENERIC
    assert result is not None
