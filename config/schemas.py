"""Pydantic schemas for the two-level agent configuration.

These schemas validate the agent-specific ``config.yaml`` files. They
define registry *references* — never implementations. Model, prompt,
and tool content all live in their respective registries.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentSettings(BaseModel):
    """Optional framework-level agent settings.

    Only settings with a clear framework purpose are supported. Agent
    behavior stays in the agent class; this is limited to knobs the
    framework itself consumes.

    Attributes:
        max_turns: Maximum agentic-loop turns for this agent.
        temperature: Sampling temperature override for this agent.
    """

    max_turns: int | None = None
    temperature: float | None = None


class AgentYaml(BaseModel):
    """Schema for an agent-specific ``config.yaml``.

    Attributes:
        name: Logical agent name matching the AgentType string.
        model: Logical model name resolved by the ModelRegistry.
        prompt: Logical prompt name resolved by the PromptRegistry.
        tools: Logical tool names resolved by the existing ToolRegistry.
            None leaves the agent's tool set unrestricted; an empty list
            means the agent uses no tools.
        settings: Optional framework-level settings.
    """

    name: str
    model: str | None = None
    prompt: str | None = None
    tools: list[str] | None = None
    settings: AgentSettings | None = None


class ApplicationAgentConfig(BaseModel):
    """Schema for one entry in the application-level ``agents.yaml``.

    Attributes:
        enabled: Whether the agent is available in this application.
    """

    enabled: bool = True