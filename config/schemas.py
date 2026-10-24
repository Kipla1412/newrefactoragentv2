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
        skill: Logical skill name activated by default for this agent,
            resolved by the SkillRegistry. None means the agent only uses a
            skill when the caller explicitly asks for one.
        agent_class: Name of the registered agent class that builds this agent,
            for config-driven agents that are not registered under their own
            name (e.g. several agents sharing GenericAgent). When None the
            agent is registered under its own ``name``.
        agent_name: Human-readable name used for MLflow tagging and error
            messages. Only meaningful for config-driven agents that share a
            class (see GenericAgent); other agents default to their class name.
        tools: Logical tool names resolved by the existing ToolRegistry.
            None leaves the agent's tool set unrestricted; an empty list
            means the agent uses no tools.
        settings: Optional framework-level settings.
    """

    name: str
    model: str | None = None
    prompt: str | None = None
    skill: str | None = None
    agent_class: str | None = None
    agent_name: str | None = None
    tools: list[str] | None = None
    settings: AgentSettings | None = None


class ApplicationAgentConfig(BaseModel):
    """Schema for one entry in the application-level ``agents.yaml``.

    Attributes:
        enabled: Whether the agent is available in this application.
    """

    enabled: bool = True