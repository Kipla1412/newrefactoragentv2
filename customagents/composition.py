"""Composition-layer collaborators for :class:`AgentFactory`.

Every class here has a single reason to change, so the factory can stay a
thin orchestrator and each concern can be substituted independently:

============================  ============================================
Collaborator                  Responsibility (SRP)
============================  ============================================
AgentTypeRegistry             logical name → agent class
AgentConfigProvider           name → (AgentYaml, ModelRegistry), enabled-aware
ModelResolver                 AgentYaml.model → derived Config
PromptResolver                AgentYaml.prompt → prompt content
ToolResolver                  AgentYaml.tools → Tool instances
AgentSettingsApplier          AgentYaml.settings → derived Config
AgentOverrideApplier          resolved values → constructed agent
AgentComponents               bundle injected into AgentFactory (DIP)
============================  ============================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from config.config import Config
from config.schemas import AgentYaml
from registries.model_registry import ModelRegistry
from registries.prompt_registry import PromptRegistry
from utils.errors import ConfigError


# Agent type registry

class AgentTypeRegistry:
    """Map logical agent names to their implementation classes.

    SRP: the only place that knows which class backs which name.
    OCP: new agents are added with :meth:`register`; the factory and its
    callers are not modified.

    Attributes:
        _agents: Lower-cased name → agent class.
    """

    def __init__(self, agents: Mapping[str, type] | None = None) -> None:
        """Initialize the registry.

        Args:
            agents: Optional initial mapping of name to agent class.
        """
        self._agents: dict[str, type] = {
            name.lower(): agent_class for name, agent_class in (agents or {}).items()
        }

    def register(self, name: str, agent_class: type) -> None:
        """Register (or replace) the class for an agent name.

        Args:
            name: Logical agent name.
            agent_class: The concrete Agent subclass.
        """
        self._agents[name.lower()] = agent_class

    def get(self, name: str) -> type | None:
        """Return the class for an agent name, or None.

        Args:
            name: Logical agent name.

        Returns:
            The agent class, or None when unregistered.
        """
        return self._agents.get(name.lower())

    def known_names(self) -> set[str]:
        """Return every registered agent name.

        Returns:
            The set of registered names.
        """
        return set(self._agents)

    def mapping(self) -> Mapping[str, type]:
        """Return a copy of the name → class mapping.

        Returns:
            A shallow copy of the registry contents.
        """
        return dict(self._agents)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._agents

# Config provider

class AgentConfigProvider:
    """Provide the resolved agent configuration, honouring ``enabled``.

    SRP: turns a name into ``(AgentYaml | None, ModelRegistry | None)``.
    DIP: the loader is injected (or built from ``known_agents``), so no
    module-level global is required and tests can supply their own.
    """

    def __init__(
        self,
        known_agents: set[str] | None = None,
        loader_factory: Callable[[Config, set[str]], Any] | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            known_agents: Agent names registered in the type registry.
            loader_factory: Optional factory ``(config, known_agents) -> loader``
                used to build the agent config loader lazily.
        """
        self._known_agents = set(known_agents or ())
        self._loader_factory = loader_factory
        self._loader: Any | None = None

    def _get_loader(self, config: Config):
        """Return the cached loader, building it on first use."""
        if self._loader is None:
            factory = self._loader_factory or self._default_loader_factory
            self._loader = factory(config, self._known_agents)
            self._loader.load()  # populate the discovery cache once
        return self._loader

    @staticmethod
    def _default_loader_factory(config: Config, known_agents: set[str]):
        """Build the framework's standard AgentConfigLoader."""
        from config.agent_config_loader import AgentConfigLoader

        return AgentConfigLoader(config, known_agents=known_agents)

    def provide(
        self,
        name: str,
        config: Config,
        resolved: Any | None = None,
    ) -> tuple[AgentYaml | None, ModelRegistry | None]:
        """Return the agent's resolved configuration and model registry.

        Args:
            name: Lower-cased agent identifier.
            config: Runtime configuration.
            resolved: Optional pre-resolved ResolvedAgentConfig.

        Returns:
            The agent's AgentYaml (or None when it has no application-level
            entry) and the ModelRegistry to resolve its model against.

        Raises:
            ConfigError: If the agent is explicitly disabled in agents.yaml.
        """
        enabled_agents: dict[str, AgentYaml] = {}
        model_registry: ModelRegistry | None = None

        if resolved is not None:
            enabled_agents = resolved.enabled_agents
            model_registry = resolved.model_registry
        else:
            loader = self._get_loader(config)
            if loader.resolved is not None:
                enabled_agents = loader.resolved.enabled_agents
                model_registry = loader.resolved.model_registry

        if name in enabled_agents:
            return enabled_agents[name], model_registry

        # Distinguish "explicitly disabled" from "not configured at all".
        app_entry = config.resolve_agent_config(name)
        if app_entry is not None and not app_entry.enabled:
            raise ConfigError(f"Agent '{name}' is disabled in agents.yaml")

        return None, model_registry


# Model / prompt / settings resolution


class ModelResolver:
    """Resolve an agent's logical model onto a derived Config copy.

    SRP: given an AgentYaml and a Config, return the Config the agent
    should run with. Never mutates the input configuration.
    """

    def resolve(
        self,
        agent_yaml: AgentYaml | None,
        config: Config,
        model_registry: ModelRegistry | None = None,
    ) -> Config:
        """Return a config with the agent's model applied.

        Args:
            agent_yaml: The agent's resolved configuration, or None.
            config: Runtime configuration.
            model_registry: Optional registry to resolve the logical name.

        Returns:
            A derived Config (or the original when no model is configured
            or the referenced model is unregistered).
        """
        if not agent_yaml or not agent_yaml.model:
            return config

        registry = model_registry or ModelRegistry(config.models.definitions)
        if not registry.has(agent_yaml.model):
            return config
        return registry.apply(agent_yaml.model, config)


class PromptResolver:
    """Resolve an agent's logical prompt name to its content.

    SRP: prompt selection only. The registry is injected so alternative
    prompt sources can be substituted (DIP).
    """

    def __init__(self, registry_factory: Callable[[], PromptRegistry] | None = None) -> None:
        """Initialize the resolver.

        Args:
            registry_factory: Factory returning the prompt registry.
                Defaults to the built-in registry.
        """
        self._registry_factory = registry_factory or PromptRegistry.from_builtin

    def resolve(
        self,
        agent_name: str,
        agent_yaml: AgentYaml | None,
        config: Config,
    ) -> str | None:
        """Return the resolved prompt content, or None.

        Args:
            agent_name: Agent identifier used in validation messages.
            agent_yaml: The agent's resolved configuration, or None.
            config: Runtime configuration (used to validate the reference).

        Returns:
            The prompt content, or None when the agent has no prompt entry.
        """
        if not agent_yaml or not agent_yaml.prompt:
            return None
        config.resolve_prompt_definition(agent_name, agent_yaml.prompt)
        return self._registry_factory().get(agent_yaml.prompt)


class AgentSettingsApplier:
    """Apply ``AgentYaml.settings`` onto a derived Config copy.

    SRP: framework-level overrides (``max_turns``, ``temperature``) only.
    """

    def apply(self, config: Config, agent_yaml: AgentYaml | None) -> Config:
        """Return a config with the agent's settings applied.

        Args:
            config: Runtime configuration.
            agent_yaml: The agent's resolved configuration, or None.

        Returns:
            A derived Config when settings are present, else the original.
        """
        settings = getattr(agent_yaml, "settings", None) if agent_yaml else None
        if not settings:
            return config
        if settings.max_turns is None and settings.temperature is None:
            return config

        derived = config.model_copy(deep=True)
        if settings.max_turns is not None:
            derived.max_turns = settings.max_turns
        if settings.temperature is not None:
            derived.model.temperature = settings.temperature
        return derived


# Tool resolution

class ToolExpander(Protocol):
    """Expands a pseudo tool name into concrete tools."""

    def __call__(self, tool_registry: Any) -> list:
        """Return the tools a pseudo name stands for."""
        ...


def expand_mcp_tools(tool_registry: Any) -> list:
    """Expand ``mcp`` to every MCP-kind tool currently registered.

    Args:
        tool_registry: The existing ToolRegistry.

    Returns:
        The MCP-kind tools.
    """
    return [
        tool
        for tool in tool_registry.get_tools()
        if getattr(getattr(tool, "kind", None), "value", None) == "mcp"
    ]


class PseudoToolRegistry:
    """Map pseudo tool names (e.g. ``mcp``) to their expanders.

    OCP: a new pseudo tool is supported by registering an expander, not by
    editing :class:`ToolResolver`.
    """

    def __init__(self, expanders: Mapping[str, ToolExpander] | None = None) -> None:
        """Initialize with an optional expander mapping.

        Args:
            expanders: Mapping of pseudo name to expander.
        """
        self._expanders: dict[str, ToolExpander] = dict(expanders or {})

    def register(self, name: str, expander: ToolExpander) -> None:
        """Register (or replace) a pseudo tool expander.

        Args:
            name: The pseudo tool name, e.g. ``"mcp"``.
            expander: Callable returning the concrete tools.
        """
        self._expanders[name] = expander

    def __contains__(self, name: str) -> bool:
        return name in self._expanders

    def expand(self, name: str, tool_registry: Any) -> list:
        """Expand a pseudo tool name.

        Args:
            name: The registered pseudo name.
            tool_registry: The existing ToolRegistry.

        Returns:
            The expanded tools.
        """
        return list(self._expanders[name](tool_registry))


def default_pseudo_tools() -> PseudoToolRegistry:
    """Build the framework's default pseudo tool registry.

    Returns:
        A registry mapping ``mcp`` to the MCP-kind tool expander.
    """
    return PseudoToolRegistry({"mcp": expand_mcp_tools})


class ToolResolver:
    """Resolve an agent's logical tool names into Tool instances.

    SRP: tool selection only. Resolution goes through the *existing*
    ToolRegistry; the pseudo-tool expansion is delegated to an injected
    registry (OCP). The registry factory is injectable so the resolver can
    be tested without a live session (DIP).
    """

    def __init__(
        self,
        pseudo_tools: PseudoToolRegistry | None = None,
        registry_factory: Callable[[Config], Any] | None = None,
    ) -> None:
        """Initialize the resolver.

        Args:
            pseudo_tools: Registry of pseudo tool expanders.
            registry_factory: Builds a ToolRegistry when no session is
                supplied. Defaults to the framework's default registry.
        """
        self._pseudo_tools = pseudo_tools or default_pseudo_tools()
        self._registry_factory = registry_factory

    def _registry_for(self, config: Config, session: Any | None):
        """Return the ToolRegistry to resolve against."""
        if session is not None and session.tool_registry is not None:
            return session.tool_registry
        factory = self._registry_factory
        if factory is None:
            from tools.registry import create_default_registry

            factory = create_default_registry
        return factory(config)

    def resolve(
        self,
        agent_name: str,
        agent_yaml: AgentYaml | None,
        config: Config,
        session: Any | None = None,
    ) -> list | None:
        """Return the tools the agent may use, or None if unconfigured.

        Args:
            agent_name: Agent identifier used in error messages.
            agent_yaml: The agent's resolved configuration, or None.
            config: Runtime configuration.
            session: Optional session carrying the shared ToolRegistry.

        Returns:
            The resolved tools; None when the agent has no ``tools`` entry
            (leaving it unrestricted); an empty list means no tools.

        Raises:
            ConfigError: If a configured tool name is not registered.
        """
        if agent_yaml is None or agent_yaml.tools is None:
            return None

        registry = self._registry_for(config, session)

        resolved: list = []
        for tool_name in agent_yaml.tools:
            if tool_name in self._pseudo_tools:
                resolved.extend(self._pseudo_tools.expand(tool_name, registry))
                continue

            tool = registry.get(tool_name)
            if tool is None:
                raise ConfigError(
                    f"Agent '{agent_name}' references unknown tool '{tool_name}'"
                )
            resolved.append(tool)
        return resolved


# Agent assembly

class AgentOverrideApplier:
    """Apply resolved prompt/tools onto a constructed agent.

    SRP: encapsulates the one place that knows how an agent receives its
    prompt and tools, keeping that knowledge out of the factory.
    """

    def apply(
        self,
        agent: Any,
        prompt: str | None,
        tools: list | None,
        skill: str | None = None,
    ) -> None:
        """Apply the resolved prompt, tools, and default skill to an agent.

        Args:
            agent: The freshly constructed agent.
            prompt: Prompt content, or None to keep the agent's default.
            tools: Resolved tools, or None to leave tools unrestricted.
            skill: Skill name configured for this agent, or None when the
                agent has no default skill.
        """
        if prompt is not None:
            agent.system_prompt = prompt
        if tools is not None:
            agent.configured_tools = tools
            agent.allowed_tool_names = [tool.name for tool in tools]
        if skill is not None:
            agent.default_skill = skill


# Injection bundle


@dataclass(frozen=True)
class AgentComponents:
    """Collaborators injected into :class:`AgentFactory` (DIP).

    Bundling them keeps the factory's dependencies explicit and swappable
    in one place.
    """

    types: AgentTypeRegistry
    configs: AgentConfigProvider
    models: ModelResolver
    prompts: PromptResolver
    tools: ToolResolver
    settings: AgentSettingsApplier
    overrides: AgentOverrideApplier

    @classmethod
    def default(
        cls,
        agent_classes: Mapping[str, type],
        known_agents: set[str] | None = None,
    ) -> "AgentComponents":
        """Build the framework's default collaborator bundle.

        Args:
            agent_classes: Default mapping of agent name to class.
            known_agents: Names known to the config loader. Defaults to the
                keys of ``agent_classes``.

        Returns:
            A fully wired AgentComponents bundle.
        """
        return cls(
            types=AgentTypeRegistry(agent_classes),
            configs=AgentConfigProvider(
                known_agents=known_agents or set(agent_classes),
            ),
            models=ModelResolver(),
            prompts=PromptResolver(),
            tools=ToolResolver(),
            settings=AgentSettingsApplier(),
            overrides=AgentOverrideApplier(),
        )