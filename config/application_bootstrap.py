"""Application bootstrap for the configuration-driven agent runtime.

This module owns the FastAPI startup sequence. It loads the application
configuration exactly once, initializes the registries, loads the
enabled agents' per-agent configuration, validates every reference, and
produces an immutable ``RuntimeState`` stored on ``app.state``.

The bootstrap layer never constructs individual agents — that is the
AgentFactory's responsibility. It only composes the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config.agent_config_loader import AgentConfigLoader, ResolvedAgentConfig
from config.config import Config
from config.loader import load_config
from customagents.sessionmanager import SessionManager
from registries.model_registry import ModelRegistry
from registries.prompt_registry import PromptRegistry
from utils.errors import ConfigError


@dataclass(frozen=True)
class RuntimeState:
    """Immutable application runtime produced at startup.

    Attributes:
        config: The global application Config.
        agent_config: The resolved two-level agent configuration.
        model_registry: ModelRegistry initialized from agents.yaml.
        prompt_registry: PromptRegistry initialized from built-in prompts.
        session_manager: Application-scoped SessionManager.
        enabled_agents: Logical names of enabled agents.
    """

    config: Config
    agent_config: ResolvedAgentConfig
    model_registry: ModelRegistry
    prompt_registry: PromptRegistry
    session_manager: SessionManager
    enabled_agents: tuple[str, ...] = field(default_factory=tuple)


class ApplicationBootstrap:
    """Compose the application runtime from configuration.

    Responsibilities (in order):

    1. load the global Config (TOML + environment)
    2. load agents.yaml (models + enabled agents)
    3. initialize the ModelRegistry from agents.yaml model definitions
    4. initialize the PromptRegistry from built-in prompts
    5. load each enabled agent's config.yaml
    6. validate every model/prompt/tool reference
    7. create the application-scoped SessionManager
    8. produce the immutable RuntimeState

    Args:
        base_dir: Project root used to locate config/ and customagents/.
            Defaults to the package root.
        known_agents: Set of agent names registered in the AgentFactory.
            When None, derived from AgentFactory.known_agent_names().
    """

    def __init__(
        self,
        base_dir: Path | None = None,
        known_agents: set[str] | None = None,
    ) -> None:
        self.base_dir = (base_dir or Path(__file__).resolve().parent.parent)
        if known_agents is None:
            from customagents.factory import AgentFactory
            known_agents = AgentFactory.known_agent_names()
        self._known_agents = known_agents
        self.state: RuntimeState | None = None

    def run(self) -> RuntimeState:
        """Execute the full startup sequence.

        Returns:
            The immutable RuntimeState.

        Raises:
            ConfigError: If any configuration is invalid.
        """
        # 1. Load the global Config (TOML + environment) once.
        config = load_config(cwd=self.base_dir)

        # 2-6. Load agents.yaml, initialize registries, load enabled agent
        # configs, and validate every reference.
        loader = AgentConfigLoader(
            config,
            base_dir=self.base_dir,
            known_agents=self._known_agents,
        )
        resolved = loader.load()

        # 7. Create the application-scoped SessionManager.
        session_manager = SessionManager()

        # 8. Produce the immutable runtime state.
        self.state = RuntimeState(
            config=config,
            agent_config=resolved,
            model_registry=resolved.model_registry,
            prompt_registry=resolved.prompt_registry,
            session_manager=session_manager,
            enabled_agents=tuple(sorted(resolved.enabled_agents.keys())),
        )
        return self.state


def bootstrap_application(
    base_dir: Path | None = None,
    known_agents: set[str] | None = None,
) -> RuntimeState:
    """Convenience wrapper that runs the bootstrap and returns RuntimeState.

    Args:
        base_dir: Project root used to locate config/ and customagents/.
        known_agents: Set of agent names registered in the AgentFactory.

    Returns:
        The immutable RuntimeState.

    Raises:
        ConfigError: If any configuration is invalid.
    """
    return ApplicationBootstrap(
        base_dir=base_dir,
        known_agents=known_agents,
    ).run()
