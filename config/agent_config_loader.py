"""Two-level agent configuration loader.

Loads the application-level ``agents.yaml`` (which agents are enabled)
and each enabled agent's ``config.yaml`` (model/prompt/tools
references), then validates every reference against the ModelRegistry,
PromptRegistry, and known tool names.

Agent config files are discovered by scanning ``customagents/`` for
``config.yaml`` and ``*.config.yaml`` files, keyed by the ``name`` field
inside each file. This supports one folder per agent and folders that
contain multiple agents (e.g. reportagent/ with soap, assessment, and
clinical_extraction).

Configuration is loaded once at application startup. It is never
re-read per API request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from config.config import Config, ModelConfig
from config.schemas import ApplicationAgentConfig, AgentYaml
from registries.model_registry import ModelRegistry
from registries.prompt_registry import PromptRegistry
from skills import SkillRegistry
from skills.errors import SkillError
from utils.errors import ConfigError

AGENTS_YAML_FILE = "agents.yaml"
AGENT_CONFIG_PATTERNS = ("config.yaml", "*.config.yaml")


@dataclass(frozen=True)
class ResolvedAgentConfig:
    """Immutable result of loading the two-level agent configuration.

    Attributes:
        enabled_agents: Mapping of agent name to its resolved AgentYaml.
        model_registry: ModelRegistry populated with configured models.
        prompt_registry: PromptRegistry populated with configured prompts.
    """

    enabled_agents: dict[str, AgentYaml] = field(default_factory=dict)
    model_registry: ModelRegistry = field(default_factory=ModelRegistry)
    prompt_registry: PromptRegistry = field(default_factory=PromptRegistry)


class AgentConfigLoader:
    """Load and validate the two-level agent configuration.

    Responsibilities:

    - load the application-level agents.yaml (enabled flags)
    - load the models section into the ModelRegistry
    - discover and load each enabled agent's config.yaml
    - validate every model/prompt/tool reference
    - fail early with clear ConfigError messages

    Args:
        config: The runtime application Config (already loaded once).
        base_dir: Project root used to locate config/ and customagents/.
            Defaults to the package root.
        known_agents: Set of agent names registered in the AgentFactory,
            used to validate that enabled agents exist.
    """

    def __init__(
        self,
        config: Config,
        *,
        base_dir: Path | None = None,
        known_agents: set[str] | None = None,
    ) -> None:
        self.config = config
        self.base_dir = (base_dir or Path(__file__).resolve().parent.parent)
        # The application-level agents.yaml lives at the project root.
        self._config_dir = self.base_dir
        self._customagents_dir = self.base_dir / "customagents"
        self._known_agents = known_agents or set()
        self.resolved: ResolvedAgentConfig | None = None
        self._discovered: dict[str, AgentYaml] | None = None

    def load(self) -> ResolvedAgentConfig:
        """Load and validate the full two-level configuration.

        Returns:
            The resolved, immutable agent configuration.

        Raises:
            ConfigError: If any configuration is invalid.
        """
        app_config = self._load_application_yaml()
        models = self._load_models(app_config.get("models", {}))
        prompt_registry = PromptRegistry.from_builtin()
        skill_registry = SkillRegistry.from_builtin()

        agent_files = self._discover_agent_config_files()

        enabled_agents: dict[str, AgentYaml] = {}
        for name, app_agent in app_config.get("agents", {}).items():
            if not app_agent.enabled:
                continue
            agent_yaml = agent_files.get(name)
            if agent_yaml is None:
                raise ConfigError(
                    f"Agent '{name}' is enabled but has no config.yaml "
                    f"under {self._customagents_dir}"
                )
            self._validate_references(
                name, agent_yaml, models, prompt_registry, skill_registry
            )
            enabled_agents[name] = agent_yaml

        self.resolved = ResolvedAgentConfig(
            enabled_agents=enabled_agents,
            model_registry=ModelRegistry(models),
            prompt_registry=prompt_registry,
        )
        return self.resolved

    def load_agent(self, name: str) -> AgentYaml | None:
        """Load a single agent's config.yaml, if present.

        Args:
            name: Logical agent name.

        Returns:
            The parsed AgentYaml, or None if no config file exists.
        """
        return self._discover_agent_config_files().get(name)

    def validate(self) -> list[str]:
        """Validate the configuration and return any error messages.

        Returns:
            A list of error strings. Empty when configuration is valid.
        """
        try:
            self.load()
            return []
        except ConfigError as exc:
            return [str(exc)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_agent_config_files(self) -> dict[str, AgentYaml]:
        """Discover agent config files under customagents/, keyed by name.

        Scans every ``config.yaml`` and ``*.config.yaml`` file below
        ``customagents/``. Files are keyed by their ``name`` field, so
        folders containing multiple agents are supported.

        The discovery result is cached on the loader so repeated calls do
        not re-scan the filesystem.

        Returns:
            Mapping of agent name to parsed AgentYaml.

        Raises:
            ConfigError: If an agent name is duplicated or a file is invalid.
        """
        if self._discovered is not None:
            return self._discovered

        discovered: dict[str, AgentYaml] = {}
        for pattern in AGENT_CONFIG_PATTERNS:
            for path in sorted(self._customagents_dir.rglob(pattern)):
                data = self._parse_yaml(path)
                try:
                    agent_yaml = AgentYaml(**data)
                except Exception as exc:
                    raise ConfigError(
                        f"Invalid agent config in {path}: {exc}",
                        config_file=str(path),
                    ) from exc

                if agent_yaml.name in discovered:
                    raise ConfigError(
                        f"Duplicate agent definition '{agent_yaml.name}' "
                        f"(conflicts between config files)"
                    )
                discovered[agent_yaml.name] = agent_yaml

        self._discovered = discovered
        return discovered

    def _load_application_yaml(self) -> dict[str, Any]:
        """Load and validate the application-level agents.yaml."""
        path = self._config_dir / AGENTS_YAML_FILE
        data = self._parse_yaml(path)

        models_raw = data.get("models", {})
        agents_raw = data.get("agents", {})

        if not isinstance(models_raw, dict) or not isinstance(agents_raw, dict):
            raise ConfigError(
                f"Invalid agents.yaml schema in {path}: expected 'models' and "
                "'agents' mappings",
                config_file=str(path),
            )

        agents: dict[str, ApplicationAgentConfig] = {}
        for name, value in agents_raw.items():
            if isinstance(value, bool):
                agents[name] = ApplicationAgentConfig(enabled=value)
            elif isinstance(value, dict):
                agents[name] = ApplicationAgentConfig(**value)
            else:
                raise ConfigError(
                    f"Invalid agent entry '{name}' in {path}",
                    config_file=str(path),
                )

        return {"models": models_raw, "agents": agents}

    def _load_models(self, models_raw: Any) -> dict[str, ModelConfig]:
        """Load the models section, validating each entry via pydantic.

        Args:
            models_raw: Raw models mapping from agents.yaml.

        Returns:
            Mapping of logical model name to ModelConfig.

        Raises:
            ConfigError: If a model definition is invalid.
        """
        if not isinstance(models_raw, dict):
            raise ConfigError("Expected a 'models' mapping in agents.yaml")
        models: dict[str, ModelConfig] = {}
        for name, value in models_raw.items():
            try:
                models[name] = ModelConfig(**value) if isinstance(value, dict) else value
            except Exception as exc:
                raise ConfigError(
                    f"Invalid model definition '{name}': {exc}"
                ) from exc
        return models

    def _validate_references(
        self,
        agent_name: str,
        agent_yaml: AgentYaml,
        models: dict[str, ModelConfig],
        prompt_registry: PromptRegistry,
        skill_registry: SkillRegistry,
    ) -> None:
        """Validate all registry references for an agent.

        Args:
            agent_name: Logical agent name used in error messages.
            agent_yaml: The agent's resolved configuration.
            models: Loaded model definitions.
            prompt_registry: Prompt registry to validate against.
            skill_registry: Skill registry to validate against.

        Raises:
            ConfigError: If any reference is unknown.
        """
        if agent_name not in self._known_agents:
            # A config-driven agent shares a registered class rather than being
            # registered under its own name; it names that class explicitly
            # (see AgentYaml.agent_class).
            if not agent_yaml.agent_class:
                raise ConfigError(
                    f"Agent '{agent_name}' is enabled but not registered in "
                    "AgentFactory, and its config.yaml sets no 'agent_class'"
                )
            if agent_yaml.agent_class not in self._known_agents:
                raise ConfigError(
                    f"Agent '{agent_name}' references unknown agent_class "
                    f"'{agent_yaml.agent_class}'"
                )

        if agent_yaml.model is not None and agent_yaml.model not in models:
            raise ConfigError(
                f"Agent '{agent_name}' references unknown model '{agent_yaml.model}'"
            )

        if agent_yaml.prompt is not None and not prompt_registry.has(agent_yaml.prompt):
            raise ConfigError(
                f"Agent '{agent_name}' references unknown prompt '{agent_yaml.prompt}'"
            )

        if agent_yaml.skill is not None:
            try:
                available = skill_registry.has(agent_yaml.skill)
            except SkillError as exc:
                raise ConfigError(
                    f"Agent '{agent_name}' skill library is invalid: {exc}"
                ) from exc

            if not available:
                raise ConfigError(
                    f"Agent '{agent_name}' references unknown skill "
                    f"'{agent_yaml.skill}'"
                )

        # Tool references are resolved through the existing ToolRegistry at
        # factory time; startup validation catches empty references and leaves
        # unknown names to the registry's own error behavior.
        for tool_name in agent_yaml.tools or []:
            if not tool_name:
                raise ConfigError(
                    f"Agent '{agent_name}' contains an empty tool reference"
                )

    @staticmethod
    def _parse_yaml(path: Path) -> dict[str, Any]:
        """Parse a YAML file into a mapping.

        Args:
            path: Path to the YAML file.

        Returns:
            The parsed YAML content as a mapping.

        Raises:
            ConfigError: If the file cannot be read or parsed.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
            return content if isinstance(content, dict) else {}
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Invalid YAML in {path}: {exc}", config_file=str(path)
            ) from exc
        except (OSError, IOError) as exc:
            raise ConfigError(
                f"Failed to read config file {path}: {exc}", config_file=str(path)
            ) from exc