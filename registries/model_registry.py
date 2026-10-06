from __future__ import annotations

from config.config import Config, ModelConfig
from utils.errors import ConfigError


class ModelRegistry:
    """Resolve logical model names to model configurations.

    Decouples agents from concrete model configuration. The registry
    returns a ModelConfig; callers apply it to a derived copy of the
    app Config so the LLM client keeps reading ``config.model_name`` at
    call time.

    Attributes:
        models: Mapping of logical model names to ModelConfig.
    """

    def __init__(self, models: dict[str, ModelConfig] | None = None) -> None:
        """Initialize the registry with an optional model mapping.

        Args:
            models: Mapping of logical model names to ModelConfig.
        """
        self.models: dict[str, ModelConfig] = dict(models or {})

    def get(self, name: str) -> ModelConfig:
        """Return the model configuration for a logical name.

        Args:
            name: Logical model identifier configured for an agent.

        Returns:
            The resolved ModelConfig.

        Raises:
            ConfigError: If the model is not registered.
        """
        definition = self.models.get(name)
        if definition is None:
            raise ConfigError(f"Model '{name}' is not registered")
        return definition

    def has(self, name: str) -> bool:
        """Return whether a logical model name is registered.

        Args:
            name: Logical model identifier.

        Returns:
            True if the name resolves to a model configuration.
        """
        return name in self.models

    def resolve_litellm(self, name: str) -> str:
        """Resolve a logical model name to a LiteLLM model string.

        Registered logical names are mapped through their ModelConfig
        (respecting an explicit ``litellm_model``). Unknown names are
        passed through unchanged, so callers may supply a raw LiteLLM
        model string (e.g. ``"openai/gpt-4.1"``) directly.

        Args:
            name: Logical model name or a raw LiteLLM model string.

        Returns:
            The LiteLLM model identifier.
        """
        definition = self.models.get(name)
        if definition is None:
            return name
        return definition.to_litellm_model()

    def apply(self, name: str, config: Config) -> Config:
        """Return a copy of config with a model configuration applied.

        The returned config has ``model.name``, ``model.temperature``, and
        ``model.context_window`` overridden so the LLM client picks them up
        at call time. The original config is never mutated.

        Args:
            name: Logical model identifier configured for an agent.
            config: Runtime configuration to copy.

        Returns:
            A deep copy of config with the resolved model applied.

        Raises:
            ConfigError: If the model is not registered.
        """
        definition = self.get(name)
        derived = config.model_copy(deep=True)
        derived.model.name = definition.name
        derived.model.temperature = definition.temperature
        derived.model.context_window = definition.context_window
        return derived