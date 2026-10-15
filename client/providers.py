"""Provider credential strategies for LiteLLM.

*How* a provider authenticates is its own concern. Keeping that knowledge
out of the ``LLMClient`` gives us:

- **SRP** — the client only talks to LiteLLM; it does not know which
  provider needs which key or base URL.
- **OCP** — a new provider is supported by registering a strategy, not by
  editing the client.
- **DIP** — the client depends on the ``ProviderCredentials`` abstraction,
  not on concrete provider branches.

    LiteLLM model string → ProviderCredentialResolver → ProviderCredentials
                                                       → call kwargs
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProviderCredentials(Protocol):
    """Strategy that supplies extra LiteLLM kwargs for a provider family."""

    def call_kwargs(self, model: str, config: Any) -> dict[str, Any]:
        """Return extra kwargs for a LiteLLM call to ``model``.

        Args:
            model: The LiteLLM model string (e.g. ``"openai/gpt-4.1"``).
            config: The runtime configuration.

        Returns:
            Keyword arguments merged into the LiteLLM call.
        """
        ...


class OpenAIGatewayCredentials:
    """OpenAI-compatible gateways authenticated by the framework's API_KEY.

    These providers reuse ``API_KEY`` / ``BASE_URL`` (OpenAI, OpenRouter,
    Azure, Groq, Together, xAI, Fireworks, Mistral, DeepSeek, ...).
    """

    def call_kwargs(self, model: str, config: Any) -> dict[str, Any]:
        """Forward the framework API_KEY / BASE_URL.

        Args:
            model: The LiteLLM model string.
            config: The runtime configuration.

        Returns:
            ``api_key`` and/or ``api_base`` when configured.
        """
        kwargs: dict[str, Any] = {}
        api_key = getattr(config, "api_key", None)
        base_url = getattr(config, "base_url", None)
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["api_base"] = base_url
        return kwargs


class LocalServerCredentials:
    """Self-hosted servers with their own base URL environment variable.

    Used by Ollama. Several variable names are accepted because LiteLLM's
    native name (``OLLAMA_API_BASE``) differs from the more common
    ``OLLAMA_BASE_URL``. The framework's OpenAI ``BASE_URL`` / ``API_KEY``
    are deliberately *not* forwarded — doing so would point the local
    client at the wrong server and send a key it does not expect.
    """

    def __init__(self, *base_envs: str) -> None:
        """Initialize the strategy.

        Args:
            *base_envs: Environment variable names holding the base URL,
                in priority order.
        """
        self.base_envs = tuple(base_envs)

    def call_kwargs(self, model: str, config: Any) -> dict[str, Any]:
        """Return ``api_base`` from the provider's own env var, if set.

        Args:
            model: The LiteLLM model string.
            config: The runtime configuration (unused).

        Returns:
            ``{"api_base": ...}`` when an env var is set, else ``{}``.
        """
        for env_name in self.base_envs:
            value = os.environ.get(env_name)
            if value and value.strip():
                return {"api_base": value.strip()}
        return {}


class ProviderEnvCredentials:
    """Providers authenticated purely by LiteLLM's own environment.

    Anthropic (``ANTHROPIC_API_KEY``) and Gemini (``GEMINI_API_KEY``) read
    their keys from LiteLLM's native environment variables, so nothing is
    forwarded from the framework configuration. This is also the safe
    default for any unregistered provider.
    """

    def call_kwargs(self, model: str, config: Any) -> dict[str, Any]:
        """Return no extra kwargs.

        Args:
            model: The LiteLLM model string.
            config: The runtime configuration.

        Returns:
            An empty mapping.
        """
        return {}


class ProviderCredentialResolver:
    """Resolve a LiteLLM model string to its credential kwargs.

    Depends on the ``ProviderCredentials`` abstraction (DIP) and is open
    for extension through :meth:`register` (OCP).

    Attributes:
        default: Strategy used for providers with no registration.
    """

    def __init__(
        self,
        strategies: dict[str, ProviderCredentials] | None = None,
        default: ProviderCredentials | None = None,
    ) -> None:
        """Initialize the resolver.

        Args:
            strategies: Optional mapping of provider name to strategy.
            default: Fallback strategy for unregistered providers.
                Defaults to :class:`ProviderEnvCredentials`.
        """
        self._strategies: dict[str, ProviderCredentials] = dict(strategies or {})
        self.default: ProviderCredentials = default or ProviderEnvCredentials()

    def register(self, provider: str, strategy: ProviderCredentials) -> None:
        """Register (or replace) the strategy for a provider.

        Args:
            provider: Provider name, e.g. ``"gemini"``.
            strategy: The strategy that resolves its credentials.
        """
        self._strategies[provider.lower()] = strategy

    @staticmethod
    def provider_of(model: str) -> str:
        """Return the provider prefix of a LiteLLM model string.

        Args:
            model: The LiteLLM model string.

        Returns:
            The provider name (e.g. ``"gemini"``), or the whole string when
            it carries no provider prefix.
        """
        return model.split("/", 1)[0].lower() if "/" in model else model.lower()

    def strategy_for(self, model: str) -> ProviderCredentials:
        """Return the strategy for a model's provider.

        Args:
            model: The LiteLLM model string.

        Returns:
            The registered strategy, or :attr:`default`.
        """
        return self._strategies.get(self.provider_of(model), self.default)

    def resolve(self, model: str, config: Any) -> dict[str, Any]:
        """Return the credential kwargs for a model.

        Args:
            model: The LiteLLM model string.
            config: The runtime configuration.

        Returns:
            Keyword arguments merged into the LiteLLM call.
        """
        return self.strategy_for(model).call_kwargs(model, config)


# Providers that authenticate with the framework's API_KEY / BASE_URL.
OPENAI_GATEWAY_PROVIDERS: frozenset[str] = frozenset({
    "openai", "openrouter", "azure", "deepseek",
    "groq", "together_ai", "xai", "fireworks_ai", "mistral",
})

# Providers with their own base URL environment variables (priority order).
LOCAL_SERVER_BASE_ENV: dict[str, tuple[str, ...]] = {
    "ollama": ("OLLAMA_API_BASE", "OLLAMA_BASE_URL"),
}


def default_resolver() -> ProviderCredentialResolver:
    """Build the resolver with the framework's built-in provider registry.

    Returns:
        A resolver covering the OpenAI-compatible gateways and local
        servers; all other providers fall back to LiteLLM environment
        variables (Anthropic, Gemini, ...).
    """
    strategies: dict[str, ProviderCredentials] = {
        name: OpenAIGatewayCredentials() for name in OPENAI_GATEWAY_PROVIDERS
    }
    for name, env_vars in LOCAL_SERVER_BASE_ENV.items():
        strategies[name] = LocalServerCredentials(*env_vars)
    return ProviderCredentialResolver(strategies)