"""LiteLLM-backed LLM client.

This is the single place in the framework that talks to LiteLLM. Agents
never import LiteLLM or a provider SDK; they only consume the framework's
``StreamEvent`` stream produced here.

    LLMClient → LiteLLM → OpenAI / Gemini / Anthropic / Ollama

Model selection accepts either a logical model name (resolved through the
ModelRegistry) or a raw LiteLLM model string. Retry/backoff and model
fallback are handled here so callers do not duplicate that logic.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from client.response import (
    StreamEvent,
    StreamEventType,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    parse_tool_call_arguments,
)
from config.config import Config
from client.providers import ProviderCredentialResolver, default_resolver
from utils.logging_config import describe_messages, get_logger, prompts_enabled

logger = get_logger(__name__)

# Errors that are worth retrying against the same model.
_RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    Timeout,
    ServiceUnavailableError,
    InternalServerError,
)

# Errors that are structural and will not succeed on retry, but may succeed
# on a different (fallback) model.
_NON_RETRYABLE_ERRORS = (AuthenticationError, BadRequestError, NotFoundError)

# LiteLLM is chatty and rejects unsupported params by default; keep it quiet
# and tolerant so provider quirks do not surface as hard failures.
litellm.drop_params = True
litellm.suppress_debug_info = True


class LLMClient:
    """Async LLM client backed by LiteLLM.

    Attributes:
        config: Runtime configuration.
        model_registry: Optional ModelRegistry used to resolve logical
            model names to LiteLLM model strings.
        model_router: Optional ModelRouter used to pick a logical model
            when the caller does not specify one.
        credentials: Resolver that supplies provider credential kwargs.
    """

    def __init__(
        self,
        config: Config,
        model_registry: Any | None = None,
        model_router: Any | None = None,
        credentials: ProviderCredentialResolver | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            config: Runtime configuration.
            model_registry: Optional ModelRegistry for logical→LiteLLM
                model resolution.
            model_router: Optional ModelRouter for automatic model choice.
            credentials: Optional provider credential resolver. Defaults to
                the framework's built-in provider registry.
        """
        self.config = config
        self.model_registry = model_registry
        self.model_router = model_router
        self.credentials = credentials or default_resolver()

    async def close(self) -> None:
        """Release client resources.

        LiteLLM manages its own HTTP sessions, so there is nothing to
        close explicitly; this is kept for interface compatibility with
        the previous OpenAI-backed client.
        """
        return None

    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------

    def _resolve_model(self, model: str | None) -> str:
        """Resolve a logical or raw model name to a LiteLLM model string.

        Args:
            model: Logical model name, raw LiteLLM model string, or None
                to use the configured default.

        Returns:
            The LiteLLM model identifier.
        """
        if model:
            if self.model_registry is not None:
                return self.model_registry.resolve_litellm(model)
            return model

        default = self.config.model_name
        if self.model_registry is not None and self.model_registry.has(default):
            return self.model_registry.resolve_litellm(default)

        return self.config.default_litellm_model

    def _candidate_models(self, model: str | None) -> list[str]:
        """Return the ordered list of models to attempt.

        The requested (or default) model is tried first, followed by any
        configured fallback models, with duplicates removed.

        Args:
            model: Logical model name, raw LiteLLM model string, or None.

        Returns:
            De-duplicated LiteLLM model identifiers.
        """
        candidates = [self._resolve_model(model)]
        candidates.extend(
            self._resolve_model(name)
            for name in self.config.litellm_fallback_models
        )

        ordered: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in ordered:
                ordered.append(candidate)
        return ordered

    def _provider_kwargs(self, model: str) -> dict[str, Any]:
        """Return provider credentials for a LiteLLM model string.

        Delegates to the injected credential resolver so the client holds
        no provider-specific knowledge (SRP) and new providers are added
        by registering a strategy (OCP).

        Args:
            model: The LiteLLM model string.

        Returns:
            Keyword arguments merged into the LiteLLM call.
        """
        return self.credentials.resolve(model, self.config)

    # ------------------------------------------------------------------
    # Tool schema conversion
    # ------------------------------------------------------------------

    def _build_tools(self, tools: list[dict[str, Any]]):
        """Convert framework tool schemas into OpenAI function tools.

        Args:
            tools: Tool schemas as produced by ``Tool.to_openai_schema``.

        Returns:
            LiteLLM/OpenAI-format tool definitions.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get(
                        "parameters",
                        {
                            "type": "object",
                            "properties": {},
                        },
                    ),
                },
            }
            for tool in tools
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs_extra,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a chat completion as framework StreamEvents.

        Tries the requested model with exponential backoff, then each
        configured fallback model in order. Emits a single ERROR event
        when every candidate fails.

        Args:
            messages: OpenAI-format chat messages.
            model: Logical model name or raw LiteLLM model string. None
                uses the router (if any) or the configured default.
            tools: Optional tool schemas to expose to the model.
            stream: Whether to stream the response.
            tool_choice: Optional tool selection directive.
            **kwargs_extra: Extra arguments forwarded to LiteLLM.

        Yields:
            StreamEvent objects (TEXT_DELTA, TOOL_CALL_*, MESSAGE_COMPLETE
            or ERROR).
        """
        # Adaptive routing is opt-in. When disabled (the default) an
        # unspecified model means "the caller's configured model", so an
        # agent's config.yaml model is never silently overridden.
        if (
            model is None
            and self.model_router is not None
            and self.config.llm_routing_enabled
        ):
            model = self.model_router.route(messages=messages)

        candidates = self._candidate_models(model)

        # Metadata only by default: this framework handles patient data, so
        # message content is logged solely when LOG_PROMPTS is enabled.
        logger.debug(
            "LLM request",
            extra={
                "fields": {
                    "candidates": candidates,
                    "stream": stream,
                    "tools": len(tools) if tools else 0,
                    **describe_messages(messages),
                }
            },
        )

        if prompts_enabled():
            logger.debug("LLM request content", extra={"fields": {"messages": messages}})

        kwargs: dict[str, Any] = {"messages": messages, "stream": stream}
        kwargs.update(kwargs_extra)

        if tools:
            kwargs["tools"] = self._build_tools(tools)
            kwargs["tool_choice"] = tool_choice if tool_choice is not None else "auto"

        if stream:
            # Ask OpenAI-compatible providers to include token usage.
            kwargs.setdefault("stream_options", {"include_usage": True})

        last_error: Exception | None = None

        for index, candidate in enumerate(candidates):
            if index:
                logger.warning(
                    "Previous model failed; falling back to %s", candidate
                )
            attempt = 0
            while True:
                try:
                    if stream:
                        async for event in self._stream(candidate, kwargs):
                            yield event
                    else:
                        async for event in self._complete(candidate, kwargs):
                            yield event
                    return
                except _NON_RETRYABLE_ERRORS as exc:
                    # Structural failure for this model: try the next one.
                    last_error = exc
                    break
                except _RETRYABLE_ERRORS as exc:
                    last_error = exc
                    if attempt < self.config.litellm_num_retries:
                        await asyncio.sleep(2**attempt)
                        attempt += 1
                        continue
                    break
                except Exception as exc:  # noqa: BLE001 - provider-agnostic
                    last_error = exc
                    break

        logger.error("All models failed. Last error: %s", last_error)
        yield StreamEvent(
            type=StreamEventType.ERROR,
            error=f"All models failed. Last error: {last_error}",
        )

    # ------------------------------------------------------------------
    # LiteLLM invocation
    # ------------------------------------------------------------------

    def _base_kwargs(self, model: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge model, provider credentials, and timeout into the call."""
        call_kwargs = dict(kwargs)
        call_kwargs["model"] = model
        call_kwargs.setdefault("timeout", self.config.litellm_timeout)
        call_kwargs.setdefault("temperature", self._temperature_for(model))
        call_kwargs.update(self._provider_kwargs(model))
        return call_kwargs

    def _temperature_for(self, model: str) -> float:
        """Return the configured temperature for a LiteLLM model string.

        Falls back to the runtime default model's temperature when the
        model does not correspond to a registered logical entry.

        Args:
            model: The LiteLLM model string.

        Returns:
            Sampling temperature.
        """
        if self.model_registry is not None:
            for definition in self.model_registry.models.values():
                if definition.to_litellm_model() == model:
                    return definition.temperature
        return self.config.model.temperature

    async def _stream(
        self,
        model: str,
        kwargs: dict[str, Any],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a single model and convert chunks to StreamEvents."""
        call_kwargs = self._base_kwargs(model, kwargs)
        response = await litellm.acompletion(**call_kwargs)

        finish_reason: str | None = None
        usage: TokenUsage | None = None
        tool_calls: dict[int, dict[str, Any]] = {}

        async for chunk in response:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage:
                usage = self._to_usage(chunk_usage)

            choices = getattr(chunk, "choices", None)
            if not choices:
                continue

            choice = choices[0]
            delta = getattr(choice, "delta", None)

            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason

            content = getattr(delta, "content", None) if delta else None
            if content:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=TextDelta(content),
                )

            delta_tool_calls = getattr(delta, "tool_calls", None) if delta else None
            if delta_tool_calls:
                for tool_call_delta in delta_tool_calls:
                    idx = getattr(tool_call_delta, "index", 0)
                    state = tool_calls.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""}
                    )

                    call_id = getattr(tool_call_delta, "id", None)
                    if call_id:
                        state["id"] = call_id

                    function = getattr(tool_call_delta, "function", None)
                    if not function:
                        continue

                    name = getattr(function, "name", None)
                    if name:
                        state["name"] = name
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_START,
                            tool_call_delta=ToolCallDelta(
                                call_id=state["id"],
                                name=name,
                            ),
                        )

                    arguments = getattr(function, "arguments", None)
                    if arguments:
                        state["arguments"] += arguments
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_DELTA,
                            tool_call_delta=ToolCallDelta(
                                call_id=state["id"],
                                name=state["name"],
                                arguments_delta=arguments,
                            ),
                        )

        for state in tool_calls.values():
            yield StreamEvent(
                type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call=ToolCall(
                    call_id=state["id"],
                    name=state["name"],
                    arguments=parse_tool_call_arguments(state["arguments"]),
                ),
            )

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def _complete(
        self,
        model: str,
        kwargs: dict[str, Any],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run a single non-streaming completion as StreamEvents.

        Emits a TOOL_CALL_COMPLETE event per tool call, then a terminal
        MESSAGE_COMPLETE carrying the text and usage (the shape the
        ChatCompactor expects).
        """
        call_kwargs = self._base_kwargs(model, kwargs)
        response = await litellm.acompletion(**call_kwargs)

        choice = response.choices[0]
        message = choice.message

        text_delta = None
        if getattr(message, "content", None):
            text_delta = TextDelta(content=message.content)

        usage = None
        if getattr(response, "usage", None):
            usage = self._to_usage(response.usage)

        for tc in getattr(message, "tool_calls", None) or []:
            function = getattr(tc, "function", None)
            yield StreamEvent(
                type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call=ToolCall(
                    call_id=getattr(tc, "id", ""),
                    name=getattr(function, "name", None) if function else None,
                    arguments=parse_tool_call_arguments(
                        getattr(function, "arguments", "") if function else ""
                    ),
                ),
            )

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage,
        )

    @staticmethod
    def _to_usage(usage: Any) -> TokenUsage:
        """Convert a LiteLLM usage object into framework TokenUsage."""
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0

        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            cached_tokens=cached,
        )