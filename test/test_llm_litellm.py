"""Tests for the LiteLLM integration.

Covers generation, streaming, model routing, provider failure/fallback,
and the Agent → ModelRouter → ModelRegistry → LLMClient → LiteLLM flow.

LiteLLM itself is mocked (``litellm.acompletion``) so the tests never
touch the network.
"""

from types import SimpleNamespace as NS

import pytest

import client.llm_client as llm_module
from client.llm_client import LLMClient
from client.model_router import ModelRouter, TaskType
from client.response import StreamEvent, StreamEventType, TokenUsage
from config.config import Config, ModelConfig, ModelsSection
from registries.model_registry import ModelRegistry

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeStream:
    """Minimal async iterator that mimics LiteLLM's streaming wrapper."""

    def __init__(self, chunks):
        self._chunks = chunks
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


def _chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    delta = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(delta=delta, finish_reason=finish_reason)], usage=usage)


def _usage(prompt=5, completion=2, total=7):
    return NS(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        prompt_tokens_details=None,
    )


def _tool_delta(index=0, call_id="call_1", name=None, arguments=None):
    function = NS(name=name, arguments=arguments)
    return NS(index=index, id=call_id, function=function)


def _models_config() -> ModelsSection:
    return ModelsSection(definitions={
        "fast": ModelConfig(provider="openai", name="gpt-4o-mini",
                            litellm_model="openai/gpt-4o-mini", temperature=0.3),
        "clinical": ModelConfig(provider="openai", name="gpt-4.1",
                                litellm_model="openai/gpt-4.1", temperature=0.1),
        "reasoning": ModelConfig(provider="anthropic",
                                 name="claude-3-5-sonnet-20241022",
                                 litellm_model="anthropic/claude-3-5-sonnet-20241022"),
        "gemini": ModelConfig(provider="gemini", name="gemini-2.0-flash",
                              litellm_model="gemini/gemini-2.0-flash"),
        "ollama": ModelConfig(provider="ollama", name="llama3.2",
                              litellm_model="ollama/llama3.2"),
    })


class _StubConfig:
    """Minimal stand-in for Config used by the credential strategy tests."""

    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url


# ---------------------------------------------------------------------------
# ModelRouter (independent of LiteLLM)
# ---------------------------------------------------------------------------


def test_router_default_routes():
    router = ModelRouter()
    assert router.routes == {"simple": "fast", "clinical": "reasoning", "retrieval": "fast"}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("hello there", TaskType.SIMPLE),
        ("Assess the patient and provide a differential diagnosis", TaskType.CLINICAL),
        ("look up the patient record for elizabeth", TaskType.RETRIEVAL),
    ],
)
def test_router_classifies_text(text, expected):
    assert ModelRouter().classify(text) == expected


def test_router_routes_messages_to_logical_model():
    router = ModelRouter()
    assert router.route(messages=[{"role": "user", "content": "hi"}]) == "fast"
    assert router.route(
        messages=[{"role": "user", "content": "assess the patient's risk"}]
    ) == "reasoning"
    assert router.route(
        messages=[{"role": "user", "content": "look up the patient record"}]
    ) == "fast"


def test_router_explicit_task_type_overrides_classification():
    router = ModelRouter()
    assert router.route(task_type="clinical") == "reasoning"
    assert router.route(task_type=TaskType.RETRIEVAL) == "fast"


def test_router_does_not_import_litellm():
    import inspect

    source = inspect.getsource(ModelRouter)
    assert "litellm" not in source.lower()


# ---------------------------------------------------------------------------
# Provider credentials (strategy / OCP / DIP)
# ---------------------------------------------------------------------------


def test_openai_gateway_forwards_api_key_and_base_url():
    from client.providers import OpenAIGatewayCredentials

    cfg = _StubConfig(api_key="k", base_url="https://openrouter.ai/api/v1")
    kwargs = OpenAIGatewayCredentials().call_kwargs("openrouter/gpt-4o", cfg)

    assert kwargs == {"api_key": "k", "api_base": "https://openrouter.ai/api/v1"}


def test_gemini_forwards_nothing_and_uses_its_env(monkeypatch):
    """Gemini authenticates via GEMINI_API_KEY, not the framework API_KEY."""
    from client.providers import default_resolver

    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    cfg = _StubConfig(api_key="openai-key", base_url="https://api.openai.com/v1")

    assert default_resolver().resolve("gemini/gemini-2.0-flash", cfg) == {}


def test_ollama_never_receives_the_openai_key_or_base_url(monkeypatch):
    from client.providers import default_resolver

    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    cfg = _StubConfig(api_key="openai-key", base_url="https://api.openai.com/v1")

    # No api_key, and crucially no OpenAI api_base.
    assert default_resolver().resolve("ollama/llama3.2", cfg) == {}


def test_ollama_uses_its_own_base_url_env(monkeypatch):
    from client.providers import default_resolver

    monkeypatch.setenv("OLLAMA_API_BASE", "http://localhost:11434")
    cfg = _StubConfig(api_key="openai-key", base_url="https://api.openai.com/v1")

    assert default_resolver().resolve("ollama/llama3.2", cfg) == {
        "api_base": "http://localhost:11434"
    }


def test_ollama_accepts_the_alternate_base_url_env(monkeypatch):
    """OLLAMA_BASE_URL works too (whitespace trimmed)."""
    from client.providers import default_resolver

    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "  http://localhost:11434  ")
    cfg = _StubConfig(api_key="openai-key", base_url="https://api.openai.com/v1")

    assert default_resolver().resolve("ollama/medgemma-multi:latest", cfg) == {
        "api_base": "http://localhost:11434"
    }


def test_ollama_api_base_takes_priority(monkeypatch):
    from client.providers import default_resolver

    monkeypatch.setenv("OLLAMA_API_BASE", "http://api-base:11434")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://base-url:11434")
    cfg = _StubConfig()

    assert default_resolver().resolve("ollama/llama3.2", cfg) == {
        "api_base": "http://api-base:11434"
    }


def test_unknown_provider_falls_back_to_env_credentials():
    from client.providers import default_resolver

    cfg = _StubConfig(api_key="openai-key", base_url="https://api.openai.com/v1")
    assert default_resolver().resolve("anthropic/claude-3-5-sonnet", cfg) == {}


def test_resolver_is_open_for_extension():
    """A new provider is supported by registering a strategy (OCP)."""
    from client.providers import ProviderCredentialResolver

    class _Custom:
        def call_kwargs(self, model, config):
            return {"api_key": "custom-key"}

    resolver = ProviderCredentialResolver()
    resolver.register("myprovider", _Custom())

    cfg = _StubConfig(api_key="openai-key", base_url="https://api.openai.com/v1")
    assert resolver.resolve("myprovider/model-x", cfg) == {"api_key": "custom-key"}
    # Other providers keep the default behaviour.
    assert resolver.resolve("gemini/gemini-2.0-flash", cfg) == {}


def test_resolver_provider_of_handles_bare_model_strings():
    from client.providers import ProviderCredentialResolver

    assert ProviderCredentialResolver.provider_of("openai/gpt-4.1") == "openai"
    assert ProviderCredentialResolver.provider_of("gpt-4.1") == "gpt-4.1"


def test_llm_client_uses_the_injected_credentials_resolver():
    """The client depends on the abstraction, not a concrete provider map."""
    from client.providers import ProviderCredentialResolver

    class _Recording:
        def __init__(self):
            self.seen = []

        def call_kwargs(self, model, config):
            self.seen.append(model)
            return {}

    recording = _Recording()
    client = LLMClient(Config(), credentials=ProviderCredentialResolver(
        {"openai": recording}
    ))

    client._provider_kwargs("openai/gpt-4.1")
    assert recording.seen == ["openai/gpt-4.1"]


# ---------------------------------------------------------------------------
# ModelRegistry resolution
# ---------------------------------------------------------------------------


def test_registry_resolves_logical_model_to_litellm_string():
    registry = ModelRegistry(_models_config().definitions)
    assert registry.resolve_litellm("fast") == "openai/gpt-4o-mini"
    assert registry.resolve_litellm("reasoning") == "anthropic/claude-3-5-sonnet-20241022"


def test_registry_passes_through_raw_litellm_string():
    registry = ModelRegistry({})
    assert registry.resolve_litellm("ollama/llama3") == "ollama/llama3"


def test_registry_builds_litellm_string_from_provider_and_name():
    registry = ModelRegistry({
        "x": ModelConfig(provider="gemini", name="gemini-1.5-pro"),
    })
    assert registry.resolve_litellm("x") == "gemini/gemini-1.5-pro"


# ---------------------------------------------------------------------------
# Generation + streaming
# ---------------------------------------------------------------------------


def test_generation_non_streaming(monkeypatch):
    async def fake_acompletion(**kwargs):
        assert kwargs["model"] == "openai/gpt-4o-mini"
        return NS(
            choices=[NS(message=NS(content="Hello there", tool_calls=None),
                        finish_reason="stop")],
            usage=_usage(3, 1, 4),
        )

    monkeypatch.setattr(llm_module.litellm, "acompletion", fake_acompletion)

    client = LLMClient(Config(), model_registry=ModelRegistry(_models_config().definitions))
    events = _collect(client, [{"role": "user", "content": "hi"}], stream=False, model="fast")

    assert events[-1].type == StreamEventType.MESSAGE_COMPLETE
    assert events[-1].text_delta.content == "Hello there"
    assert events[-1].usage.total_tokens == 4


def test_streaming_converts_chunks_to_stream_events(monkeypatch):
    async def fake_acompletion(**kwargs):
        assert kwargs["stream"] is True
        return _FakeStream([
            _chunk(content="Hello "),
            _chunk(content="world"),
            _chunk(finish_reason="stop", usage=_usage()),
        ])

    monkeypatch.setattr(llm_module.litellm, "acompletion", fake_acompletion)

    client = LLMClient(Config(), model_registry=ModelRegistry(_models_config().definitions))
    events = _collect(client, [{"role": "user", "content": "hi"}], stream=True, model="fast")

    text = "".join(
        e.text_delta.content for e in events
        if e.type == StreamEventType.TEXT_DELTA
    )
    assert text == "Hello world"
    assert events[-1].type == StreamEventType.MESSAGE_COMPLETE
    assert events[-1].usage.total_tokens == 7


def test_streaming_converts_tool_calls(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _FakeStream([
            _chunk(tool_calls=[_tool_delta(name="wiki_search")]),
            _chunk(tool_calls=[_tool_delta(arguments='{"query": "eliz')]),
            _chunk(tool_calls=[_tool_delta(arguments='abeth"}')]),
            _chunk(finish_reason="tool_calls"),
        ])

    monkeypatch.setattr(llm_module.litellm, "acompletion", fake_acompletion)

    client = LLMClient(Config(), model_registry=ModelRegistry(_models_config().definitions))
    events = _collect(client, [{"role": "user", "content": "hi"}], stream=True, model="fast")

    completes = [e for e in events if e.type == StreamEventType.TOOL_CALL_COMPLETE]
    assert len(completes) == 1
    assert completes[0].tool_call.name == "wiki_search"
    assert completes[0].tool_call.arguments == {"query": "elizabeth"}


# ---------------------------------------------------------------------------
# Provider failure / fallback
# ---------------------------------------------------------------------------


def test_fallback_to_next_model_on_failure(monkeypatch):
    from litellm.exceptions import RateLimitError

    monkeypatch.setenv("LLM_FALLBACK_MODELS", "fast")
    monkeypatch.setenv("LLM_NUM_RETRIES", "0")

    attempted = []

    async def fake_acompletion(**kwargs):
        attempted.append(kwargs["model"])
        if kwargs["model"] == "openai/gpt-4.1":
            raise RateLimitError("rate limited", llm_provider="openai", model="x")
        return _FakeStream([_chunk(content="fallback ok", finish_reason="stop")])

    monkeypatch.setattr(llm_module.litellm, "acompletion", fake_acompletion)

    client = LLMClient(Config(), model_registry=ModelRegistry(_models_config().definitions))
    events = _collect(client, [{"role": "user", "content": "hi"}], model="clinical")

    assert attempted == ["openai/gpt-4.1", "openai/gpt-4o-mini"]
    assert any(
        e.type == StreamEventType.TEXT_DELTA and e.text_delta.content == "fallback ok"
        for e in events
    )


def test_fallback_chain_reaches_gemini_then_ollama(monkeypatch):
    """primary (anthropic) fails → gemini fails → ollama succeeds."""
    from litellm.exceptions import RateLimitError

    monkeypatch.setenv("LLM_FALLBACK_MODELS", "gemini,ollama")
    monkeypatch.setenv("LLM_NUM_RETRIES", "0")
    monkeypatch.setenv("OLLAMA_API_BASE", "http://localhost:11434")

    attempted = []
    sent_kwargs = {}

    async def fake_acompletion(**kwargs):
        attempted.append(kwargs["model"])
        sent_kwargs[kwargs["model"]] = kwargs
        if kwargs["model"] != "ollama/llama3.2":
            raise RateLimitError("rate limited", llm_provider="x", model="x")
        return _FakeStream([_chunk(content="ollama ok", finish_reason="stop")])

    monkeypatch.setattr(llm_module.litellm, "acompletion", fake_acompletion)

    client = LLMClient(Config(), model_registry=ModelRegistry(_models_config().definitions))
    events = _collect(client, [{"role": "user", "content": "hi"}], model="reasoning")

    assert attempted == [
        "anthropic/claude-3-5-sonnet-20241022",
        "gemini/gemini-2.0-flash",
        "ollama/llama3.2",
    ]
    assert any(
        e.type == StreamEventType.TEXT_DELTA and e.text_delta.content == "ollama ok"
        for e in events
    )
    # Ollama is reached through its own base URL, never the OpenAI key/base.
    assert sent_kwargs["ollama/llama3.2"].get("api_base") == "http://localhost:11434"
    assert "api_key" not in sent_kwargs["ollama/llama3.2"]


def test_all_models_failing_emits_error_event(monkeypatch):
    from litellm.exceptions import AuthenticationError

    monkeypatch.setenv("LLM_FALLBACK_MODELS", "")
    monkeypatch.setenv("LLM_NUM_RETRIES", "0")

    async def fake_acompletion(**kwargs):
        raise AuthenticationError("bad key", llm_provider="openai", model="x")

    monkeypatch.setattr(llm_module.litellm, "acompletion", fake_acompletion)

    client = LLMClient(Config(), model_registry=ModelRegistry(_models_config().definitions))
    events = _collect(client, [{"role": "user", "content": "hi"}], model="clinical")

    assert events[-1].type == StreamEventType.ERROR
    assert "All models failed" in events[-1].error


def test_retryable_error_retries_same_model(monkeypatch):
    from litellm.exceptions import RateLimitError

    monkeypatch.setenv("LLM_NUM_RETRIES", "2")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "")

    calls = {"n": 0}

    async def fake_acompletion(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RateLimitError("rate limited", llm_provider="openai", model="x")
        return _FakeStream([_chunk(content="ok", finish_reason="stop")])

    monkeypatch.setattr(llm_module.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_module.asyncio, "sleep", _no_sleep)

    client = LLMClient(Config(), model_registry=ModelRegistry(_models_config().definitions))
    _collect(client, [{"role": "user", "content": "hi"}], model="clinical")

    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Agent → Router → Registry → Client → LiteLLM
# ---------------------------------------------------------------------------


def test_client_uses_router_when_no_model_given(monkeypatch):
    seen = {}

    async def fake_acompletion(**kwargs):
        seen["model"] = kwargs["model"]
        return _FakeStream([_chunk(content="ok", finish_reason="stop")])

    monkeypatch.setattr(llm_module.litellm, "acompletion", fake_acompletion)
    monkeypatch.setenv("LLM_ROUTING_ENABLED", "true")

    client = LLMClient(
        Config(),
        model_registry=ModelRegistry(_models_config().definitions),
        model_router=ModelRouter(),
    )
    _collect(client, [{"role": "user", "content": "hello, simple question"}])

    assert seen["model"] == "openai/gpt-4o-mini"


def test_client_does_not_route_when_routing_disabled(monkeypatch):
    """An agent's configured model must not be silently overridden."""
    seen = {}

    async def fake_acompletion(**kwargs):
        seen["model"] = kwargs["model"]
        return _FakeStream([_chunk(content="ok", finish_reason="stop")])

    monkeypatch.setattr(llm_module.litellm, "acompletion", fake_acompletion)
    monkeypatch.setenv("LLM_ROUTING_ENABLED", "false")

    client = LLMClient(
        Config(),
        model_registry=ModelRegistry(_models_config().definitions),
        model_router=ModelRouter(),
    )
    _collect(client, [{"role": "user", "content": "assess the patient"}])

    # Routed would be reasoning/anthropic; disabled → the configured default.
    assert seen["model"] == "openai/gpt-4.1"


def test_agent_routes_then_client_calls_litellm(monkeypatch):
    """Agent → ModelRouter → ModelRegistry → LLMClient → LiteLLM."""
    from agent.agent import Agent
    from agent.events import AgentType
    from agent.session import Session
    from context.manager import ContextManager

    monkeypatch.setenv("LLM_ROUTING_ENABLED", "true")

    config = Config(models=_models_config())
    session = Session(config)

    # Minimal context so the agentic loop has messages to route on.
    session.context_manager = ContextManager(config, user_memory=None, tools=[])
    session.context_manager.set_system_prompt("You are a test agent.")
    session.context_manager.add_user_message("hi, a simple question")

    recorded = {}

    async def fake_chat(messages, model=None, tools=None, tool_choice=None, **kwargs):
        recorded["model"] = model
        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason="stop",
            usage=TokenUsage(),
        )

    session.client.chat_completion = fake_chat

    class _TestAgent(Agent):
        def __init__(self, cfg, sess):
            super().__init__(cfg, "You are a test agent.", AgentType.CONSULT)
            self.session = sess

    agent = _TestAgent(config, session)
    _collect_agent(agent)

    # Router classified the request as simple → logical "fast".
    assert recorded["model"] == "fast"
    # And the registry resolves the logical name to the LiteLLM string.
    assert session.model_registry.resolve_litellm(recorded["model"]) == "openai/gpt-4o-mini"


def test_agent_routing_disabled_uses_default_model(monkeypatch):
    from agent.agent import Agent
    from agent.events import AgentType
    from agent.session import Session
    from context.manager import ContextManager

    monkeypatch.setenv("LLM_ROUTING_ENABLED", "false")

    config = Config(models=_models_config())
    session = Session(config)
    session.context_manager = ContextManager(config, user_memory=None, tools=[])
    session.context_manager.set_system_prompt("sys")
    session.context_manager.add_user_message("hi")

    recorded = {}

    async def fake_chat(messages, model=None, **kwargs):
        recorded["model"] = model
        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason="stop",
            usage=TokenUsage(),
        )

    session.client.chat_completion = fake_chat

    class _TestAgent(Agent):
        def __init__(self, cfg, sess):
            super().__init__(cfg, "sys", AgentType.CONSULT)
            self.session = sess

    agent = _TestAgent(config, session)
    _collect_agent(agent)

    assert recorded["model"] is None  # → client falls back to the default model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _no_sleep(*_args, **_kwargs):
    return None


def _collect(client, messages, stream=True, model=None):
    """Drive an async generator to completion synchronously."""
    import asyncio

    async def _run():
        return [
            event
            async for event in client.chat_completion(
                messages, model=model, stream=stream
            )
        ]

    return asyncio.run(_run())


def _collect_agent(agent):
    """Drive the agent's agentic loop to completion."""
    import asyncio

    async def _run():
        return [event async for event in agent._agentic_loop()]

    return asyncio.run(_run())