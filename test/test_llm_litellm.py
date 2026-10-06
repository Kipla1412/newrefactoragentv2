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
        "coding": ModelConfig(provider="openai", name="gpt-4.1",
                              litellm_model="openai/gpt-4.1", temperature=0.1),
        "reasoning": ModelConfig(provider="anthropic",
                                 name="claude-3-5-sonnet-20241022",
                                 litellm_model="anthropic/claude-3-5-sonnet-20241022"),
    })


# ---------------------------------------------------------------------------
# ModelRouter (independent of LiteLLM)
# ---------------------------------------------------------------------------


def test_router_default_routes():
    router = ModelRouter()
    assert router.routes == {"simple": "fast", "coding": "coding", "complex": "reasoning"}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("hello there", TaskType.SIMPLE),
        ("Please fix this python bug in my function", TaskType.CODING),
        ("Analyze the trade-offs and design an architecture", TaskType.COMPLEX),
    ],
)
def test_router_classifies_text(text, expected):
    assert ModelRouter().classify(text) == expected


def test_router_routes_messages_to_logical_model():
    router = ModelRouter()
    assert router.route(messages=[{"role": "user", "content": "hi"}]) == "fast"
    assert router.route(
        messages=[{"role": "user", "content": "debug this sql query"}]
    ) == "coding"
    assert router.route(
        messages=[{"role": "user", "content": "analyze the architecture"}]
    ) == "reasoning"


def test_router_explicit_task_type_overrides_classification():
    router = ModelRouter()
    assert router.route(task_type="complex") == "reasoning"
    assert router.route(task_type=TaskType.CODING) == "coding"


def test_router_does_not_import_litellm():
    import inspect

    source = inspect.getsource(ModelRouter)
    assert "litellm" not in source.lower()


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
    events = _collect(client, [{"role": "user", "content": "hi"}], model="coding")

    assert attempted == ["openai/gpt-4.1", "openai/gpt-4o-mini"]
    assert any(
        e.type == StreamEventType.TEXT_DELTA and e.text_delta.content == "fallback ok"
        for e in events
    )


def test_all_models_failing_emits_error_event(monkeypatch):
    from litellm.exceptions import AuthenticationError

    monkeypatch.setenv("LLM_FALLBACK_MODELS", "")
    monkeypatch.setenv("LLM_NUM_RETRIES", "0")

    async def fake_acompletion(**kwargs):
        raise AuthenticationError("bad key", llm_provider="openai", model="x")

    monkeypatch.setattr(llm_module.litellm, "acompletion", fake_acompletion)

    client = LLMClient(Config(), model_registry=ModelRegistry(_models_config().definitions))
    events = _collect(client, [{"role": "user", "content": "hi"}], model="coding")

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
    _collect(client, [{"role": "user", "content": "hi"}], model="coding")

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

    client = LLMClient(
        Config(),
        model_registry=ModelRegistry(_models_config().definitions),
        model_router=ModelRouter(),
    )
    _collect(client, [{"role": "user", "content": "hello, simple question"}])

    assert seen["model"] == "openai/gpt-4o-mini"


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