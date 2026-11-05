"""Tests for the intake case runner.

The runner drives the real agent exactly as production does, replays the case's
conversation context, and normalizes the event stream into an
``EvaluationContext`` — in particular keeping the one-time greeting apart from
the answer.

The LLM is stubbed at the session's client, so no network or API key is used and
the real ``IntakeAgent`` (and the shared agentic loop) still runs.
"""

from __future__ import annotations

import asyncio

import pytest

from agent.session import Session
from client.response import StreamEvent, StreamEventType, TextDelta, TokenUsage
from evaluation.adapters.agent_case_runner import AgentCaseRunner
from evaluation.datasets import EvalCase

QUESTION = "Thanks. What is the date of your last visit?"


def _chat(*texts: str):
    """Stub chat_completion; each call consumes the next scripted reply."""
    script = list(texts) or [QUESTION]
    state = {"turn": 0}

    async def chat(messages, model=None, tools=None, tool_choice=None, **kwargs):
        text = script[min(state["turn"], len(script) - 1)]
        state["turn"] += 1
        yield StreamEvent(
            type=StreamEventType.TEXT_DELTA, text_delta=TextDelta(content=text)
        )
        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason="stop",
            usage=TokenUsage(),
        )

    return chat


def _recording_chat(sink: list, *texts: str):
    """Like ``_chat``, but records the messages sent to the model."""
    inner = _chat(*texts)

    async def chat(messages, model=None, tools=None, tool_choice=None, **kwargs):
        sink.append(messages)
        async for event in inner(
            messages, model=model, tools=tools, tool_choice=tool_choice, **kwargs
        ):
            yield event

    return chat


def _factory(config, *texts):
    """Session factory whose LLM client is stubbed."""
    def make(cfg):
        session = Session(cfg)
        session.client.chat_completion = _chat(*texts)
        return session

    return make


def _recording_factory(config, sink, *texts):
    def make(cfg):
        session = Session(cfg)
        session.client.chat_completion = _recording_chat(sink, *texts)
        return session

    return make


def _run(runner, case):
    return asyncio.run(runner.run_case(case))


def _case(input_text="I'm Maria Garcia.", context=None, **overrides) -> EvalCase:
    return EvalCase(
        test_case_id=overrides.pop("test_case_id", "test-case"),
        scenario=overrides.pop("scenario", "Normal opening"),
        context=context or [],
        input=input_text,
        expected_output=overrides.pop("expected_output", "Ask one question."),
        critical_checks=overrides.pop("critical_checks", ["one_question_only"]),
        **overrides,
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_keeps_the_greeting_apart_from_the_answer(config):
    runner = AgentCaseRunner(config, session_factory=_factory(config, QUESTION))

    context = _run(runner, _case())

    assert context.metadata["greeting"] is not None
    assert context.metadata["greeting"].startswith(
        "Hello, I'm your medical intake assistant"
    )
    assert context.answer == QUESTION
    assert QUESTION not in context.metadata["greeting"]
    assert context.error is None


def test_full_text_includes_greeting_and_answer(config):
    runner = AgentCaseRunner(config, session_factory=_factory(config, QUESTION))

    context = _run(runner, _case())

    # The old adapter exposed greeting and answer as a single ``full_text``;
    # rebuild it locally from the context's parts.
    full_text = " ".join(
        part
        for part in (context.metadata["greeting"] or "", context.output)
        if part
    ).strip()

    assert "medical intake assistant" in full_text
    assert QUESTION in full_text


def test_records_turns_events_and_duration(config):
    runner = AgentCaseRunner(config, session_factory=_factory(config, QUESTION))

    case = _case()
    context = _run(runner, case)

    assert len(case.prompts) == 1
    assert context.metadata["history_turns"] == 0
    assert context.metadata["event_count"] >= 2
    assert context.duration_ms >= 0


# ---------------------------------------------------------------------------
# Context replay
# ---------------------------------------------------------------------------


def test_replays_the_case_context_before_the_evaluated_turn(config):
    sink: list = []
    runner = AgentCaseRunner(config, session_factory=_recording_factory(config, sink))

    case = _case(
        input_text="Actually, it started about two weeks ago.",
        context=[
            "Assistant: When did the pain start?",
            "Patient: Two days ago.",
        ],
    )

    context = _run(runner, case)

    assert context.metadata["history_turns"] == 2

    messages = sink[0]
    assert messages[0]["role"] == "system"
    contents = " ".join(str(m.get("content", "")) for m in messages)
    assert "When did the pain start?" in contents
    assert "Two days ago." in contents
    # The evaluated turn is last.
    assert messages[-1]["role"] == "user"
    assert "two weeks ago" in str(messages[-1]["content"])


def test_a_replayed_conversation_suppresses_the_greeting(config):
    """Mid-conversation cases must not get a synthetic greeting first."""
    runner = AgentCaseRunner(config, session_factory=_factory(config, QUESTION))

    case = _case(context=["Assistant: What's your full name?"])

    context = _run(runner, case)

    assert context.metadata["history_turns"] == 1
    assert context.metadata["greeting"] is None


def test_prose_context_is_not_replayed(config):
    sink: list = []
    runner = AgentCaseRunner(config, session_factory=_recording_factory(config, sink))

    case = _case(context=["The agent has collected all required categories."])

    context = _run(runner, case)

    assert context.metadata["history_turns"] == 0
    # Fresh session, so the greeting is still expected.
    assert context.metadata["greeting"] is not None


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def test_detects_completion(config):
    completion = "Thank you. Your intake is complete."
    runner = AgentCaseRunner(config, session_factory=_factory(config, completion))

    assert _run(runner, _case()).ended is True


def test_does_not_report_completion_for_a_normal_reply(config):
    runner = AgentCaseRunner(config, session_factory=_factory(config, QUESTION))

    assert _run(runner, _case()).ended is False


def test_records_an_agent_error_without_raising(config):
    def factory(cfg):
        session = Session(cfg)

        async def boom(messages, model=None, tools=None, tool_choice=None, **kwargs):
            raise RuntimeError("judge unavailable")
            yield  # pragma: no cover - keeps this an async generator

        session.client.chat_completion = boom
        return session

    runner = AgentCaseRunner(config, session_factory=factory)

    context = _run(runner, _case())

    assert context.error is not None
    assert context.error
    assert "judge unavailable" in context.error


def test_closes_the_llm_client(config):
    closed: list[bool] = []

    def factory(cfg):
        session = Session(cfg)
        session.client.chat_completion = _chat(QUESTION)

        async def fake_close():
            closed.append(True)

        session.client.close = fake_close
        return session

    _run(AgentCaseRunner(config, session_factory=factory), _case())

    assert closed == [True]


def test_each_case_gets_a_fresh_session(config):
    sessions: list[Session] = []

    def factory(cfg):
        session = Session(cfg)
        session.client.chat_completion = _chat(QUESTION)
        sessions.append(session)
        return session

    runner = AgentCaseRunner(config, session_factory=factory)

    _run(runner, _case())
    _run(runner, _case())

    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]
