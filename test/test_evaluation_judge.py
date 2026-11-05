"""Tests for the DeepEval judge adapter.

The judge must use one pinned model and must **not** fall through the fallback
chain: a weak fallback answers evaluation prompts with unparseable text, which
DeepEval surfaces as "invalid JSON" with no clue about the real cause.
"""

from __future__ import annotations

import pytest

from client.response import StreamEvent, StreamEventType, TextDelta, TokenUsage
from evaluation.adapters.deepeval_judge import build_judge, deepeval_installed

pytestmark = pytest.mark.skipif(
    not deepeval_installed(), reason="deepeval is not installed"
)


class _Client:
    """Records the kwargs each completion is called with."""

    def __init__(self, reply: str = '{"score": 1}') -> None:
        self._reply = reply
        self.calls: list[dict] = []

    async def chat_completion(self, messages, **kwargs):
        self.calls.append(kwargs)
        yield StreamEvent(
            type=StreamEventType.TEXT_DELTA, text_delta=TextDelta(content=self._reply)
        )
        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason="stop",
            usage=TokenUsage(),
        )


def test_judge_does_not_use_the_fallback_chain():
    client = _Client()

    judge = build_judge(client, model="clinical_model")
    judge.generate("prompt")

    assert client.calls[0]["use_fallbacks"] is False


def test_judge_defaults_to_the_agents_model():
    judge = build_judge(_Client())

    assert judge.get_model_name() == "clinical_model"


def test_judge_uses_the_requested_model():
    client = _Client()

    judge = build_judge(client, model="fast")
    judge.generate("prompt")

    assert judge.get_model_name() == "fast"
    assert client.calls[0]["model"] == "fast"


def test_judge_returns_the_joined_completion():
    assert build_judge(_Client('{"a": 1}')).generate("prompt") == '{"a": 1}'


def test_judge_exposes_the_underlying_client():
    client = _Client()

    assert build_judge(client).load_model() is client
