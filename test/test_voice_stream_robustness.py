"""Voice stream robustness: a stalled TTS must not hang a turn, and an
interrupted turn must not leave an unanswered patient message in history.
"""

import asyncio
import base64
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from agent.events import AgentEvent, AgentType
from customagents.voiceagent.voiceagent import (
    VoiceSession,
    _drop_unanswered_user_message,
)


@dataclass
class _Event:
    type: str
    data: object


class _AudioResult:
    def __init__(self, audio):
        self.data = SimpleNamespace(audio=audio)


class _ScriptedTTS:
    def __init__(self, results):
        self._results = list(results)
        self.reconnects = 0
        self.sent = []

    async def reconnect(self):
        self.reconnects += 1

    async def send_text(self, text):
        self.sent.append(text)

    async def flush(self):
        pass

    async def receive_audio(self):
        if not self._results:
            return None
        return self._results.pop(0)


class _SilentThenWorksTTS:
    """First synthesis produces no audio; the retry produces some."""

    def __init__(self):
        self.reconnects = 0
        self.sent = []
        self._step = 0

    async def reconnect(self):
        self.reconnects += 1

    async def send_text(self, text):
        self.sent.append(text)

    async def flush(self):
        pass

    async def receive_audio(self):
        self._step += 1
        if self._step == 1:
            # Attempt 1 ends immediately with no audio at all.
            return _Event(type="event", data=SimpleNamespace(event_type="final"))
        if self._step == 2:
            return _AudioResult(base64.b64encode(b"RETRY-AUDIO").decode())
        if self._step == 3:
            return _Event(type="event", data=SimpleNamespace(event_type="final"))
        return None


class _StallingTTS:
    async def receive_audio(self):
        await asyncio.sleep(3600)


async def _collect(agen):
    return [item async for item in agen]


@pytest.mark.asyncio
async def test_drain_tts_yields_audio_and_stops_on_the_final_event():
    payload = base64.b64encode(b"PCMDATA").decode()
    trailing = base64.b64encode(b"SHOULD-NOT-APPEAR").decode()
    tts = _ScriptedTTS([
        _AudioResult(payload),
        _Event(type="event", data=SimpleNamespace(event_type="final")),
        _AudioResult(trailing),
    ])
    session = VoiceSession(agent=None, tts=tts)

    chunks = await _collect(session._drain_tts())

    assert chunks == [b"PCMDATA"]


@pytest.mark.asyncio
async def test_drain_tts_returns_when_the_tts_socket_stalls():
    session = VoiceSession(agent=None, tts=_StallingTTS())
    session.TTS_DRAIN_TIMEOUT_SECONDS = 0.05

    chunks = await asyncio.wait_for(_collect(session._drain_tts()), timeout=1.0)

    assert chunks == []


class _Msg:
    def __init__(self, role):
        self.role = role


class _FakeContextManager:
    def __init__(self, messages):
        self._messages = messages


class _FakeSession:
    def __init__(self, messages):
        self.context_manager = _FakeContextManager(messages)


def _roles(session):
    return [m.role for m in session.context_manager._messages]


def test_drop_unanswered_user_message_removes_a_trailing_user_turn():
    session = _FakeSession([_Msg("assistant"), _Msg("user")])

    _drop_unanswered_user_message(session)

    assert _roles(session) == ["assistant"]


def test_drop_unanswered_user_message_keeps_an_answered_turn():
    session = _FakeSession([_Msg("user"), _Msg("assistant")])

    _drop_unanswered_user_message(session)

    assert _roles(session) == ["user", "assistant"]


def test_drop_unanswered_user_message_tolerates_missing_state():
    _drop_unanswered_user_message(SimpleNamespace())
    _drop_unanswered_user_message(None)


@pytest.mark.asyncio
async def test_text_to_audio_retries_once_when_no_audio_is_produced():
    tts = _SilentThenWorksTTS()
    session = VoiceSession(agent=None, tts=tts)

    chunks = await _collect(session.text_to_audio("Hello there"))

    assert chunks == [b"RETRY-AUDIO"]
    assert tts.reconnects == 2                    # initial connect + retry
    assert tts.sent == ["Hello there", "Hello there"]


@pytest.mark.asyncio
async def test_text_to_audio_does_not_retry_when_audio_is_produced():
    tts = _ScriptedTTS([
        _AudioResult(base64.b64encode(b"PCM").decode()),
        _Event(type="event", data=SimpleNamespace(event_type="final")),
    ])
    session = VoiceSession(agent=None, tts=tts)

    chunks = await _collect(session.text_to_audio("Hello"))

    assert chunks == [b"PCM"]
    assert tts.reconnects == 1


@pytest.mark.asyncio
async def test_resynthesize_does_nothing_for_blank_text():
    tts = _SilentThenWorksTTS()
    session = VoiceSession(agent=None, tts=tts)

    chunks = await _collect(session._resynthesize("   "))

    assert chunks == []
    assert tts.reconnects == 0


class _TextAgent:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def run(self, transcript):
        for chunk in self._chunks:
            yield AgentEvent.text_delta(chunk, AgentType.VOICE_INTAKE)


@pytest.mark.asyncio
async def test_tts_socket_is_not_opened_when_the_agent_says_nothing():
    """An idle socket is what Sarvam hangs up on, so don't open one for nothing."""
    tts = _ScriptedTTS([])
    session = VoiceSession(agent=_TextAgent([]), tts=tts)

    frames = await _collect(
        session.process_transcript_to_audio(
            "hello", target_language="en-IN", source_language="en-IN"
        )
    )

    assert tts.reconnects == 0
    assert frames == []


@pytest.mark.asyncio
async def test_tts_socket_opens_lazily_on_the_first_chunk():
    tts = _ScriptedTTS([
        _AudioResult(base64.b64encode(b"PCM").decode()),
        _Event(type="event", data=SimpleNamespace(event_type="final")),
    ])
    session = VoiceSession(
        agent=_TextAgent(["Hello there, how are you today? "]), tts=tts
    )

    frames = await _collect(
        session.process_transcript_to_audio(
            "hi", target_language="en-IN", source_language="en-IN"
        )
    )

    assert tts.reconnects == 1
    assert any(frame.get("type") == "audio" for frame in frames)