"""Tests for voice-turn observability.

The voice path uses **Sarvam** for STT and TTS, while the agent's reasoning and
translation go through the configured LLM. These tests pin that distinction:
turns record latency and volume, and the recorded model parameters are the
Sarvam ones rather than an LLM model name.

No transcript or audio content may be recorded.
"""

from __future__ import annotations

import asyncio
import base64
import time
from types import SimpleNamespace

import pytest

from agent.events import AgentEvent, AgentType
from customagents.voiceagent.voiceagent import VoiceSession


class _Tracker:
    """Records what would be sent to MLflow."""

    def __init__(self) -> None:
        self.metrics: dict = {}
        self.params: dict = {}

    def log_metrics(self, metrics: dict) -> None:
        self.metrics.update(metrics)

    def log_param(self, name: str, value) -> None:
        self.params[name] = value


def _session(tracker=None, config=None) -> VoiceSession:
    agent = SimpleNamespace(
        config=config or SimpleNamespace(
            sarvam_stt_model="saaras:v3",
            sarvam_tts_model="bulbul:v3",
            sarvam_speaker="neha",
        ),
        session=SimpleNamespace(mlflow_tracker=tracker),
    )
    return VoiceSession(agent=agent, tts=None)


def _record(session: VoiceSession, **overrides) -> None:
    defaults = dict(
        turn_start=time.monotonic() - 1.5,
        first_token_at=time.monotonic() - 1.3,
        first_audio_at=time.monotonic() - 0.6,
        text_chunks=3,
        audio_chunks=7,
        audio_bytes=12_345,
        resynthesized=False,
        end_of_conversation=True,
        source_language="ta-IN",
        target_language="en-IN",
    )
    defaults.update(overrides)
    session._record_turn(**defaults)


# ---------------------------------------------------------------------------
# _record_turn
# ---------------------------------------------------------------------------


def test_records_turn_metrics():
    tracker = _Tracker()

    _record(_session(tracker))

    assert tracker.metrics["text_chunks"] == 3
    assert tracker.metrics["audio_chunks"] == 7
    assert tracker.metrics["audio_bytes"] == 12_345
    assert tracker.metrics["end_of_conversation"] == 1
    assert tracker.metrics["resynthesized"] == 0
    assert 0 < tracker.metrics["turn_seconds"] < 5


def test_records_first_token_and_first_audio_latency():
    tracker = _Tracker()
    start = time.monotonic() - 2.0

    _record(
        _session(tracker),
        turn_start=start,
        first_token_at=start + 0.25,
        first_audio_at=start + 1.1,
    )

    assert tracker.metrics["llm_first_token_seconds"] == pytest.approx(0.25, abs=0.05)
    assert tracker.metrics["tts_first_audio_seconds"] == pytest.approx(1.1, abs=0.05)


def test_records_the_sarvam_models_not_an_llm_model():
    tracker = _Tracker()

    _record(_session(tracker))

    assert tracker.params["stt_provider"] == "sarvam"
    assert tracker.params["stt_model"] == "saaras:v3"
    assert tracker.params["tts_provider"] == "sarvam"
    assert tracker.params["tts_model"] == "bulbul:v3"
    assert tracker.params["tts_speaker"] == "neha"


def test_records_the_language_pair():
    tracker = _Tracker()

    _record(_session(tracker), source_language="ta-IN", target_language="en-IN")

    assert tracker.params["source_language"] == "ta-IN"
    assert tracker.params["target_language"] == "en-IN"


def test_marks_a_resynthesised_turn():
    tracker = _Tracker()

    _record(_session(tracker), resynthesized=True)

    assert tracker.metrics["resynthesized"] == 1


def test_is_a_noop_without_a_tracker():
    """Tracking off is the default; recording must simply do nothing."""
    _record(_session(tracker=None))  # must not raise


def test_survives_an_agent_without_config():
    """Telemetry must never break a turn, even with a stripped-down agent."""
    session = VoiceSession(agent=SimpleNamespace(session=SimpleNamespace()), tts=None)

    _record(session, first_token_at=None, first_audio_at=None)  # must not raise


def test_omits_latency_keys_when_nothing_was_produced():
    tracker = _Tracker()

    _record(_session(tracker), first_token_at=None, first_audio_at=None)

    assert "llm_first_token_seconds" not in tracker.metrics
    assert "tts_first_audio_seconds" not in tracker.metrics


# ---------------------------------------------------------------------------
# End to end through process_transcript_to_audio
# ---------------------------------------------------------------------------


class _TextAgent:
    """Minimal stand-in for a voice agent."""

    def __init__(self, config) -> None:
        self.config = config
        self.session = SimpleNamespace()

    async def run(self, message):
        yield AgentEvent.text_delta(
            "Hello there. How are you today?", agent=AgentType.VOICE_CONSULT
        )
        yield AgentEvent.status_end(agent=AgentType.VOICE_CONSULT)


class _AudioTTS:
    """One audio chunk, then the stream ends."""

    def __init__(self) -> None:
        self.reconnects = 0
        self.sent: list[str] = []
        self.flushes = 0
        self._chunks = 1

    async def reconnect(self):
        self.reconnects += 1

    async def send_text(self, text: str):
        self.sent.append(text)

    async def flush(self):
        self.flushes += 1

    async def receive_audio(self):
        if self._chunks:
            self._chunks -= 1
            return SimpleNamespace(
                type="chunk",
                data=SimpleNamespace(
                    audio=base64.b64encode(b"abcdef").decode("utf-8")
                ),
            )
        return None


def test_a_full_turn_records_metrics_end_to_end():
    tracker = _Tracker()
    config = SimpleNamespace(
        sarvam_stt_model="saaras:v3",
        sarvam_tts_model="bulbul:v3",
        sarvam_speaker="neha",
    )
    agent = _TextAgent(config)
    agent.session = SimpleNamespace(mlflow_tracker=tracker)
    tts = _AudioTTS()
    session = VoiceSession(agent=agent, tts=tts)

    async def _run():
        return [item async for item in session.process_transcript_to_audio("hello")]

    items = asyncio.run(_run())

    # The turn produced speech and audio.
    assert any(item["type"] == "text" for item in items)
    assert any(item["type"] == "audio" for item in items)
    assert tts.sent, "text was never sent to TTS"

    # And it was recorded, with the Sarvam configuration.
    assert tracker.metrics["audio_chunks"] == 1
    assert tracker.metrics["audio_bytes"] == len(b"abcdef")
    assert tracker.metrics["text_chunks"] >= 1
    assert tracker.metrics["end_of_conversation"] == 1
    assert "llm_first_token_seconds" in tracker.metrics
    assert "tts_first_audio_seconds" in tracker.metrics
    assert tracker.params["stt_model"] == "saaras:v3"
    assert tracker.params["tts_model"] == "bulbul:v3"
    assert tracker.params["tts_speaker"] == "neha"


def test_a_silent_turn_opens_no_tts_socket_and_records_it():
    """No agent text means no TTS, no audio, and the turn still records."""
    tracker = _Tracker()
    config = SimpleNamespace(
        sarvam_stt_model="saaras:v3",
        sarvam_tts_model="bulbul:v3",
        sarvam_speaker="neha",
    )

    class _SilentAgent(_TextAgent):
        async def run(self, message):
            if False:  # pragma: no cover - keeps this an async generator
                yield
            return

    agent = _SilentAgent(config)
    agent.session = SimpleNamespace(mlflow_tracker=tracker)
    tts = _AudioTTS()
    session = VoiceSession(agent=agent, tts=tts)

    async def _run():
        return [item async for item in session.process_transcript_to_audio("hello")]

    asyncio.run(_run())

    assert tts.reconnects == 0
    assert tracker.metrics["audio_chunks"] == 0
    assert tracker.metrics["audio_bytes"] == 0
