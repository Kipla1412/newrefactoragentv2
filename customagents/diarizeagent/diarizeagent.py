from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncGenerator, Callable, Awaitable

from agent.agent import Agent
from agent.events import AgentEvent, AgentEventType, AgentType
from client.response import StreamEventType
from customagents.diarizeagent.diarizeprompt import DIARIZE_ROLE_MAP_PROMPT
from customagents.voiceagent.voiceagent import translate_text
from speechtospeech.hybrid.orchestrator import DiarizationOrchestrator
from speechtospeech.hybrid.stt_provider import HybridSTTProvider, _relabel_segments
from speechtospeech.webvad import VoiceActivityDetector
from utils.logging_config import get_logger

logger = get_logger(__name__)


class DiarizeAgent(Agent):
    """Agent that captures audio, streams partial transcripts, and returns a
    fully diarized (speaker-attributed) transcript via batch processing.

    ``run()`` accepts an async generator of audio chunks and yields
    AgentEvent objects for lifecycle tracking.  The optional
    ``on_partial`` callback is used for real-time transcript forwarding
    (e.g. to WebSocket).
    """

    def __init__(self, config, session=None):
        super().__init__(config, DIARIZE_ROLE_MAP_PROMPT, AgentType.DIARIZE)
        if session:
            self.session = session

    async def run(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        *,
        silence_timeout: float = 15.0,
        max_session_duration: float = 600.0,
        num_speakers: int = 2,
        on_partial: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run the hybrid capture + diarization pipeline.

        Args:
            audio_stream: Async generator yielding raw PCM16 audio chunks.
            silence_timeout: Seconds of silence before auto-ending session.
            max_session_duration: Maximum capture duration in seconds.
            num_speakers: Expected speakers for diarization.
            on_partial: Async callback for forwarding partial transcripts.

        Yields:
            AgentEvent.agent_start / agent_end / agent_error.
            AgentEvent.text_complete with the final diarized transcript.
        """
        yield AgentEvent.agent_start(
            "Starting hybrid diarization session",
            self.agent_type,
        )

        self.session.agent_name = self.__class__.__name__
        self.session.start_mlflow_run("hybrid_diarization")
        session_start = time.monotonic()

        try:
            provider = HybridSTTProvider(
                api_key=self.config.sarvam_api_key,
                streaming_model=self.config.sarvam_stt_model,
                batch_model=getattr(self.config, "hybrid_batch_model", None)
                            or self.config.sarvam_stt_model,
                language_code="unknown",
                sample_rate=16000,
                num_speakers=num_speakers,
                session_id=self.session.session_id,
            )

            vad = VoiceActivityDetector(
                threshold=0.05,
                silence_duration=silence_timeout,
            )

            orchestrator = DiarizationOrchestrator(
                provider=provider,
                vad=vad,
                silence_timeout=silence_timeout,
                max_session_duration=max_session_duration,
                on_partial=on_partial,
            )

            await provider.connect()
            logger.info("DiarizeAgent: provider connected")

            # Wrap the orchestration in an MLflow span
            async with self.session.trace_agent_run("hybrid_diarization"):
                segments = await orchestrator.run(audio_stream)

            final_transcript = await self._label_segments(segments, provider)

            if final_transcript and self.session.client:
                try:
                    final_transcript = await translate_text(
                        self.session.client,
                        final_transcript,
                        target_language="en",
                        source_language="unknown",
                    )
                    logger.info("DiarizeAgent: transcript translated to English")
                except Exception as e:
                    logger.warning("DiarizeAgent: translation failed: %s", e)

            session_duration = time.monotonic() - session_start
            transcript_len = len(final_transcript or "")

            # Log metrics and params
            if self.session.mlflow_tracker and self.session.mlflow_run:
                try:
                    self.session.mlflow_tracker.log_metrics({
                        "session_duration_seconds": round(session_duration, 1),
                        "transcript_length_chars": transcript_len,
                        "num_speakers": num_speakers,
                    })
                    self.session.mlflow_tracker.log_param(
                        "silence_timeout", silence_timeout
                    )
                    self.session.mlflow_tracker.log_param(
                        "max_session_duration", max_session_duration
                    )
                except Exception as e:
                    logger.warning("DiarizeAgent: failed to log MLflow metrics: %s", e)

            logger.info(
                "DiarizeAgent: session complete — duration=%.1fs transcript=%d chars",
                session_duration,
                transcript_len,
            )

            yield AgentEvent.text_complete(
                final_transcript or "(no transcript produced)",
                agent=self.agent_type,
            )

            self.session.context_manager.add_user_message(
                "Diarization session completed"
            )
            self.session.context_manager.add_assistant_message(
                final_transcript or "(no transcript produced)"
            )

            return

        except Exception as exc:
            logger.exception("DiarizeAgent: session failed")
            if self.session.mlflow_tracker and self.session.mlflow_run:
                try:
                    self.session.mlflow_tracker.log_param("error", str(exc)[:200])
                except Exception:
                    pass
            yield AgentEvent.agent_error(str(exc))
            raise

        finally:
            self.session.end_mlflow_run()

    async def _label_segments(
        self,
        segments: list[dict[str, Any]],
        provider: HybridSTTProvider,
    ) -> str:
        """Label speaker IDs as Doctor/Patient using the LLM role-mapping prompt.

        Falls back to the streaming transcript when no diarized segments are
        available, and to the heuristic ``_relabel_segments`` if the LLM fails
        or returns an unusable response.
        """
        if not segments:
            logger.info("DiarizeAgent: no diarized segments, using streaming transcript")
            return provider.get_streaming_transcript()

        transcript = self._build_segment_transcript(segments)

        try:
            role_map = await self._call_role_mapper(transcript)
        except Exception as e:
            logger.warning("DiarizeAgent: LLM role mapping failed: %s", e)
            role_map = None

        if not role_map:
            logger.info("DiarizeAgent: falling back to heuristic speaker relabeling")
            return _relabel_segments(segments)

        labeled: list[str] = []
        for seg in segments:
            speaker_id = str(seg.get("speaker_id", ""))
            speaker = role_map.get(speaker_id) or role_map.get(f"SPEAKER_{speaker_id}")
            if not speaker:
                speaker = f"Speaker_{speaker_id}"
            labeled.append(f"{speaker}: {seg.get('text', '')}")

        return "\n".join(labeled)

    @staticmethod
    def _build_segment_transcript(segments: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for seg in segments:
            speaker_id = str(seg.get("speaker_id", "0"))
            lines.append(f"SPEAKER_{speaker_id}: {seg.get('text', '')}")
        return "\n".join(lines)

    async def _call_role_mapper(self, transcript: str) -> dict[str, str] | None:
        """Call the LLM with DIARIZE_ROLE_MAP_PROMPT and parse the JSON result."""
        messages = [
            {"role": "system", "content": DIARIZE_ROLE_MAP_PROMPT},
            {"role": "user", "content": transcript},
        ]

        response_text = ""
        async for event in self.session.client.chat_completion(messages, stream=False):
            if event.type == StreamEventType.MESSAGE_COMPLETE and event.text_delta:
                response_text += event.text_delta.content

        parsed = _parse_role_mapping(response_text)
        if not parsed:
            return None

        role_map: dict[str, str] = {}
        doctor_id = parsed.get("doctor_speaker_id")
        patient_id = parsed.get("patient_speaker_id")

        if doctor_id and doctor_id != "NONE":
            role_map[str(doctor_id)] = "Doctor"
        if patient_id and patient_id != "NONE":
            role_map[str(patient_id)] = "Patient"

        return role_map if role_map else None

    async def __aenter__(self):
        if not self.session.context_manager:
            await self.session.initialize()
        return self


def _parse_role_mapping(raw_text: str) -> dict[str, Any] | None:
    """Parse the LLM role-mapping JSON, tolerating markdown fences."""
    if not raw_text:
        return None

    clean = re.sub(r"```json\s*|\s*```", "", raw_text).strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first JSON object if the model added surrounding text.
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return None
