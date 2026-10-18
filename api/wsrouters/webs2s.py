from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import base64
import re
import time
from customagents.factory import AgentFactory
from customagents.sessionmanager import SessionManager
from customagents.voiceagent.voiceagent import VoiceSession
from customagents.voiceagent.voiceintake import VoiceIntakeAgent
from customagents.voiceagent.voiceconsult import VoiceConsultAgent
from speechtospeech.providers.stt.streamsarvam import SarvamStreamingSTTProvider
from speechtospeech.providers.tts.streamsarvam import SarvamStreamingTTSProvider

from api.auth import decode_token
from utils.logging_config import get_logger, prompts_enabled

logger = get_logger(__name__)
router = APIRouter()

_SCRIPT_MAP = [
    (r'[\u0B80-\u0BFF]', "ta-IN"),
]

_GREETINGS = {
    "voice_intake": (
        "Hello, I'm your medical intake assistant. I'll gather some "
        "information before your visit."
    ),
    "voice_consult": (
        "Hi, I'm your medical assistant. What problem or symptom would "
        "you like to discuss today?"
    ),
}


def detect_language(text: str) -> str | None:
    if not text:
        return None
    for pattern, lang_code in _SCRIPT_MAP:
        if re.search(pattern, text):
            return lang_code
    return None


async def _authenticate_ws(ws: WebSocket):
    token = None
    auth_header = ws.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    if not token:
        token = ws.query_params.get("token")
    if not token:
        logger.warning("WebSocket handshake rejected: Missing authentication token.")
        await ws.close(code=1008)
        return None
    try:
        user = decode_token(token)
        ws.state.user = user
        return user
    except Exception as e:
        logger.error(f"WebSocket Auth Failed: {e}")
        await ws.close(code=1008)
        return None


@router.websocket("/audio")
async def audio_intake_stream(ws: WebSocket):
    await _voice_stream(ws, "voice_intake")


@router.websocket("/consultaudio")
async def audio_consult_stream(ws: WebSocket):
    await _voice_stream(ws, "voice_consult")


async def _voice_stream(ws: WebSocket, agent_type: str):
    user = await _authenticate_ws(ws)
    if user is None:
        return

    await ws.accept()
    label = "Voice Intake" if agent_type == "voice_intake" else "Voice Consult"
    logger.info("Session started", extra={"fields": {"agent": agent_type}})

    user_id = str(user.get("id", "unknown_user"))
    config = ws.app.state.config
    session_manager: SessionManager = ws.app.state.session_manager

    patient_language = None
    ws_write_lock = asyncio.Lock()
    connection_open = True

    async def safe_send_json(data):
        nonlocal connection_open
        async with ws_write_lock:
            if not connection_open:
                return
            try:
                await ws.send_json(data)
            except WebSocketDisconnect:
                connection_open = False
                raise
            except Exception as e:
                connection_open = False
                logger.error(f"WebSocket send failed: {e}")
                raise

    async def safe_close(code: int = 1000, reason: str = ""):
        nonlocal connection_open
        async with ws_write_lock:
            if not connection_open:
                return
            try:
                await ws.close(code=code, reason=reason)
            except Exception as e:
                logger.debug(f"WebSocket close failed: {e}")
            finally:
                connection_open = False

    session = await session_manager.get_session(user_id, config)
    if not session.context_manager:
        await session.initialize()

    # Client-supplied patient context (for example "name = X, age = 23") is
    # sent as a WebSocket query param. When present, the agent treats those
    # demographics as already collected instead of asking for them again.
    patient_context = ws.query_params.get("patient_context")
    if patient_context:
        session.metadata["patient_context"] = patient_context
    else:
        session.metadata.pop("patient_context", None)

    voice_agent = AgentFactory.create(agent_type, config, session=session)

    stt = SarvamStreamingSTTProvider(
        api_key=config.sarvam_api_key,
        model=config.sarvam_stt_model
    )

    tts = SarvamStreamingTTSProvider(
        api_key=config.sarvam_api_key,
        model=config.sarvam_tts_model,
        speaker=config.sarvam_speaker
    )

    voice_session = VoiceSession(agent=voice_agent, tts=tts)

    logger.info("Pre-warming streaming providers")
    await asyncio.gather(stt.connect(), tts.connect())

    # Send the initial greeting and mark the agent ready for input.
    greeting = _GREETINGS.get(agent_type)
    if greeting:
        try:
            # Record the greeting as the assistant's first turn so the agent
            # does not introduce itself a second time on the first user turn.
            if session.context_manager and not session.metadata.get("greeted"):
                session.context_manager.add_assistant_message(greeting)
                session.metadata["greeted"] = True
            await safe_send_json({"type": "status", "status": "greeting"})
            await safe_send_json({"type": "text", "text": greeting})
            async for audio_bytes in voice_session.text_to_audio(greeting):
                audio_payload = base64.b64encode(audio_bytes).decode("utf-8")
                await safe_send_json({"type": "audio", "audio": audio_payload})
        except Exception as e:
            logger.error(f"Greeting playback failed: {e}")

    await safe_send_json({"type": "status", "status": "ready"})

    transcript_queue = asyncio.Queue()
    active_response_task = None

    try:
        async def receive_audio():
            try:
                while True:
                    audio_chunk = await ws.receive_bytes()
                    session.last_activity = time.time()
                    await stt.send_audio(audio_chunk)
            except WebSocketDisconnect:
                raise
            except Exception as e:
                logger.error(f"Error reading client audio bytes: {e}")

        async def generate_and_send_response(transcript_text: str, source_language: str):
            try:
                await safe_send_json({"type": "status", "status": "thinking"})
                text_buffer = ""
                audio_buffer_len = 0
                ended = False
                async for response in voice_session.process_transcript_to_audio(
                    transcript_text,
                    target_language=patient_language,
                    source_language=source_language,
                ):
                    if not response:
                        continue

                    if response.get("type") == "text":
                        content = response.get("content")
                        text_buffer += content
                        await safe_send_json({"type": "text", "text": content})

                    elif response.get("type") == "audio":
                        audio_bytes = response.get("content")
                        if isinstance(audio_bytes, bytes):
                            audio_payload = base64.b64encode(audio_bytes).decode("utf-8")
                            audio_buffer_len += len(audio_payload)
                            await safe_send_json({"type": "audio", "audio": audio_payload})

                    elif response.get("type") == "status" and response.get("status") == "end":
                        # Same wire shape the text agents emit, so clients can
                        # handle voice and text conversations identically.
                        ended = True
                        await safe_send_json({
                            "type": "status_end",
                            "data": {"status": "end", "agent": agent_type},
                        })

                if text_buffer.strip():
                    if prompts_enabled():
                        logger.debug("Assistant text", extra={"fields": {"text": text_buffer.strip()}})
                if audio_buffer_len > 0:
                    logger.debug("Assistant audio", extra={"fields": {"bytes": audio_buffer_len}})

                if not ended:
                    await safe_send_json({"type": "status", "status": "ready"})

            except asyncio.CancelledError:
                logger.info("Response generation task explicitly cancelled.")
            except Exception as e:
                logger.error(f"Error in response generation: {e}")

        async def process_transcripts():
            nonlocal patient_language
            async for transcript_data in stt.stream_transcripts():
                raw_text = transcript_data["text"].strip()
                if not raw_text:
                    continue

                is_final = transcript_data.get("is_final", True)

                stt_language = transcript_data.get("language", "unknown")
                unicode_detected = detect_language(raw_text)

                detected = None
                detection_source = None

                if stt_language in ("ta-IN", "ta"):
                    detected = stt_language
                    detection_source = "sarvam"
                elif unicode_detected:
                    detected = unicode_detected
                    detection_source = "unicode"
                elif stt_language in ("en-IN", "en"):
                    detected = stt_language
                    detection_source = "sarvam"
                else:
                    detected = "en-IN"
                    detection_source = "default"

                if detected:
                    if patient_language != detected:
                        patient_language = detected
                        logger.info(
                            "Patient language detected",
                            extra={"fields": {"language": patient_language, "source": detection_source}},
                        )
                        try:
                            await tts.update_config(
                                language_code=patient_language,
                                speaker=config.sarvam_speaker
                            )
                        except Exception as e:
                            logger.error(f"TTS update failed: {e}")

                source_language = detected
                logger.debug(
                    "User turn received",
                    extra={"fields": {"user_id": user_id, "is_final": is_final}},
                )
                if prompts_enabled():
                    logger.debug("User transcript", extra={"fields": {"text": raw_text}})
                await safe_send_json({"type": "transcript", "text": raw_text})

                # Only final transcripts trigger an agent response. Partial
                # transcripts update the live display but must not restart the
                # agent/TTS pipeline (that churn tears down the TTS connection
                # before audio can complete).
                if is_final:
                    await transcript_queue.put((raw_text, source_language))

        async def dispatch_responses():
            nonlocal active_response_task
            while True:
                try:
                    transcript, source_language = await transcript_queue.get()
                except asyncio.CancelledError:
                    break

                try:
                    if active_response_task and not active_response_task.done():
                        logger.debug("Interrupting active speaking block for new phrase")
                        active_response_task.cancel()
                        try:
                            await active_response_task
                        except asyncio.CancelledError:
                            pass
                        try:
                            await tts.flush()
                        except Exception:
                            pass

                    active_response_task = asyncio.create_task(
                        generate_and_send_response(transcript, source_language)
                    )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Queue orchestration failure: {e}")
                finally:
                    transcript_queue.task_done()

        receive_task = asyncio.create_task(receive_audio())
        transcript_task = asyncio.create_task(process_transcripts())
        dispatch_task = asyncio.create_task(dispatch_responses())

        done, pending = await asyncio.wait(
            [receive_task, transcript_task, dispatch_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            task.result()

    except WebSocketDisconnect:
        logger.info("Client disconnected gracefully", extra={"fields": {"user_id": user_id}})
    except Exception as e:
        logger.exception(e)
        try:
            await safe_send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if active_response_task and not active_response_task.done():
            active_response_task.cancel()

        for task_var in ['receive_task', 'transcript_task', 'dispatch_task']:
            if task_var in locals():
                task_obj = locals()[task_var]
                if not task_obj.done():
                    task_obj.cancel()

        try:
            await stt.close()
            await tts.close()
        except Exception:
            pass

        try:
            await safe_close()
        except Exception:
            pass

        if voice_agent and voice_agent.session and voice_agent.session.client:
            await voice_agent.session.client.close()

        logger.info("Session closed safely", extra={"fields": {"agent": agent_type}})
