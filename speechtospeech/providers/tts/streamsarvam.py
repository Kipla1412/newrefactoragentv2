from __future__ import annotations
import asyncio
from sarvamai import AsyncSarvamAI

from utils.logging_config import get_logger

logger = get_logger(__name__)


class SarvamStreamingTTSProvider:

    def __init__(
        self,
        api_key: str,
        language_code: str = "en-IN",
        speaker: str = "neha",
        model: str = "bulbul:v3",
    ):
        self.client = AsyncSarvamAI(api_subscription_key=api_key)
        self.language_code = language_code
        self.speaker = speaker
        self.model = model

        self.ctx = None
        self.socket = None
        # Prevent race conditions when background tasks interrupt each other
        self.lock = asyncio.Lock()

    async def connect(self):
        """Establishes a clean connection state, closing old contexts safely."""
        async with self.lock:
            await self._cleanup()

            logger.info("Connecting to Sarvam TTS")
            try:
                self.ctx = self.client.text_to_speech_streaming.connect(
                    model=self.model,
                    send_completion_event=True
                )
                self.socket = await self.ctx.__aenter__()

                await self.socket.configure(
                    target_language_code=self.language_code,
                    speaker=self.speaker,
                    speech_sample_rate=24000,
                    output_audio_codec="wav",
                )
                logger.info("TTS connected and configured")
            except Exception as e:
                logger.warning("Failed to build TTS socket connection layout: %s", e)
                await self._cleanup()

    async def reconnect(self):
        """Close the current connection and open a fresh one.

        Sarvam finalizes a synthesis session after each ``flush()`` that drains
        to the completion event. Sending more text afterward produces no audio,
        so every new synthesis turn (greeting or response) must start with a
        clean connection.
        """
        async with self.lock:
            await self._cleanup()

        await self.connect()


    # Add this method to your SarvamStreamingTTSProvider
    async def update_config(self, language_code: str, speaker: str):
        self.language_code = language_code
        self.speaker = speaker
        if self.socket:
            await self.socket.configure(
                target_language_code=self.language_code,
                speaker=self.speaker,
                speech_sample_rate=24000,
                output_audio_codec="wav",
            )

    async def ensure_connected(self):
        """Lazy connection guard utility."""
        # FIX: Removed the false-positive 'closed' attribute check.
        # If self.socket exists, we trust it and let exceptions handle genuine drops.
        if self.socket is None:
            await self.connect()

    async def send_text(self, text: str):
        if not text.strip():
            return

        try:
            await self.ensure_connected()
            if self.socket:
                await self.socket.convert(text)
        except Exception as e:
            logger.warning("Convert error, attempting reconnection recovery: %s", e)
            await self.connect()
            if self.socket:
                await self.socket.convert(text)

    async def receive_audio(self):
        try:
            await self.ensure_connected()
            if not self.socket:
                return None

            res = await self.socket.recv()

            # Catch raw API error drops gracefully
            if res and getattr(res, "type", None) == "error":
                code = getattr(res.data, "code", None) if hasattr(res, "data") else None
                if code == 408:
                    logger.info("TTS idle session timeout notification received (408)")
                    await self.close()
            return res
        except Exception as e:
            logger.warning("TTS receive error: %s", e)
            if "closed" in str(e).lower() or "1000" in str(e) or "none" in str(e).lower():
                logger.info("TTS connection dead, cleaning up references")
                await self.close()
            return None

    async def flush(self):
        # FIX: Cleaned up the check here as well to protect the streaming flush
        if self.socket:
            try:
                await self.socket.flush()
            except Exception:
                pass

    async def _cleanup(self):
        """Internal low-level resetting utility."""
        if self.ctx:
            try:
                await self.ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self.ctx = None
        self.socket = None

    async def close(self):
        """Safely cleans up execution variables and exits active connection contexts."""
        async with self.lock:
            await self._cleanup()
            logger.info("TTS connection reference cleared")
