from __future__ import annotations

import asyncio

from utils.logging_config import get_logger, prompts_enabled

logger = get_logger(__name__)


class UtteranceCoalescer:
    """Merge STT finals that belong to the same spoken utterance.

    Sarvam's VAD can split one sentence into several "final" transcripts (for
    example "I have a headache" followed by "and a fever"). Feeding each one to
    the agent makes it answer a fragment, and the router's barge-in then cancels
    that answer as soon as the next fragment arrives — so the beginning of the
    sentence is effectively ignored.

    Finals are buffered here and released only once the speaker has been quiet
    for ``silence_seconds``. Fragments are merged defensively: a cumulative
    transcript (one that already contains the buffered text) replaces the
    buffer, while a genuinely separate fragment is appended.
    """

    def __init__(self, silence_seconds: float = 0.35):
        self._silence = silence_seconds
        self._text = ""
        self._language: str | None = None
        self._task: asyncio.Task | None = None
        self._queue: asyncio.Queue = asyncio.Queue()

    @property
    def pending(self) -> str:
        return self._text

    @staticmethod
    def merge(previous: str, incoming: str) -> str:
        """Combine a buffered fragment with the next one."""
        if not previous:
            return incoming
        if incoming.startswith(previous):
            # Cumulative transcript — the new one supersedes the buffer.
            return incoming
        if previous.endswith(incoming):
            # Repeated tail — nothing new to add.
            return previous
        return f"{previous} {incoming}"

    def add(self, text: str, language: str | None = None) -> None:
        """Buffer a final transcript; release it after the silence window."""
        text = text.strip()
        if not text:
            return

        self._text = self.merge(self._text, text)
        self._language = language

        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.create_task(self._release_after_silence())

    async def _release_after_silence(self) -> None:
        try:
            await asyncio.sleep(self._silence)
        except asyncio.CancelledError:
            return

        text = self._text.strip()
        language = self._language
        self._text = ""
        self._language = None
        self._task = None

        if text:
            # Always record that an utterance was released; the text itself is
            # patient speech, so it only appears when LOG_PROMPTS is enabled.
            logger.debug(
                "Utterance ready",
                extra={"fields": {"chars": len(text), "language": language}},
            )
            if prompts_enabled():
                logger.debug("Utterance text", extra={"fields": {"text": text}})
            await self._queue.put((text, language))

    async def utterances(self):
        """Yield ``(text, language)`` once each utterance is complete."""
        while True:
            yield await self._queue.get()

    def close(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None