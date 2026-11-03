"""Route DeepEval's LLM judge through the framework's own LLM client.

DeepEval normally builds its own OpenAI client. Wrapping ``LLMClient`` instead
means the judge uses the same configured provider, credential strategy, retry and
fallback chain, and logging as every agent — and needs no second set of
credentials.

Import of ``deepeval`` is deferred to :func:`build_judge`, so the evaluation
runner works with the dependency absent.
"""

from __future__ import annotations

import asyncio
from typing import Any

from utils.logging_config import get_logger

logger = get_logger(__name__)


def deepeval_installed() -> bool:
    """Whether the optional ``deepeval`` dependency is importable."""
    try:
        import deepeval  # noqa: F401
    except ImportError:
        return False
    return True


def build_judge(client: Any, model: str | None = None):
    """Build a DeepEval judge backed by the framework's LLM client.

    The judge is pinned to a single model and **does not use the fallback
    chain**. A weaker fallback model answers evaluation prompts with
    unparseable text, which DeepEval reports as "invalid JSON"; failing loudly
    on the intended judge model is far more useful.

    Args:
        client: An ``LLMClient`` (``session.client``).
        model: Logical or raw model name to judge with. Defaults to
            ``clinical_model``, the model the agents run on.

    Returns:
        A ``DeepEvalBaseLLM`` subclass instance, or None when ``deepeval`` is
        not installed.
    """
    try:
        from deepeval.models import DeepEvalBaseLLM
    except ImportError:
        logger.info("deepeval is not installed; LLM-judged metrics are disabled")
        return None

    class LLMClientJudge(DeepEvalBaseLLM):
        """DeepEval model adapter over ``LLMClient``.

        ``generate`` is synchronous because DeepEval's metrics are; it is only
        called from the runner's scoring phase, which runs outside any event
        loop, so ``asyncio.run`` is safe there.
        """

        def __init__(self) -> None:
            self._client = client
            self._model_name = model or "clinical_model"
            super().__init__(model=self._model_name)

        def load_model(self) -> Any:
            """Return the wrapped client (required by DeepEval)."""
            return self._client

        def get_model_name(self) -> str:
            """Name DeepEval reports for this judge."""
            return self._model_name

        def generate(self, prompt: str) -> str:
            """Synchronously generate a completion."""
            return asyncio.run(self.a_generate(prompt))

        async def a_generate(self, prompt: str) -> str:
            """Generate a completion through the framework's LLM client."""
            chunks: list[str] = []

            async for event in self._client.chat_completion(
                [{"role": "user", "content": prompt}],
                model=self._model_name,
                stream=False,
                use_fallbacks=False,
            ):
                if event.text_delta and event.text_delta.content:
                    chunks.append(event.text_delta.content)

            return "".join(chunks).strip()

    return LLMClientJudge()
