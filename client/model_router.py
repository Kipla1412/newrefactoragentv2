"""Task-to-model routing.

The router maps a request onto a *logical* model name (``fast``,
``coding``, ``reasoning``). It is deliberately independent of LiteLLM:
it never imports or references a provider — it only produces logical
names that the ModelRegistry then resolves to LiteLLM model strings.

    request → ModelRouter → logical model → ModelRegistry → LLMClient
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable


class TaskType(str, Enum):
    """Coarse task categories the router understands."""

    SIMPLE = "simple"
    CODING = "coding"
    COMPLEX = "complex"


class ModelRouter:
    """Route a request to a logical model name.

    Routing is a pure function of the request: no network calls, no
    provider knowledge. The mapping from task type to logical model is
    configurable so applications can point ``fast``/``coding``/
    ``reasoning`` at different models without changing this class.

    Attributes:
        routes: Mapping of TaskType value to logical model name.
    """

    DEFAULT_ROUTES: dict[str, str] = {
        TaskType.SIMPLE.value: "fast",
        TaskType.CODING.value: "coding",
        TaskType.COMPLEX.value: "reasoning",
    }

    # Substrings that hint at a coding-heavy request.
    _CODING_HINTS: tuple[str, ...] = (
        "code", "function", "class ", "method", "refactor", "bug", "stack trace",
        "traceback", "compile", "syntax", "debug", "unit test", "pytest",
        "regex", "algorithm", "api ", "endpoint", "sql", "query", "json",
        "python", "javascript", "typescript", "java", "golang", "rust",
        "import ", "def ", "async ", "database", "dockerfile",
    )

    # Substrings that hint at a reasoning-heavy request.
    _COMPLEX_HINTS: tuple[str, ...] = (
        "analyze", "analyse", "analysis", "design", "architecture", "strategy",
        "trade-off", "tradeoff", "compare", "evaluate", "critique", "research",
        "why ", "step by step", "multi-step", "in depth", "plan ", "reasoning",
        "diagnos", "differential", "root cause", "optimi", "prove", "derive",
    )

    def __init__(self, routes: dict[str, str] | None = None) -> None:
        """Initialize the router.

        Args:
            routes: Optional override mapping of task type to logical
                model name. Defaults to :attr:`DEFAULT_ROUTES`.
        """
        self.routes: dict[str, str] = dict(routes or self.DEFAULT_ROUTES)

    def classify(self, text: str | None) -> TaskType:
        """Classify free text into a coarse task type.

        Coding hints take precedence over reasoning hints, which take
        precedence over the simple default.

        Args:
            text: The user request text.

        Returns:
            The classified TaskType.
        """
        if not text:
            return TaskType.SIMPLE

        lowered = text.lower()
        if any(hint in lowered for hint in self._CODING_HINTS):
            return TaskType.CODING
        if any(hint in lowered for hint in self._COMPLEX_HINTS):
            return TaskType.COMPLEX
        return TaskType.SIMPLE

    def route(
        self,
        task_type: str | TaskType | None = None,
        messages: Iterable[dict[str, Any]] | None = None,
    ) -> str | None:
        """Return the logical model name for a request.

        Args:
            task_type: Explicit task type. When provided it bypasses
                classification.
            messages: Chat messages used to infer the task type when
                ``task_type`` is not given. The most recent user message
                is used.

        Returns:
            The logical model name, or None if the task type has no route.
        """
        if task_type is not None:
            task = TaskType(task_type) if not isinstance(task_type, TaskType) else task_type
        else:
            task = self.classify(self._last_user_text(messages))

        return self.routes.get(task.value)

    @staticmethod
    def _last_user_text(messages: Iterable[dict[str, Any]] | None) -> str | None:
        """Return the most recent user message's text content."""
        if not messages:
            return None

        last: str | None = None
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                last = content
            elif isinstance(content, list):
                # OpenAI-style content parts.
                parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                ]
                last = " ".join(p for p in parts if p)
        return last