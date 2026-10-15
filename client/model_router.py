"""Task-to-model routing.

The router maps a clinical request onto a *logical* model name (``fast``,
``clinical``, ``retrieval``). It is deliberately independent of LiteLLM:
it never imports or references a provider — it only produces logical
names that the ModelRegistry then resolves to LiteLLM model strings.

    request → ModelRouter → logical model → ModelRegistry → LLMClient
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable


class TaskType(str, Enum):
    """Coarse task categories the router understands."""

    # Short conversational turns: greetings, clarifications, intake follow-ups.
    SIMPLE = "simple"
    # Clinical reasoning and documentation: assessment, diagnosis, SOAP,
    # treatment planning — routed to the strongest model.
    CLINICAL = "clinical"
    # Record / knowledge lookups: patient records, wiki, knowledge graph,
    # document retrieval — tool-driven, so a cheap model is enough.
    RETRIEVAL = "retrieval"


class ModelRouter:
    """Route a clinical request to a logical model name.

    Routing is a pure function of the request: no network calls, no
    provider knowledge. The mapping from task type to logical model is
    configurable so applications can point ``fast``/``clinical``/
    ``retrieval`` at different models without changing this class.

    Attributes:
        routes: Mapping of TaskType value to logical model name.
    """

    DEFAULT_ROUTES: dict[str, str] = {
        TaskType.SIMPLE.value: "fast",
        TaskType.CLINICAL.value: "reasoning",
        TaskType.RETRIEVAL.value: "fast",
    }

    # Substrings that hint at clinical reasoning / documentation work.
    # Checked first so ambiguous requests bias toward the stronger model.
    _CLINICAL_HINTS: tuple[str, ...] = (
        "assess", "assessment", "diagnos", "differential", "treatment",
        "prescribe", "prescription", "dosage", "dose", "risk", "red flag",
        "triage", "prognosis", "clinical", "soap", "recommend", "plan of care",
        "contraindication", "interpret", "acuity", "severity", "comorbid",
        "symptom", "vital sign", "refer",
    )

    # Substrings that hint at a record / knowledge lookup.
    _RETRIEVAL_HINTS: tuple[str, ...] = (
        "look up", "lookup", "find", "search", "retrieve", "pull up",
        "show me", "patient record", "records", "history of", "chart",
        "wiki", "knowledge graph", "ehr", "who treats", "which doctor",
        "allergies", "medication", "diseases", "taking", "lab result", "report",
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

        Clinical hints take precedence over retrieval hints, which take
        precedence over the simple default. The ordering is deliberate: a
        clinical request that also mentions a record is still routed to
        the stronger model.

        Args:
            text: The user request text.

        Returns:
            The classified TaskType.
        """
        if not text:
            return TaskType.SIMPLE

        lowered = text.lower()
        if any(hint in lowered for hint in self._CLINICAL_HINTS):
            return TaskType.CLINICAL
        if any(hint in lowered for hint in self._RETRIEVAL_HINTS):
            return TaskType.RETRIEVAL
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