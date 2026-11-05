"""Standardized evaluation value objects.

These are deliberately engine-agnostic: no DeepEval, no MLflow, no FastAPI. Both
evaluation modes (real-time and offline) produce the same
:class:`EvaluationContext` and consume the same :class:`EvaluationResult`, which
is what lets one engine serve both.

``EvaluationContext`` exposes ``answer`` / ``input`` / ``context`` / ``ended`` /
``error`` so the deterministic rules in :mod:`evaluation.agents.intake.rules` —
written against the intake adapter's shape — can be reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


class EvaluationMode(str, Enum):
    """Where an evaluation event came from.

    Subclasses ``str`` so the value round-trips through JSON, MLflow tags and
    YAML config unchanged, matching ``AgentType`` in :mod:`agent.events`.
    """

    REALTIME = "realtime"
    OFFLINE = "offline"


class MetricStatus(str, Enum):
    """Outcome of one metric measurement."""

    OK = "ok"
    SKIPPED = "skipped"
    ERROR = "error"


def _value(item: Any) -> Any:
    """Return an enum member's plain value, passing anything else through.

    ``str(SomeStrEnum.MEMBER)`` renders ``"SomeStrEnum.MEMBER"``, so
    serialization must not rely on ``str()``.
    """
    return item.value if isinstance(item, Enum) else item


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class EvaluationContext:
    """Everything one metric needs about a single execution.

    Attributes:
        agent_name: Logical agent that produced the response.
        input: The user input for this execution.
        output: The final response text.
        context: Conversation lines before the input, e.g.
            ``"Patient: I have a headache."``.
        ended: Whether the agent signalled end-of-conversation.
        error: Agent error text, when the execution failed.
        execution_id: Unique id for this execution (real-time) or case id.
        session_id: Session the execution belonged to.
        user_id: Caller identity, when known.
        model: Model the agent ran on, when known.
        duration_ms: Agent execution duration.
        retrieval_context: Retrieved passages, when the agent used any.
        tool_calls: Tool names the agent invoked.
        metadata: Extra, non-content context (tags, dataset version, ...).
        mode: :attr:`EvaluationMode.REALTIME` or :attr:`EvaluationMode.OFFLINE`.
    """

    agent_name: str
    input: str = ""
    output: str = ""
    context: tuple[str, ...] = ()
    ended: bool = False
    error: str | None = None
    execution_id: str = ""
    session_id: str | None = None
    user_id: str | None = None
    model: str | None = None
    duration_ms: float = 0.0
    retrieval_context: tuple[str, ...] = ()
    tool_calls: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    mode: EvaluationMode = EvaluationMode.OFFLINE

    @property
    def answer(self) -> str:
        """Alias for :attr:`output`, matching the deterministic checks."""
        return self.output


@dataclass(frozen=True)
class MetricResult:
    """One metric's verdict for one execution.

    Attributes:
        metric_name: Registered metric name.
        agent_name: Agent evaluated.
        execution_id: Execution (real-time) or case id (offline).
        session_id: Session the execution belonged to.
        score: Numeric score, 0.0-1.0 for deterministic metrics.
        threshold: Score needed to pass.
        passed: Whether the score met the threshold.
        reason: Short human-readable explanation.
        status: :class:`MetricStatus` — ``ok``, ``skipped`` or ``error``.
        evaluation_mode: ``realtime`` or ``offline``.
        timestamp: When the metric was measured.
    """

    metric_name: str
    agent_name: str
    execution_id: str
    session_id: str | None
    score: float
    threshold: float
    passed: bool
    reason: str
    status: MetricStatus
    evaluation_mode: EvaluationMode
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "metric_name": self.metric_name,
            "agent_name": self.agent_name,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "score": self.score,
            "threshold": self.threshold,
            "passed": self.passed,
            "reason": self.reason,
            "status": _value(self.status),
            "evaluation_mode": _value(self.evaluation_mode),
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """All metric results for one execution.

    Attributes:
        context: The execution that was evaluated.
        metrics: Per-metric results.
        duration_ms: Time spent evaluating.
    """

    context: EvaluationContext
    metrics: tuple[MetricResult, ...] = ()
    duration_ms: float = 0.0

    @property
    def passed(self) -> int:
        """Number of metrics that passed."""
        return sum(1 for result in self.metrics if result.passed)

    @property
    def total(self) -> int:
        """Number of metrics that ran."""
        return sum(1 for r in self.metrics if r.status != MetricStatus.SKIPPED)

    @property
    def failed(self) -> tuple[MetricResult, ...]:
        """Metrics that ran and did not pass."""
        return tuple(
            result
            for result in self.metrics
            if result.status == MetricStatus.OK and not result.passed
        )

    @property
    def errored(self) -> tuple[MetricResult, ...]:
        """Metrics that raised."""
        return tuple(r for r in self.metrics if r.status == MetricStatus.ERROR)

    @property
    def ok(self) -> bool:
        """Whether every executed metric passed and none errored."""
        return not self.failed and not self.errored

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "agent_name": self.context.agent_name,
            "execution_id": self.context.execution_id,
            "session_id": self.context.session_id,
            "evaluation_mode": _value(self.context.mode),
            "duration_ms": round(self.duration_ms, 1),
            "passed": self.passed,
            "total": self.total,
            "metrics": [result.to_dict() for result in self.metrics],
        }
