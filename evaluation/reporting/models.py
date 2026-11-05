"""Reporting value objects for an offline run.

The shape is deliberately unchanged from the previous runner — what changed is how
it is produced. Each case now carries the engine's
:class:`~evaluation.core.models.EvaluationResult`, so the report is a view over
standardized results rather than a second, parallel scoring path.

The reporting layer depends only on core models, so it can be reused by a CI job,
a notebook or the CLI without dragging in an adapter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from evaluation.core.models import (
    EvaluationContext,
    EvaluationMode,
    EvaluationResult,
    MetricResult,
    MetricStatus,
)

#: Default file name for a saved report.
DEFAULT_REPORT_PATH = "evaluation_report.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class CaseResult:
    """Scored result for one dataset case.

    Attributes:
        case_id: Dataset case identifier.
        scenario: Human label from the dataset.
        answer: The agent's reply to the evaluated turn.
        greeting: One-time greeting, when a fresh session produced one.
        ok: Whether the agent ran without error.
        error: Error text, when it did not.
        history_turns: Conversation turns replayed before the evaluated turn.
        duration_ms: Case wall-clock time.
        checks: Deterministic metric outcomes.
        skipped_checks: Metrics requested but not run (with the reason).
        checks_passed: Metrics that ran and passed.
        checks_total: Metrics that ran.
        check_score: Passed / total.
        llm_scores: LLM-judged metric scores, when any ran.
        llm_reasons: LLM-judged metric rationales.
        tool_calls: Tool names the agent invoked.
        event_count: Total ``AgentEvent`` objects produced.
    """

    case_id: str
    scenario: str
    answer: str
    greeting: str | None
    ok: bool
    error: str | None
    history_turns: int
    duration_ms: float
    checks: tuple[dict[str, Any], ...]
    skipped_checks: tuple[str, ...]
    checks_passed: int
    checks_total: int
    check_score: float
    llm_scores: dict[str, float] = field(default_factory=dict)
    llm_reasons: dict[str, str] = field(default_factory=dict)
    tool_calls: tuple[str, ...] = ()
    event_count: int = 0
    execution_id: str = ""

    @property
    def fully_passed(self) -> bool:
        """Whether the case ran cleanly and every executed metric passed."""
        return self.ok and self.check_score >= 1.0

    @classmethod
    def from_evaluation(
        cls,
        case_id: str,
        scenario: str,
        context: EvaluationContext,
        result: EvaluationResult,
        *,
        llm_metric_names: Iterable[str] = (),
    ) -> "CaseResult":
        """Build a report row from the engine's standardized result.

        Args:
            case_id: Dataset case identifier.
            scenario: Human label from the dataset.
            context: The context the agent produced for this case.
            result: The engine's verdict.
            llm_metric_names: Names backed by an LLM judge, reported separately
                so a semantic score is never mistaken for a rule.
        """
        llm_names = set(llm_metric_names)

        checks = tuple(
            {
                "name": metric.metric_name,
                "passed": metric.passed,
                "detail": metric.reason,
                "status": metric.status,
            }
            for metric in result.metrics
            if metric.status == MetricStatus.OK
        )
        skipped = tuple(
            f"{metric.metric_name}: {metric.reason}"
            for metric in result.metrics
            if metric.status == MetricStatus.SKIPPED
        )
        errored = tuple(
            f"{metric.metric_name}: {metric.reason}"
            for metric in result.metrics
            if metric.status == MetricStatus.ERROR
        )

        llm_scores: dict[str, float] = {}
        llm_reasons: dict[str, str] = {}
        for metric in result.metrics:
            if metric.metric_name in llm_names and metric.status != MetricStatus.SKIPPED:
                llm_scores[metric.metric_name] = round(metric.score, 4)
                llm_reasons[metric.metric_name] = metric.reason

        return cls(
            case_id=case_id,
            scenario=scenario,
            answer=_string(context.metadata.get("answer")) or context.output,
            greeting=_optional_string(context.metadata.get("greeting")),
            ok=context.error is None,
            error=context.error,
            history_turns=int(context.metadata.get("history_turns", 0) or 0),
            duration_ms=context.duration_ms,
            checks=checks + tuple({"name": e.split(":", 1)[0], "passed": False,
                                   "detail": e, "status": MetricStatus.ERROR} for e in errored),
            skipped_checks=skipped,
            checks_passed=result.passed,
            checks_total=result.total,
            check_score=round(result.passed / result.total, 4) if result.total else 1.0,
            llm_scores=llm_scores,
            llm_reasons=llm_reasons,
            tool_calls=tuple(context.tool_calls),
            event_count=int(context.metadata.get("event_count", 0) or 0),
            execution_id=context.execution_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the result."""
        return {
            "case_id": self.case_id,
            "scenario": self.scenario,
            "answer": self.answer,
            "greeting": self.greeting,
            "ok": self.ok,
            "error": self.error,
            "history_turns": self.history_turns,
            "duration_ms": self.duration_ms,
            "event_count": self.event_count,
            "tool_calls": list(self.tool_calls),
            "execution_id": self.execution_id,
            "checks": list(self.checks),
            "skipped_checks": list(self.skipped_checks),
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "check_score": self.check_score,
            "llm_scores": self.llm_scores,
            "llm_reasons": self.llm_reasons,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """A complete offline evaluation run.

    Attributes:
        dataset_name: Dataset identifier.
        agent: Agent the dataset targets.
        dataset_version: Version of the dataset used.
        started_at: UTC timestamp when the run began.
        duration_ms: Total run time.
        cases: Per-case results.
        llm_metrics: Status of the optional LLM-judged metrics.
    """

    dataset_name: str
    agent: str
    started_at: str
    duration_ms: float
    cases: tuple[CaseResult, ...]
    llm_metrics: dict[str, Any]
    dataset_version: str = "1.0"
    mode: EvaluationMode = EvaluationMode.OFFLINE

    @classmethod
    def start_run(cls) -> str:
        """Timestamp for a new run."""
        return _utc_now()

    @property
    def summary(self) -> dict[str, Any]:
        """Aggregate scores for the whole run."""
        checks_passed = sum(case.checks_passed for case in self.cases)
        checks_total = sum(case.checks_total for case in self.cases)
        return {
            "cases": len(self.cases),
            "cases_passed": sum(1 for case in self.cases if case.fully_passed),
            "cases_failed": sum(1 for case in self.cases if not case.fully_passed),
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "check_score": round(checks_passed / checks_total, 4) if checks_total else 0.0,
            "errors": sum(1 for case in self.cases if not case.ok),
            "skipped_checks": sorted({
                name for case in self.cases for name in case.skipped_checks
            }),
            "duration_ms": round(self.duration_ms, 1),
        }

    @property
    def failures(self) -> tuple[CaseResult, ...]:
        """Cases that did not pass cleanly."""
        return tuple(case for case in self.cases if not case.fully_passed)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the report."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "agent": self.agent,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 1),
            "summary": self.summary,
            "llm_metrics": self.llm_metrics,
            "cases": [case.to_dict() for case in self.cases],
        }


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
