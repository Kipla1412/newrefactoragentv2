"""The shared evaluation engine (Facade over registry + selection).

One engine serves real-time and offline evaluation; the only difference between
the modes is where the :class:`~evaluation.core.models.EvaluationContext` came
from.

It depends on abstractions only — a :class:`~evaluation.core.registry.MetricRegistry`
and a :class:`~evaluation.core.selection.MetricSelection` — so it knows nothing
about DeepEval, MLflow, FastAPI, configuration files or any concrete metric.

Failures are isolated per metric: a broken or unavailable metric is reported as
``error`` and never prevents the others from running, and never propagates to the
caller.
"""

from __future__ import annotations

import time
from typing import Iterable

from evaluation.core.metric import Metric
from evaluation.core.models import (
    EvaluationContext,
    EvaluationResult,
    MetricResult,
    MetricStatus,
    utc_now,
)
from evaluation.core.registry import MetricRegistry
from evaluation.core.selection import MetricSelection
from utils.logging_config import get_logger

logger = get_logger(__name__)


class EvaluationEngine:
    """Runs the selected metrics for one execution.

    Args:
        registry: Resolves metric names to strategies.
        selection: Which names apply to which agent. Defaults to no metrics, so
            an unconfigured engine evaluates nothing rather than guessing.
    """

    def __init__(
        self,
        registry: MetricRegistry,
        selection: MetricSelection | None = None,
    ) -> None:
        self._registry = registry
        self._selection = selection or MetricSelection()

    @property
    def selection(self) -> MetricSelection:
        """The metric selection in use."""
        return self._selection

    @property
    def registry(self) -> MetricRegistry:
        """The registry in use."""
        return self._registry

    def metric_names(self, agent_name: str) -> tuple[str, ...]:
        """Metric names that would run for ``agent_name``."""
        return self._selection.for_agent(agent_name)

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        """Evaluate one execution.

        Never raises: metrics that fail are reported with status ``error``,
        names that do not resolve with status ``skipped``, and duplicates of an
        already-measured metric with status ``skipped``.
        """
        names = self.metric_names(context.agent_name)
        if not names:
            logger.debug(
                "No metrics configured for agent '%s'",
                context.agent_name,
                extra={"fields": {"agent_name": context.agent_name}},
            )
            return EvaluationResult(context=context)

        metrics, unknown = self._registry.resolve(names)
        metrics, duplicates = dedupe_metrics(metrics)
        started = time.perf_counter()

        results = [self._measure(metric, context) for metric in metrics]
        results.extend(self._skipped(name, context) for name in unknown)
        results.extend(
            self._skipped(name, context, reason=f"duplicate of '{canonical}'")
            for name, canonical in duplicates
        )

        elapsed_ms = (time.perf_counter() - started) * 1000
        result = EvaluationResult(
            context=context,
            metrics=tuple(results),
            duration_ms=elapsed_ms,
        )
        logger.debug(
            "Evaluated %s/%s metrics for agent '%s' in %.0fms (ok=%s)",
            result.passed,
            result.total,
            context.agent_name,
            elapsed_ms,
            result.ok,
            extra={
                "fields": {
                    "agent_name": context.agent_name,
                    "execution_id": context.execution_id,
                    "session_id": context.session_id,
                    "evaluation_mode": context.mode.value,
                    "passed": result.passed,
                    "total": result.total,
                    "ok": result.ok,
                    "duration_ms": round(elapsed_ms, 1),
                    "failed": [m.metric_name for m in result.failed],
                    "errored": [m.metric_name for m in result.errored],
                }
            },
        )
        return result

    def evaluate_many(
        self, contexts: Iterable[EvaluationContext]
    ) -> list[EvaluationResult]:
        """Evaluate several executions in order."""
        return [self.evaluate(context) for context in contexts]

    def _measure(self, metric: Metric, context: EvaluationContext) -> MetricResult:
        """Measure one metric, isolating any failure."""
        try:
            return metric.measure(context)
        except Exception as exc:  # noqa: BLE001 - isolation is the guarantee
            logger.exception(
                "Metric '%s' failed for agent '%s'",
                metric.name,
                context.agent_name,
                extra={
                    "fields": {
                        "metric_name": metric.name,
                        "agent_name": context.agent_name,
                        "execution_id": context.execution_id,
                    }
                },
            )
            return self._error(metric.name, context, exc)

    @staticmethod
    def _error(name: str, context: EvaluationContext, exc: Exception) -> MetricResult:
        return MetricResult(
            metric_name=name,
            agent_name=context.agent_name,
            execution_id=context.execution_id,
            session_id=context.session_id,
            score=0.0,
            threshold=1.0,
            passed=False,
            reason=f"metric failed: {type(exc).__name__}: {exc}",
            status=MetricStatus.ERROR,
            evaluation_mode=context.mode,
            timestamp=utc_now(),
        )

    @staticmethod
    def _skipped(
        name: str,
        context: EvaluationContext,
        reason: str | None = None,
    ) -> MetricResult:
        return MetricResult(
            metric_name=name,
            agent_name=context.agent_name,
            execution_id=context.execution_id,
            session_id=context.session_id,
            score=0.0,
            threshold=1.0,
            passed=False,
            reason=reason or f"metric '{name}' is not registered",
            status=MetricStatus.SKIPPED,
            evaluation_mode=context.mode,
            timestamp=utc_now(),
        )


def dedupe_metrics(
    metrics: list[Metric],
) -> tuple[list[Metric], list[tuple[str, str]]]:
    """Split metrics into those to measure and those that duplicate another.

    Two registry names can point at the same implementation — the intake rules
    are registered under both their own names and clinical aliases
    (``intake_completion`` is ``completes_intake``). Configuring both must not run
    the rule twice, or a single failing rule would be counted twice and the
    verdict would look worse than it is.

    Returns:
        The metrics to measure, and ``(name, canonical_name)`` for the skipped
        duplicates so the report can say *why* they did not run.
    """
    seen: set[str] = set()
    measured: list[Metric] = []
    duplicates: list[tuple[str, str]] = []

    for metric in metrics:
        canonical = getattr(metric, "canonical_name", None) or metric.name
        if canonical in seen:
            logger.debug(
                "Skipping metric '%s': duplicates canonical '%s'",
                metric.name,
                canonical,
                extra={
                    "fields": {
                        "metric_name": metric.name,
                        "canonical_name": canonical,
                    }
                },
            )
            duplicates.append((metric.name, canonical))
            continue
        seen.add(canonical)
        measured.append(metric)

    return measured, duplicates
