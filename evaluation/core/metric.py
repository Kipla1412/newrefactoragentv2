"""Metric strategies — the Open/Closed seam of the evaluation layer.

The engine knows only the :class:`Metric` protocol, so a new metric is a new
strategy rather than an engine change:

- a pure-Python rule  -> :class:`DeterministicMetric`
- a DeepEval judge    -> :class:`~evaluation.adapters.deepeval_judge` user
- an MLflow scorer    -> :class:`~evaluation.adapters.mlflow_scorer.MLflowDeepEvalMetric`

:data:`MetricFactory` exists for strategies that must be built lazily because
they need configuration or a client (Registry + Factory patterns).
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from evaluation.core.models import (
    EvaluationContext,
    MetricResult,
    MetricStatus,
    utc_now,
)

#: Builds a metric on demand. Used for metrics that need a client (e.g. a judge).
MetricFactory = Callable[[], "Metric"]

#: A deterministic rule: ``(context, context) -> (passed, reason)``.
#:
#: Two parameters so existing rule libraries written as ``check(case, observation)``
#: keep working unchanged — :class:`EvaluationContext` exposes both shapes.
CheckFn = Callable[[EvaluationContext, EvaluationContext], tuple[bool, str]]


@runtime_checkable
class Metric(Protocol):
    """A single evaluation metric (Strategy).

    Implementations decide *how* to score; the engine decides *which* metrics
    run. Implementations may also expose ``canonical_name`` — the identity of the
    underlying implementation. The engine de-duplicates on it, so two registry
    names backed by the same rule are measured only once. Metrics without it fall
    back to :attr:`name`.
    """

    name: str

    def measure(self, context: EvaluationContext) -> MetricResult:
        """Score one execution and return the verdict."""
        ...


class DeterministicMetric:
    """Adapter turning a plain-Python rule into a :class:`Metric`.

    Exceptions are deliberately not caught here: the engine isolates failures, so
    one broken metric cannot hide the others.
    """

    def __init__(
        self,
        name: str,
        check: CheckFn,
        *,
        description: str = "",
        threshold: float = 1.0,
        canonical_name: str | None = None,
    ) -> None:
        self.name = name
        #: Identity of the underlying rule; aliases point at their target so the
        #: engine never measures the same rule twice.
        self.canonical_name = canonical_name or name
        self.threshold = threshold
        self.description = description or (check.__doc__ or "").strip().split("\n")[0]
        self._check = check

    def measure(self, context: EvaluationContext) -> MetricResult:
        """Run the wrapped rule and convert it to a :class:`MetricResult`."""
        passed, reason = self._check(context, context)
        return MetricResult(
            metric_name=self.name,
            agent_name=context.agent_name,
            execution_id=context.execution_id,
            session_id=context.session_id,
            score=1.0 if passed else 0.0,
            threshold=self.threshold,
            passed=bool(passed),
            reason=reason,
            status=MetricStatus.OK,
            evaluation_mode=context.mode,
            timestamp=utc_now(),
        )

    def __repr__(self) -> str:
        return f"DeterministicMetric(name={self.name!r})"
