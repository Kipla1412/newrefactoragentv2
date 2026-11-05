"""Adapter from an MLflow DeepEval scorer to the framework's Metric strategy.

Everything MLflow-shaped about a single measurement lives here: how the scorer is
built, how it is called, and how its ``Feedback`` becomes a
:class:`~evaluation.core.models.MetricResult`.

Two behaviours are deliberate:

1. **Laziness.** Building a scorer resolves credentials, so it happens on first
   measurement, never at import time.
2. **Honest skipping.** A metric whose declared requirement is unmet reports
   ``skipped`` with the reason, instead of surfacing an opaque judge error.

Only score, threshold, rationale and error are ever kept — never the raw input or
output — which is what keeps patient content out of stored results.
"""

from __future__ import annotations

from typing import Any, Mapping

from evaluation.adapters.mlflow_scorer.spec import (
    DEFAULT_MAX_REASON_CHARS,
    DEFAULT_THRESHOLD,
    ScorerSpec,
)
from evaluation.core.models import (
    EvaluationContext,
    MetricResult,
    MetricStatus,
    utc_now,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


def build_scorer(
    scorer_name: str,
    *,
    model_uri: str | None = None,
    threshold: float | None = None,
    metric_kwargs: Mapping[str, Any] | None = None,
):
    """Construct an MLflow DeepEval scorer.

    Args:
        scorer_name: MLflow metric name, e.g. ``"AnswerRelevancy"``.
        model_uri: Judge model URI. ``None`` defers to MLflow's default
            (``MLFLOW_GENAI_JUDGE_DEFAULT_MODEL``, else ``openai:/gpt-4.1-mini``).
        threshold: Passing score. ``None`` uses the metric's own default.
        metric_kwargs: Extra arguments forwarded to the DeepEval metric class.

    Returns:
        A callable scorer.

    Raises:
        Exception: Whatever import or construction raises; callers degrade
            gracefully.
    """
    from mlflow.genai.scorers.deepeval import get_scorer

    kwargs: dict[str, Any] = dict(metric_kwargs or {})
    if threshold is not None:
        kwargs["threshold"] = threshold
    return get_scorer(scorer_name, model=model_uri, **kwargs)


class MLflowDeepEvalMetric:
    """Adapter turning an MLflow DeepEval scorer into a framework metric.

    Args:
        name: Registry name, e.g. ``"answer_relevancy"``.
        spec: What the scorer needs (see :class:`ScorerSpec`).
        scorer: Pre-built scorer. Tests inject a stub; production passes ``None``
            so the scorer is built on first use.
        model_uri: Judge model URI passed to MLflow.
        threshold: Passing threshold, when the caller wants to override it.
        metric_kwargs: Extra arguments for the DeepEval metric class.
        max_reason_chars: Rationales are truncated to bound stored text.
        external_calls_allowed: Privacy gate. When False the metric reports
            ``skipped`` without contacting the judge, so patient content cannot
            leave the process by misconfiguration alone.
    """

    def __init__(
        self,
        name: str,
        spec: ScorerSpec,
        *,
        scorer: Any | None = None,
        model_uri: str | None = None,
        threshold: float | None = None,
        metric_kwargs: Mapping[str, Any] | None = None,
        max_reason_chars: int = DEFAULT_MAX_REASON_CHARS,
        external_calls_allowed: bool = True,
    ) -> None:
        self.name = name
        self.canonical_name = name
        self.spec = spec
        self.threshold = threshold
        self.max_reason_chars = max_reason_chars
        self.external_calls_allowed = external_calls_allowed
        self._scorer = scorer
        self._model_uri = model_uri
        self._metric_kwargs = dict(metric_kwargs or {})

    @property
    def scorer(self):
        """The scorer, built on first access."""
        if self._scorer is None:
            self._scorer = build_scorer(
                self.spec.scorer_name,
                model_uri=self._model_uri,
                threshold=self.threshold,
                metric_kwargs=self._metric_kwargs,
            )
        return self._scorer

    @property
    def mode(self) -> str:
        """Always semantic: these metrics are judged by an LLM."""
        return "llm"

    def measure(self, context: EvaluationContext) -> MetricResult:
        """Judge one execution.

        Returns ``skipped`` when a declared requirement is unmet, ``error`` when
        the judge fails, and never raises.
        """
        missing = self._missing_requirements(context)
        if missing:
            return self._skipped(context, missing)

        try:
            scorer = self.scorer
        except Exception as exc:  # noqa: BLE001 - construction can need credentials
            logger.warning("Could not build scorer '%s': %s", self.name, exc)
            return self._result(
                context,
                score=0.0,
                threshold=_threshold_of(self.threshold),
                passed=False,
                reason=f"scorer unavailable: {type(exc).__name__}: {exc}",
                status=MetricStatus.ERROR,
            )

        try:
            feedback = self._call(scorer, context)
        except Exception as exc:  # noqa: BLE001 - a judge failure is not fatal
            logger.warning("Scorer '%s' raised: %s", self.name, exc)
            return self._result(
                context,
                score=0.0,
                threshold=_threshold_of(self.threshold),
                passed=False,
                reason=f"scorer raised: {type(exc).__name__}: {exc}",
                status=MetricStatus.ERROR,
            )

        return feedback_to_metric_result(
            feedback,
            name=self.name,
            context=context,
            threshold=self.threshold,
            max_reason_chars=self.max_reason_chars,
        )

    def __repr__(self) -> str:
        return (
            f"MLflowDeepEvalMetric(name={self.name!r}, "
            f"scorer={self.spec.scorer_name!r})"
        )

    # Internals

    def _call(self, scorer: Any, context: EvaluationContext):
        """Invoke the scorer with the inputs this framework can supply."""
        expectations = self._expectations(context)
        kwargs: dict[str, Any] = {
            "inputs": context.input,
            "outputs": context.output,
        }
        if expectations:
            kwargs["expectations"] = expectations
        return scorer(**kwargs)

    @staticmethod
    def _expectations(context: EvaluationContext) -> dict[str, Any]:
        """Build MLflow's ``expectations`` dict from the context metadata."""
        expectations: dict[str, Any] = {}
        expected_output = context.metadata.get("expected_output")
        if expected_output:
            expectations["expected_output"] = expected_output
        return expectations

    def _missing_requirements(self, context: EvaluationContext) -> list[str]:
        """Return human-readable reasons this metric cannot run."""
        if not self.external_calls_allowed:
            return [
                "external LLM evaluation is not enabled "
                "(set evaluation.llm.allow_external_calls)"
            ]

        missing: list[str] = []

        if self.spec.requires_trace:
            missing.append(
                "requires an MLflow trace (MLflow reads this input from trace "
                "spans, and this framework does not emit per-turn traces)"
            )

        if self.spec.requires_config:
            absent = [
                key
                for key in self.spec.requires_config
                if key not in self._metric_kwargs
            ]
            if absent:
                missing.append(f"requires configuration: {', '.join(absent)}")

        for key in self.spec.requires_context:
            if not _context_value(context, key):
                missing.append(f"requires {key}")

        return missing

    def _skipped(self, context: EvaluationContext, missing: list[str]) -> MetricResult:
        return self._result(
            context,
            score=0.0,
            threshold=_threshold_of(self.threshold),
            passed=False,
            reason="; ".join(missing),
            status=MetricStatus.SKIPPED,
        )

    def _result(
        self,
        context: EvaluationContext,
        *,
        score: float,
        threshold: float,
        passed: bool,
        reason: str,
        status: MetricStatus,
    ) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            agent_name=context.agent_name,
            execution_id=context.execution_id,
            session_id=context.session_id,
            score=score,
            threshold=threshold,
            passed=passed,
            reason=_bound(reason, self.max_reason_chars),
            status=status,
            evaluation_mode=context.mode,
            timestamp=utc_now(),
        )


def feedback_to_metric_result(
    feedback: Any,
    *,
    name: str,
    context: EvaluationContext,
    threshold: float | None = None,
    max_reason_chars: int = DEFAULT_MAX_REASON_CHARS,
) -> MetricResult:
    """Convert an MLflow ``Feedback`` into a :class:`MetricResult`.

    Pure and total, so the conversion is testable without a network call. Only
    score, threshold, rationale and error are kept — never the raw input or
    output.
    """
    error = getattr(feedback, "error", None)
    if error:
        return MetricResult(
            metric_name=name,
            agent_name=context.agent_name,
            execution_id=context.execution_id,
            session_id=context.session_id,
            score=0.0,
            threshold=_threshold_of(threshold),
            passed=False,
            reason=_bound(_error_text(error), max_reason_chars),
            status=MetricStatus.ERROR,
            evaluation_mode=context.mode,
            timestamp=utc_now(),
        )

    metadata = getattr(feedback, "metadata", None) or {}
    score = _as_float(metadata.get("score"), default=0.0)
    resolved_threshold = _as_float(
        metadata.get("threshold"), default=_threshold_of(threshold)
    )
    reason = getattr(feedback, "rationale", None) or ""

    return MetricResult(
        metric_name=name,
        agent_name=context.agent_name,
        execution_id=context.execution_id,
        session_id=context.session_id,
        score=score,
        threshold=resolved_threshold,
        passed=_is_yes(getattr(feedback, "value", None)),
        reason=_bound(reason, max_reason_chars),
        status=MetricStatus.OK,
        evaluation_mode=context.mode,
        timestamp=utc_now(),
    )


# Helpers


def _context_value(context: EvaluationContext, key: str) -> Any:
    """Look a requirement up on the context, then in its metadata."""
    if hasattr(context, key):
        return getattr(context, key)
    return context.metadata.get(key)


def _threshold_of(threshold: float | None) -> float:
    return DEFAULT_THRESHOLD if threshold is None else threshold


def _as_float(value: Any, *, default: float) -> float:
    """Parse a number MLflow may have stringified (metadata is dict[str, str])."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_yes(value: Any) -> bool:
    """Interpret a Feedback value as pass/fail.

    MLflow reports ``CategoricalRating.YES``/``NO``; a boolean or the strings
    ``"yes"``/``"true"``/``"pass"`` are accepted too.
    """
    if isinstance(value, bool):
        return value
    raw = getattr(value, "value", value)
    return str(raw).strip().lower() in {"yes", "true", "pass"}


def _error_text(error: Any) -> str:
    """Render a Feedback error without ever raising."""
    message = getattr(error, "message", None) or str(error)
    name = type(error).__name__
    return f"judge failed: {name}: {message}"


def _bound(text: str, limit: int) -> str:
    """Truncate stored text to bound what reaches MLflow and the report."""
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
