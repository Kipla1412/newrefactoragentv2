"""Registration of the MLflow DeepEval scorers into a metric registry.

Kept apart from the adapter itself so that "which metrics exist" and "how one
metric is measured" can change independently.

Factories are used because a scorer is expensive to build and may need
credentials: nothing is constructed until a metric is actually measured. If
MLflow's integration is not importable, registration is skipped and the names
stay unregistered — the engine then reports them as ``skipped`` rather than
failing a run.
"""

from __future__ import annotations

from typing import Any, Mapping

from evaluation.adapters.mlflow_scorer.metric import MLflowDeepEvalMetric
from evaluation.adapters.mlflow_scorer.spec import (
    DEFAULT_MAX_REASON_CHARS,
    MLFLOW_SCORERS,
    ScorerSpec,
    mlflow_scorers_available,
)
from evaluation.core.registry import MetricRegistry
from utils.logging_config import get_logger

logger = get_logger(__name__)


def register(
    registry: MetricRegistry,
    *,
    specs: Mapping[str, ScorerSpec] | None = None,
    model_uri: str | None = None,
    thresholds: Mapping[str, float] | None = None,
    metric_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    max_reason_chars: int = DEFAULT_MAX_REASON_CHARS,
    allow_external_calls: bool = False,
) -> tuple[str, ...]:
    """Register the MLflow scorers as lazy metric factories.

    Args:
        registry: Registry to populate.
        specs: Metric specs to register; defaults to :data:`MLFLOW_SCORERS`.
        model_uri: Judge model URI; ``None`` uses MLflow's default.
        thresholds: Per-metric threshold overrides.
        metric_kwargs: Per-metric DeepEval metric kwargs.
        max_reason_chars: Truncation bound for stored rationales.
        allow_external_calls: Privacy gate. **False by default** — the scorers are
            registered but will not contact an external judge until an operator
            explicitly enables them.

    Returns:
        The names registered.
    """
    resolved = dict(specs if specs is not None else MLFLOW_SCORERS)

    if not mlflow_scorers_available():
        logger.warning(
            "MLflow DeepEval scorers are not importable; %d metric(s) left "
            "unregistered: %s",
            len(resolved),
            ", ".join(sorted(resolved)),
        )
        return ()

    thresholds = thresholds or {}
    metric_kwargs = metric_kwargs or {}

    for name, spec in resolved.items():
        registry.register_factory(
            name,
            _make_factory(
                name,
                spec,
                model_uri=model_uri,
                threshold=thresholds.get(name),
                metric_kwargs=metric_kwargs.get(name),
                max_reason_chars=max_reason_chars,
                allow_external_calls=allow_external_calls,
            ),
        )

    logger.debug(
        "Registered %d MLflow DeepEval scorer(s) (external calls allowed: %s)",
        len(resolved),
        allow_external_calls,
    )
    return tuple(sorted(resolved))


def _make_factory(
    name: str,
    spec: ScorerSpec,
    *,
    model_uri: str | None,
    threshold: float | None,
    metric_kwargs: Mapping[str, Any] | None,
    max_reason_chars: int,
    allow_external_calls: bool,
):
    """Build the zero-argument factory the registry stores."""

    def factory() -> MLflowDeepEvalMetric:
        return MLflowDeepEvalMetric(
            name,
            spec,
            model_uri=model_uri,
            threshold=threshold,
            metric_kwargs=metric_kwargs,
            max_reason_chars=max_reason_chars,
            external_calls_allowed=allow_external_calls,
        )

    factory.__name__ = f"build_{name}"
    return factory
