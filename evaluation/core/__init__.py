"""Evaluation core — abstractions with no vendor, agent or framework knowledge.

Nothing in this package imports an adapter, an agent metric set, or
configuration. Dependencies point inward only:

    adapters / agents / runtime  ->  core

See ``docs/evaluation.md`` for the layer map.
"""

from evaluation.core.dataset import DatasetError, EvalCase, EvalDataset
from evaluation.core.engine import EvaluationEngine, dedupe_metrics
from evaluation.core.metric import (
    CheckFn,
    DeterministicMetric,
    Metric,
    MetricFactory,
)
from evaluation.core.models import (
    EvaluationContext,
    EvaluationMode,
    EvaluationResult,
    MetricResult,
    MetricStatus,
    utc_now,
)
from evaluation.core.registry import MetricRegistry
from evaluation.core.selection import MetricSelection

__all__ = [
    "CheckFn",
    "DatasetError",
    "DeterministicMetric",
    "EvalCase",
    "EvalDataset",
    "EvaluationContext",
    "EvaluationEngine",
    "EvaluationMode",
    "EvaluationResult",
    "Metric",
    "MetricFactory",
    "MetricRegistry",
    "MetricResult",
    "MetricSelection",
    "MetricStatus",
    "dedupe_metrics",
    "utc_now",
]
