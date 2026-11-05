"""Agent metric sets — the plugin layer (Open/Closed).

Each agent owns a module that registers its own metrics. Adding an agent means
adding that module and one entry in :data:`AGENT_METRIC_SETS`; the evaluation
engine, the registry and the runner are untouched.

Vendor-backed metrics (MLflow's DeepEval scorers) are registered here too, but
they are *opt-in*: ``allow_external_calls`` defaults to False because this
framework handles patient data.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from evaluation.agents import intake
from evaluation.adapters import mlflow_scorer
from evaluation.core.registry import MetricRegistry
from utils.logging_config import get_logger

logger = get_logger(__name__)

#: Agent name -> registrar, invoked once per registry.
#:
#: Keyed by the same agent names used in ``agents.yaml`` and in
#: ``EvaluationContext.agent_name``, which is what links configuration to metrics.
AGENT_METRIC_SETS: dict[str, Callable[[MetricRegistry], None]] = {
    "intake": intake.register,
}


def register_all(
    registry: MetricRegistry,
    *,
    allow_external_calls: bool = False,
    model_uri: str | None = None,
    thresholds: Mapping[str, float] | None = None,
    metric_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Register every built-in metric set into ``registry``.

    Args:
        registry: Registry to populate.
        allow_external_calls: Whether LLM-based metrics may contact an external
            judge. **False by default.**
        model_uri: Judge model URI for the scorers; ``None`` uses MLflow's default.
        thresholds: Per-metric threshold overrides for the scorers.
        metric_kwargs: Per-metric DeepEval metric kwargs for the scorers.
    """
    for agent_name, registrar in AGENT_METRIC_SETS.items():
        registrar(registry)
        logger.debug("Registered metric set for agent '%s'", agent_name)

    mlflow_scorer.register(
        registry,
        model_uri=model_uri,
        thresholds=thresholds,
        metric_kwargs=metric_kwargs,
        allow_external_calls=allow_external_calls,
    )

    logger.debug("Registered built-in metric sets")
