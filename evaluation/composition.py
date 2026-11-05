"""The composition root.

The only module that knows every layer at once: it decides *which* registry,
selection, adapters and runtime object to build. Everything else depends on
abstractions, which is what keeps the dependency graph pointing inward:

.. code-block:: text

    composition  ──►  runtime  ──►  ports ◄── adapters
                        │                        │
                        └──────► core ◄──────────┘
                                   ▲
                    agents ────────┘   (plugins, registered here only)

Nothing imports :mod:`evaluation.composition` except the entry points — the CLI
and, later, application startup.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from evaluation.adapters.mlflow_tracker import EvaluationTracker
from evaluation.agents import register_all
from evaluation.core.engine import EvaluationEngine
from evaluation.core.models import EvaluationResult
from evaluation.core.registry import MetricRegistry
from evaluation.core.selection import MetricSelection
from utils.logging_config import get_logger

logger = get_logger(__name__)


def build_registry(
    config: Any = None,
    *,
    allow_external_calls: bool | None = None,
    model_uri: str | None = None,
    thresholds: Mapping[str, float] | None = None,
    metric_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
) -> MetricRegistry:
    """Build a registry with every built-in metric set registered.

    Args:
        config: Optional runtime configuration; supplies the privacy gate and the
            judge model URI when they are not passed explicitly.
        allow_external_calls: Override the privacy gate. Defaults to the
            configuration value, which is **False** when unset.
        model_uri: Judge model URI for LLM-backed metrics.
        thresholds: Per-metric threshold overrides.
        metric_kwargs: Per-metric DeepEval metric kwargs.

    Returns:
        The populated registry.
    """
    if allow_external_calls is None:
        allow_external_calls = (
            MetricSelection.external_calls_allowed(config) if config else False
        )
    if model_uri is None and config is not None:
        model_uri = MetricSelection.judge_model_uri(config)

    registry = MetricRegistry()
    register_all(
        registry,
        allow_external_calls=allow_external_calls,
        model_uri=model_uri,
        thresholds=thresholds,
        metric_kwargs=metric_kwargs,
    )
    logger.debug("Built registry with %d metric(s)", len(registry))
    return registry


def build_engine(
    config: Any = None,
    *,
    registry: MetricRegistry | None = None,
    selection: MetricSelection | None = None,
) -> EvaluationEngine:
    """Build the shared evaluation engine.

    Args:
        config: Runtime configuration; drives both the registry and the selection.
        registry: Override the registry (tests inject one).
        selection: Override the selection (tests inject one).
    """
    resolved_registry = registry if registry is not None else build_registry(config)
    resolved_selection = (
        selection if selection is not None else MetricSelection.from_config(config)
    )
    return EvaluationEngine(resolved_registry, resolved_selection)


def build_tracker(config: Any, **kwargs: Any) -> EvaluationTracker:
    """Build the MLflow tracker, reusing the framework's MLflow setup."""
    return EvaluationTracker(config, **kwargs)


def build_observer(
    config: Any = None,
    *,
    engine: EvaluationEngine | None = None,
    tracker: Any | None = None,
    on_result: Callable[[EvaluationResult], None] | None = None,
    enabled: bool | None = None,
) -> "EvaluationObserver":  # noqa: F821 - imported below to keep this light
    """Build the real-time observer.

    Args:
        config: Runtime configuration.
        engine: Override the engine.
        tracker: Override the tracker; ``None`` builds the MLflow one from config.
        on_result: Optional callback per result.
        enabled: Override the configured on/off switch.
    """
    from evaluation.runtime.observer import EvaluationObserver

    if tracker is None and config is not None:
        tracker = build_tracker(config)
    if enabled is None:
        enabled = _realtime_enabled(config)

    return EvaluationObserver(
        engine if engine is not None else build_engine(config),
        tracker=tracker,
        on_result=on_result,
        max_concurrency=_realtime_int(config, "max_concurrency", 2),
        timeout=float(_realtime_float(config, "timeout", 60.0)),
        enabled=enabled,
    )


def build_offline_runner(config: Any, **overrides: Any) -> "OfflineRunner":  # noqa: F821
    """Build the offline runner with its default collaborators.

    Supplies the engine, which :class:`~evaluation.runtime.offline.OfflineRunner`
    requires so that the runtime never reaches back here.
    """
    from evaluation.runtime.offline import OfflineRunner

    overrides.setdefault("engine", build_engine(config))
    return OfflineRunner(config, **overrides)


def _realtime_section(config: Any) -> Any:
    section = getattr(config, "evaluation", None)
    return getattr(section, "realtime", None) if section is not None else None


def _realtime_enabled(config: Any) -> bool:
    if config is None:
        return False
    section = getattr(config, "evaluation", None)
    if section is None or not getattr(section, "enabled", True):
        return False
    realtime = _realtime_section(config)
    return bool(getattr(realtime, "enabled", False))


def _realtime_int(config: Any, name: str, default: int) -> int:
    value = getattr(_realtime_section(config), name, default)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _realtime_float(config: Any, name: str, default: float) -> float:
    value = getattr(_realtime_section(config), name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
