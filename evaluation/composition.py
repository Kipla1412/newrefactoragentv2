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


def apply_framework_credentials(config: Any = None) -> bool:
    """Point MLflow's judge at the framework's own ``API_KEY``/``BASE_URL``.

    MLflow's scorers build their own LLM and read ``OPENAI_API_KEY`` (or another
    provider's key), while this framework authenticates with ``API_KEY`` /
    ``BASE_URL``. Without a bridge the two use different credentials, which is
    how a working agent can sit in the same deployment as a judge that fails with
    ``insufficient_quota``.

    Opt-in via ``evaluation.llm.use_framework_credentials``. When enabled the
    framework's values **win** — that is the point of the flag: one credential
    for the whole deployment. A provider-native key is only kept when the flag is
    off.

    Returns:
        True when at least one variable was set.
    """
    llm = getattr(getattr(config, "evaluation", None), "llm", None)
    if llm is None or not getattr(llm, "use_framework_credentials", False):
        return False

    import os

    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    changed = False
    replaced = False

    if api_key:
        if os.environ.get("OPENAI_API_KEY") != api_key:
            replaced = bool(os.environ.get("OPENAI_API_KEY"))
            os.environ["OPENAI_API_KEY"] = api_key
            changed = True

    if base_url:
        # LiteLLM and the OpenAI SDK read different names for the same thing;
        # set both so either route reaches the configured endpoint.
        for variable in ("OPENAI_API_BASE", "OPENAI_BASE_URL"):
            if os.environ.get(variable) != base_url:
                os.environ[variable] = base_url
                changed = True

    if changed:
        if replaced:
            logger.warning(
                "Evaluation judge credentials replaced by the framework API_KEY "
                "(evaluation.llm.use_framework_credentials is on)"
            )
        else:
            logger.info("Evaluation judge now uses the framework credentials")
    return changed

    if changed:
        logger.info(
            "Evaluation judge now uses the framework credentials",
            extra={"fields": {"base_url_set": bool(base_url)}},
        )
    return changed


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

    # Before any scorer exists: a scorer built later resolves its credentials
    # when constructed, so the bridge has to be in place first.
    if allow_external_calls:
        apply_framework_credentials(config)

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

    if engine is None:
        # Real-time uses its own selection: deterministic metrics only, unless
        # `evaluation.realtime.allow_llm_metrics` opts into per-turn semantic
        # scoring. See MetricSelection.from_config.
        engine = build_engine(
            config, selection=MetricSelection.from_config(config, realtime=True)
        )

    return EvaluationObserver(
        engine,
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


#: The process-wide real-time observer. One per process, like the MLflow tracker
#: (`get_mlflow_tracker`): an engine rebuilds the registry and every metric, which
#: is far too expensive to do per agent or per request.
_observer: Any | None = None
_observer_built = False


def get_observer(config: Any = None) -> Any | None:
    """Return the process-wide real-time observer, building it once.

    Returns:
        The observer, or None when real-time evaluation is disabled (the
        default) or cannot be built. Callers treat None as "no evaluation".
    """
    global _observer, _observer_built

    if not _realtime_enabled(config):
        return None

    if not _observer_built:
        _observer_built = True
        try:
            _observer = build_observer(config)
        except Exception as exc:  # noqa: BLE001 - evaluation is optional
            logger.warning("Real-time evaluation is unavailable: %s", exc)
            _observer = None

    return _observer


def reset_observer() -> None:
    """Drop the cached observer (used by tests and on config reload)."""
    global _observer, _observer_built
    _observer = None
    _observer_built = False


def attach_observer(agent: Any, config: Any = None) -> bool:
    """Attach the real-time observer to an agent, when one is available.

    Called wherever agents are built, so every agent gets evaluation without each
    call site knowing about it. Never raises: an agent that cannot be evaluated
    must still be usable.

    Args:
        agent: The agent instance to attach to.
        config: Runtime configuration.

    Returns:
        True when an observer was attached.
    """
    try:
        observer = get_observer(config)
        if observer is None:
            return False
        agent.evaluation_observer = observer
        return True
    except Exception as exc:  # noqa: BLE001 - evaluation is optional
        logger.debug("Real-time evaluation not attached: %s", exc)
        return False


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
