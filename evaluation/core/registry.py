"""The metric registry (Registry + Factory patterns).

Names are resolved to strategies at evaluation time, so configuration and
datasets reference *names* and never classes. Adding a metric means registering a
name — the engine is untouched.

The registry is a pure lookup: it has no knowledge of configuration, agents or
vendors. :func:`build_default_registry` is the one place that knows which metric
sets ship with the framework, and it is the composition root for them.
"""

from __future__ import annotations

from evaluation.core.metric import (
    CheckFn,
    DeterministicMetric,
    Metric,
    MetricFactory,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


class MetricRegistry:
    """Name-keyed registry of metric strategies.

    Registration is idempotent per name: re-registering replaces, which keeps
    test setups and project overrides straightforward.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}
        self._factories: dict[str, MetricFactory] = {}

    def register(self, metric: Metric) -> None:
        """Register a ready metric instance under ``metric.name``."""
        self._metrics[metric.name] = metric
        self._factories.pop(metric.name, None)
        logger.debug("Registered metric '%s'", metric.name)

    def register_factory(self, name: str, factory: MetricFactory) -> None:
        """Register a factory for a metric that must be built lazily."""
        self._factories[name] = factory
        self._metrics.pop(name, None)
        logger.debug("Registered metric factory '%s'", name)

    def register_from_checks(
        self,
        checks: dict[str, CheckFn],
        *,
        aliases: dict[str, str] | None = None,
    ) -> None:
        """Register every deterministic rule, plus named aliases over them.

        An alias shares the target rule's ``canonical_name``, so configuring both
        the alias and its target measures the rule once.
        """
        for name, check in checks.items():
            self.register(DeterministicMetric(name, check))
        for alias, target in (aliases or {}).items():
            check = checks.get(target)
            if check is None:
                logger.warning("Alias '%s' points at unknown check '%s'", alias, target)
                continue
            self.register(DeterministicMetric(alias, check, canonical_name=target))

    def has(self, name: str) -> bool:
        """Whether ``name`` resolves to a metric."""
        return name in self._metrics or name in self._factories

    def get(self, name: str) -> Metric | None:
        """Resolve one metric, building it if it was registered as a factory."""
        if name in self._factories:
            try:
                return self._factories[name]()
            except Exception:
                logger.exception("Metric factory for '%s' failed", name)
                return None
        return self._metrics.get(name)

    def resolve(self, names: tuple[str, ...]) -> tuple[list[Metric], list[str]]:
        """Resolve names to metrics.

        Returns:
            The metrics that resolved, and the names that did not. Unresolved
            names are reported rather than raising, so a dataset referencing a
            retired metric still evaluates its other metrics.
        """
        metrics: list[Metric] = []
        unknown: list[str] = []
        for name in names:
            metric = self.get(name)
            if metric is None:
                unknown.append(name)
            else:
                metrics.append(metric)
        return metrics, unknown

    def names(self) -> tuple[str, ...]:
        """Every registered metric name, sorted."""
        return tuple(sorted(set(self._metrics) | set(self._factories)))

    def __len__(self) -> int:
        return len(set(self._metrics) | set(self._factories))
