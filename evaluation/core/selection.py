"""Metric selection — which metric names apply to which agent (Specification).

Configuration is a *specification* over the registry: it answers "which metrics
does this agent run?" without knowing anything about the metrics themselves. The
engine consumes it, so changing how configuration is expressed never changes how
evaluation runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class MetricSelection:
    """Which metric names apply to which agent.

    Attributes:
        common: Metrics that run for every agent.
        per_agent: Extra metrics for a given agent name.
    """

    common: tuple[str, ...] = ()
    per_agent: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def for_agent(self, agent_name: str) -> tuple[str, ...]:
        """Return the ordered, de-duplicated metric names for ``agent_name``."""
        names: list[str] = list(self.common)
        names.extend(self.per_agent.get(agent_name, ()))
        seen: set[str] = set()
        ordered: list[str] = []
        for name in names:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return tuple(ordered)

    def agents(self) -> tuple[str, ...]:
        """Agent names that have agent-specific metrics."""
        return tuple(sorted(self.per_agent))

    def is_empty(self) -> bool:
        """Whether no metric would ever run."""
        return not self.common and not any(self.per_agent.values())

    @classmethod
    def from_config(cls, config: Any) -> "MetricSelection":
        """Build a selection from the framework's ``evaluation`` configuration.

        Reads the ``evaluation:`` block of ``agents.yaml`` (see
        :class:`config.config.EvaluationSection`). Two behaviours are deliberate:

        - ``evaluation.enabled: false`` yields an empty selection, so the engine
          evaluates nothing at all — the disabled mode is enforced here rather
          than by every caller remembering to check.
        - ``llm_metrics`` are always selected when listed, even if external calls
          are off. The metric then reports itself as ``skipped`` with the reason,
          so the report says *why* a semantic metric did not run instead of
          leaving a silent gap. The privacy gate itself lives on the metric.
        """
        section = getattr(config, "evaluation", None)
        if section is None or not getattr(section, "enabled", True):
            return cls()

        common = tuple(getattr(section, "common_metrics", ()) or ())

        per_agent: dict[str, tuple[str, ...]] = {}
        for agent_name, agent_config in (getattr(section, "agents", None) or {}).items():
            names: list[str] = list(getattr(agent_config, "metrics", ()) or ())
            names.extend(getattr(agent_config, "llm_metrics", ()) or ())
            per_agent[agent_name] = tuple(names)

        return cls(common=common, per_agent=per_agent)

    @classmethod
    def external_calls_allowed(cls, config: Any) -> bool:
        """Whether the configuration permits LLM-judged metrics to leave the process.

        Defaults to **False**: an absent or malformed ``evaluation.llm`` block must
        never silently enable sending patient data to an external provider.
        """
        section = getattr(config, "evaluation", None)
        llm = getattr(section, "llm", None) if section is not None else None
        return bool(getattr(llm, "allow_external_calls", False))

    @classmethod
    def judge_model_uri(cls, config: Any) -> str | None:
        """Configured judge model URI, or None to let MLflow choose."""
        section = getattr(config, "evaluation", None)
        llm = getattr(section, "llm", None) if section is not None else None
        uri = getattr(llm, "model_uri", None)
        return uri or None

    @classmethod
    def llm_thresholds(cls, config: Any) -> dict[str, float]:
        """Per-metric passing thresholds for LLM-judged metrics (may be empty)."""
        section = getattr(config, "evaluation", None)
        llm = getattr(section, "llm", None) if section is not None else None
        thresholds = getattr(llm, "thresholds", None) or {}
        return dict(thresholds)
