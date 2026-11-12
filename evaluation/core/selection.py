"""Metric selection — which metric names apply to which agent (Specification).

Configuration is a *specification* over the registry: it answers "which metrics
does this agent run?" without knowing anything about the metrics themselves. The
engine consumes it, so changing how configuration is expressed never changes how
evaluation runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


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

    def with_extra(self, agent_name: str, extra: Iterable[str]) -> "MetricSelection":
        """Return a new selection with additional names for one agent.

        Used by offline evaluation to honour a dataset case's own expectations:
        a case that describes an emergency asks for ``emergency_escalation``, a
        summary case for ``summary_verification``. Those rules would fail on an
        ordinary mid-intake turn, so they are not part of the agent's per-turn
        set — they are requested by the case that should exhibit them.

        Args:
            agent_name: Agent the extra metrics apply to.
            extra: Additional metric names.

        Returns:
            A new selection; this instance is unchanged.
        """
        merged: list[str] = list(self.per_agent.get(agent_name, ()))
        merged.extend(extra)

        per_agent = dict(self.per_agent)
        per_agent[agent_name] = tuple(merged)
        return MetricSelection(common=self.common, per_agent=per_agent)

    @classmethod
    def from_config(cls, config: Any, *, realtime: bool = False) -> "MetricSelection":
        """Build a selection from the framework's ``evaluation`` configuration.

        Reads the ``evaluation:`` block of ``agents.yaml`` (see
        :class:`config.config.EvaluationSection`). Three behaviours are deliberate:

        - ``evaluation.enabled: false`` yields an empty selection, so the engine
          evaluates nothing at all — the disabled mode is enforced here rather
          than by every caller remembering to check.
        - ``llm_metrics`` are selected for **offline** evaluation whenever listed.
          The metric reports itself as ``skipped`` with the reason when external
          calls are off, so the report says *why* it did not run instead of
          leaving a silent gap. The privacy gate itself lives on the metric.
        - For **real-time** evaluation they are excluded unless
          ``evaluation.realtime.allow_llm_metrics`` is true. A semantic metric
          calls an external judge on every turn: real cost, real latency, and
          patient text leaving the process live. Offline runs are reviewed by a
          human before anything depends on them; live traffic is not.

        Args:
            config: Runtime configuration.
            realtime: Build the selection for live, per-turn evaluation.
        """
        section = getattr(config, "evaluation", None)
        if section is None or not getattr(section, "enabled", True):
            return cls()

        common = tuple(getattr(section, "common_metrics", ()) or ())

        # Semantic metrics are opt-in for live traffic only.
        include_llm = not realtime or bool(
            getattr(getattr(section, "realtime", None), "allow_llm_metrics", False)
        )

        per_agent: dict[str, tuple[str, ...]] = {}
        for agent_name, agent_config in (getattr(section, "agents", None) or {}).items():
            names: list[str] = list(getattr(agent_config, "metrics", ()) or ())
            if include_llm:
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
