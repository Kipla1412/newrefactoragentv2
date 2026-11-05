"""Port: experiment tracking.

The runtime reports results here; whether that reaches MLflow, stdout or nothing
at all is an adapter concern. Every method is best-effort by contract: a tracking
failure must never abort an evaluation.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from evaluation.core.models import EvaluationResult


@runtime_checkable
class TrackerPort(Protocol):
    """Records evaluation runs and per-execution results."""

    @property
    def enabled(self) -> bool:
        """Whether a backend is configured."""
        ...

    def start(self, run_name: str, tags: Mapping[str, Any]) -> None:
        """Open a run for an offline evaluation."""
        ...

    def log_execution(self, result: EvaluationResult, **tags: Any) -> None:
        """Record one execution's metrics, statuses and rationale."""
        ...

    def log_summary(
        self,
        summary: Mapping[str, Any],
        failures: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record aggregate scores and, optionally, the failing cases."""
        ...

    def end(self) -> None:
        """Close the run, if one is open."""
        ...
