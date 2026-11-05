"""Port: record evaluation results.

Two implementations exist today — MLflow for experiment tracking
(:mod:`evaluation.adapters.mlflow_tracker`) and JSON for local reports
(:mod:`evaluation.adapters.json_repository`). The runtime depends on this port, so
adding a third backend never touches it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from evaluation.core.models import EvaluationResult


@runtime_checkable
class EvaluationRepository(Protocol):
    """Persists evaluation results."""

    def save(self, payload: dict[str, Any], name: str) -> Path:
        """Persist a JSON-serializable payload.

        Args:
            payload: The report, already JSON-serializable.
            name: Logical name for the artifact.

        Returns:
            The path written.
        """
        ...


@runtime_checkable
class ResultSink(Protocol):
    """Receives one execution's results.

    Used by the real-time observer, which produces the same
    :class:`~evaluation.core.models.EvaluationResult` as the offline runner.
    """

    def record(self, result: EvaluationResult) -> None:
        """Record one evaluation result. Must never raise."""
        ...
