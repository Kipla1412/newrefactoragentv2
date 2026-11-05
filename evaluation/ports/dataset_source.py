"""Port: where golden datasets come from (Repository pattern).

The runtime asks for "the intake dataset, version 1.1" and does not care whether
that is a file, a database row or an HTTP response.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evaluation.core.dataset import EvalDataset


@runtime_checkable
class DatasetSource(Protocol):
    """Loads versioned golden datasets by agent."""

    def load(self, agent: str, version: str | None = None) -> EvalDataset:
        """Load one agent's dataset.

        Args:
            agent: Agent name, e.g. ``"intake"``.
            version: Version to load; ``None`` means the default version.

        Returns:
            The validated dataset.

        Raises:
            DatasetError: If the dataset is missing or invalid.
        """
        ...
