"""JSON persistence for evaluation payloads (Repository pattern).

Writing a report to disk is an adapter concern, not something the runner should
know how to do. Swapping this for a database or an object store means a new class
implementing :class:`~evaluation.ports.repository.EvaluationRepository` — the
runtime is untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.reporting.models import DEFAULT_REPORT_PATH
from utils.logging_config import get_logger

logger = get_logger(__name__)


class JsonEvaluationRepository:
    """Writes evaluation payloads as indented JSON.

    Args:
        directory: Directory for reports; defaults to the working directory.
        default_name: File name used when the caller does not supply one.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        default_name: str = DEFAULT_REPORT_PATH,
    ) -> None:
        self._directory = Path(directory) if directory is not None else None
        self._default_name = default_name

    def path_for(self, name: str | None = None) -> Path:
        """Resolve the destination path without writing."""
        filename = name or self._default_name
        if self._directory is None:
            return Path(filename)
        return self._directory / filename

    def save(self, payload: dict[str, Any], name: str | None = None) -> Path:
        """Write ``payload`` as JSON.

        Args:
            payload: A JSON-serializable report.
            name: File name or path. Relative names resolve against the
                configured directory.

        Returns:
            The path written.

        Raises:
            OSError: If the file cannot be written.
        """
        target = self.path_for(name)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("Wrote evaluation report to %s", target)
        return target
