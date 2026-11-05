"""Loading datasets — the Repository pattern over versioned golden files.

Layout::

    evaluation/datasets/
    ├── intake/
    │   ├── golden.json      # default version
    │   └── 1.1.json         # a pinned version
    ├── pdf_chat/golden.json
    └── text_to_sql/golden.json

One file per agent per version, so datasets are versioned and independently
maintained. :class:`FileDatasetSource` implements the
:class:`~evaluation.ports.dataset_source.DatasetSource` port, which is what lets
the runner be tested without touching the filesystem.

This module is infrastructure: it knows about paths and JSON. The schema it loads
lives in :mod:`evaluation.core.dataset`.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from evaluation.core.dataset import DatasetError, EvalCase, EvalDataset

#: Root of the bundled datasets.
DATASETS_ROOT = Path(__file__).resolve().parent

#: File name used when no version is pinned.
DEFAULT_VERSION_FILE = "golden.json"


def dataset_path(
    agent: str,
    version: str | None = None,
    *,
    root: Path | None = None,
) -> Path:
    """Resolve the file for one agent and version.

    Args:
        agent: Agent name, e.g. ``"intake"``.
        version: Version file stem, e.g. ``"1.1"``. ``None`` means the default.
        root: Dataset root; defaults to the bundled directory.

    Returns:
        The path the dataset would live at.
    """
    base = (root or DATASETS_ROOT) / agent
    if version is None or version == "":
        return base / DEFAULT_VERSION_FILE
    stem = version if version.endswith(".json") else f"{version}.json"
    return base / stem


def load_dataset(
    path: str | Path | None = None,
    *,
    agent: str = "intake",
    version: str | None = None,
    root: Path | None = None,
) -> EvalDataset:
    """Load and validate an evaluation dataset.

    Args:
        path: Explicit file to load. When given, ``agent``/``version`` are ignored.
        agent: Agent whose default dataset to load.
        version: Dataset version; ``None`` loads ``golden.json``.
        root: Dataset root; defaults to the bundled directory.

    Returns:
        The validated dataset.

    Raises:
        DatasetError: If the file is missing, is not valid JSON, or fails
            validation.
    """
    resolved = Path(path) if path is not None else dataset_path(agent, version, root=root)
    display = str(resolved)

    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"Could not read dataset at {display}", cause=exc) from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Invalid JSON in dataset {display}: {exc}") from exc

    if not isinstance(payload, dict):
        raise DatasetError(f"Invalid dataset {display}: expected a JSON object")

    try:
        return EvalDataset(**payload)
    except ValidationError as exc:
        raise DatasetError(
            f"Invalid dataset {display}: {exc.error_count()} problem(s)",
            details={"errors": exc.errors(include_url=False)},
        ) from exc
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"Invalid dataset {display}: {exc}") from exc


class FileDatasetSource:
    """Filesystem-backed dataset repository.

    Implements :class:`~evaluation.ports.dataset_source.DatasetSource`, so the
    offline runner depends on the port rather than on this class.

    Args:
        root: Dataset root; defaults to the bundled directory.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root) if root is not None else DATASETS_ROOT

    @property
    def root(self) -> Path:
        """The dataset root this source reads from."""
        return self._root

    def load(self, agent: str, version: str | None = None) -> EvalDataset:
        """Load one agent's dataset."""
        return load_dataset(agent=agent, version=version, root=self._root)

    def path_for(self, agent: str, version: str | None = None) -> Path:
        """The path a dataset would live at."""
        return dataset_path(agent, version, root=self._root)

    def available(self) -> tuple[str, ...]:
        """Agent names that have at least one dataset on disk."""
        if not self._root.is_dir():
            return ()
        return tuple(
            sorted(
                child.name
                for child in self._root.iterdir()
                if child.is_dir() and any(child.glob("*.json"))
            )
        )

    def versions(self, agent: str) -> tuple[str, ...]:
        """Version stems available for one agent, ``golden`` first."""
        directory = self._root / agent
        if not directory.is_dir():
            return ()
        stems = sorted(p.stem for p in directory.glob("*.json"))
        return tuple(sorted(stems, key=lambda s: (s != "golden", s)))


__all__ = [
    "DATASETS_ROOT",
    "DEFAULT_VERSION_FILE",
    "DatasetError",
    "EvalCase",
    "EvalDataset",
    "FileDatasetSource",
    "dataset_path",
    "load_dataset",
]
