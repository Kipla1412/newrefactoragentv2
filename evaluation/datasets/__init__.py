"""Golden datasets — one versioned file per agent.

The dataset *schema* lives in :mod:`evaluation.core.dataset`; this package is the
infrastructure that finds and reads the files.

- :mod:`evaluation.datasets.loader` — path resolution and loading.
- ``datasets/<agent>/golden.json`` — the default dataset for an agent.
- ``datasets/<agent>/<version>.json`` — a pinned version.

The schema types are re-exported here so callers can use one import site.
"""

from evaluation.core.dataset import DatasetError, EvalCase, EvalDataset
from evaluation.datasets.loader import (
    DATASETS_ROOT,
    DEFAULT_VERSION_FILE,
    FileDatasetSource,
    dataset_path,
    load_dataset,
)

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
