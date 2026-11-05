"""Ports — the interfaces the runtime depends on (Dependency Inversion).

Each port is a :class:`~typing.Protocol`, so implementations need not inherit
from anything and tests can substitute a stub without a subclass. The runtime in
:mod:`evaluation.runtime` imports only these, never a concrete adapter.
"""

from evaluation.ports.case_runner import CaseRunner
from evaluation.ports.dataset_source import DatasetSource
from evaluation.ports.repository import EvaluationRepository
from evaluation.ports.tracker import TrackerPort

__all__ = [
    "CaseRunner",
    "DatasetSource",
    "EvaluationRepository",
    "TrackerPort",
]
