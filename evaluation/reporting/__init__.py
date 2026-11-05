"""Reporting — turns standardized results into a run report.

Depends only on :mod:`evaluation.core`, so a report can be produced without an
agent, an MLflow server or a dataset file.
"""

from evaluation.reporting.models import (
    DEFAULT_REPORT_PATH,
    CaseResult,
    EvaluationReport,
)

__all__ = [
    "DEFAULT_REPORT_PATH",
    "CaseResult",
    "EvaluationReport",
]
