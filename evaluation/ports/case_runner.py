"""Port: run one dataset case through an agent and normalize the result.

This is the seam between "a case in a file" and "an
:class:`~evaluation.core.models.EvaluationContext`". Implementations reuse the
production execution path; the runtime never builds an agent itself.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evaluation.core.dataset import EvalCase
from evaluation.core.models import EvaluationContext


@runtime_checkable
class CaseRunner(Protocol):
    """Executes evaluation cases against one agent.

    Implementations must not raise for a case that fails: the failure belongs on
    the returned context (``error``), so one bad case cannot abort a dataset run.
    """

    #: Agent name recorded on the produced contexts; must match the name used in
    #: configuration and in ``EvaluationContext.agent_name``.
    agent_name: str

    async def run_case(self, case: EvalCase) -> EvaluationContext:
        """Execute one case and return its evaluation context."""
        ...
