"""Real-time evaluation.

Evaluation of a live execution runs *after* the response has been produced, in a
managed background task. Nothing in this module can block or break an agent turn:

- :meth:`EvaluationObserver.schedule` never awaits anything and never raises.
- every task is bounded by a concurrency limit and a timeout.
- every failure is logged and swallowed, including at task teardown.

The observer is the real-time half of the evaluation seam. It produces the same
:class:`~evaluation.core.models.EvaluationContext` and calls the same
:class:`~evaluation.core.engine.EvaluationEngine` as the offline runner, which is
why the two modes cannot drift apart.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any, Callable

from evaluation.core.engine import EvaluationEngine
from evaluation.core.models import EvaluationContext, EvaluationResult
from utils.logging_config import get_logger

logger = get_logger(__name__)

#: Executions remembered for duplicate prevention.
DEFAULT_REMEMBERED = 1024


class EvaluationObserver:
    """Evaluate live executions in the background.

    Args:
        engine: The shared evaluation engine.
        tracker: Optional tracker; receives one result per execution.
        on_result: Optional callback invoked with each result. Must not raise.
        max_concurrency: How many evaluations may run at once.
        timeout: Per-evaluation timeout in seconds.
        max_remembered: How many execution ids to remember for duplicate
            prevention. Bounded so a long-lived process cannot leak.
        enabled: Master switch. When False, :meth:`schedule` is a no-op.
    """

    def __init__(
        self,
        engine: EvaluationEngine,
        *,
        tracker: Any | None = None,
        on_result: Callable[[EvaluationResult], None] | None = None,
        max_concurrency: int = 2,
        timeout: float = 60.0,
        max_remembered: int = DEFAULT_REMEMBERED,
        enabled: bool = True,
    ) -> None:
        self._engine = engine
        self._tracker = tracker
        self._on_result = on_result
        self._timeout = float(timeout)
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._max_remembered = max(1, int(max_remembered))
        self._enabled = enabled
        self._tasks: set[asyncio.Task] = set()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._scheduled = 0
        self._skipped_duplicates = 0
        self._failures = 0

    # Introspection

    @property
    def enabled(self) -> bool:
        """Whether the observer will schedule anything."""
        return self._enabled

    @property
    def pending(self) -> int:
        """Number of evaluations still running."""
        return len(self._tasks)

    @property
    def stats(self) -> dict[str, int]:
        """Counters for observability."""
        return {
            "scheduled": self._scheduled,
            "duplicates": self._skipped_duplicates,
            "failures": self._failures,
            "pending": len(self._tasks),
        }

    # Scheduling

    def schedule(self, context: EvaluationContext) -> bool:
        """Queue one execution for background evaluation.

        Never blocks and never raises — this is called from the agent's own
        execution path, so it must be safe there.

        Args:
            context: The execution to evaluate.

        Returns:
            True when a task was created, False when it was skipped (disabled,
            duplicate, no metrics, or no running event loop).
        """
        if not self._enabled:
            return False

        if self._is_duplicate(context):
            self._skipped_duplicates += 1
            logger.debug(
                "Skipping duplicate evaluation for execution '%s'", context.execution_id
            )
            return False

        if not self._engine.metric_names(context.agent_name):
            logger.debug(
                "No metrics configured for agent '%s'; nothing scheduled",
                context.agent_name,
            )
            return False

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Called outside an event loop (e.g. a sync script). Evaluation is
            # best-effort by design, so drop it rather than block the caller.
            logger.debug("No running event loop; evaluation not scheduled")
            return False

        task = loop.create_task(
            self._evaluate(context), name=f"evaluate:{context.agent_name}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._scheduled += 1

        logger.debug(
            "Scheduled evaluation",
            extra={
                "fields": {
                    "agent": context.agent_name,
                    "execution_id": context.execution_id,
                    "pending": len(self._tasks),
                }
            },
        )
        return True

    async def drain(self, timeout: float = 10.0) -> None:
        """Wait for in-flight evaluations, so shutdown loses nothing.

        Args:
            timeout: How long to wait before abandoning the remaining tasks.
        """
        if not self._tasks:
            return

        pending = list(self._tasks)
        logger.debug("Draining %d pending evaluation(s)", len(pending))
        done, still_pending = await asyncio.wait(pending, timeout=timeout)

        if still_pending:
            for task in still_pending:
                task.cancel()
            logger.warning(
                "Abandoned %d evaluation(s) after %.1fs", len(still_pending), timeout
            )
            # Let cancellation settle, so drain never returns with tasks still
            # tracked: a caller inspecting `pending` afterwards sees the truth.
            await asyncio.gather(*still_pending, return_exceptions=True)

    # Internals

    def _is_duplicate(self, context: EvaluationContext) -> bool:
        """Whether this execution has already been evaluated."""
        key = context.execution_id
        if not key:
            return False

        if key in self._seen:
            return True

        self._seen[key] = None
        while len(self._seen) > self._max_remembered:
            self._seen.popitem(last=False)
        return False

    async def _evaluate(self, context: EvaluationContext) -> None:
        """Run the engine under the concurrency limit, swallowing all failures."""
        try:
            async with self._semaphore:
                # The engine is synchronous: LLM-judged metrics make blocking HTTP
                # calls, so it runs in a worker thread to keep the event loop free.
                result = await asyncio.wait_for(
                    asyncio.to_thread(self._engine.evaluate, context),
                    timeout=self._timeout,
                )
            self._publish(result)
        except asyncio.TimeoutError:
            self._failures += 1
            logger.warning(
                "Evaluation timed out after %.1fs",
                self._timeout,
                extra={"fields": {"execution_id": context.execution_id}},
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - evaluation must never break the agent
            self._failures += 1
            logger.exception(
                "Evaluation failed for execution '%s'", context.execution_id
            )

    def _publish(self, result: EvaluationResult) -> None:
        """Hand a finished result to the tracker and callback, best-effort."""
        if self._tracker is not None:
            try:
                self._tracker.log_execution(result)
            except Exception as exc:  # noqa: BLE001 - tracking is best-effort
                logger.warning("Tracker failed for a real-time result: %s", exc)

        if self._on_result is not None:
            try:
                self._on_result(result)
            except Exception as exc:  # noqa: BLE001 - a bad callback is not fatal
                logger.warning("on_result callback failed: %s", exc)

        if result.errored:
            logger.warning(
                "Evaluation produced errors",
                extra={
                    "fields": {
                        "execution_id": result.context.execution_id,
                        "errored": [m.metric_name for m in result.errored],
                    }
                },
            )
