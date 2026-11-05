"""Tests for real-time evaluation.

Every test drives the observer from a plain ``asyncio.run`` wrapper (the style
used elsewhere in this suite), so no async plugin markers are needed. No network,
no agent, no MLflow: the engine is built from stub metrics and the tracker is a
recording stub.
"""

from __future__ import annotations

import asyncio
import time

from evaluation.core.engine import EvaluationEngine, MetricSelection
from evaluation.core.models import (
    EvaluationContext,
    EvaluationMode,
    EvaluationResult,
    MetricResult,
    MetricStatus,
    utc_now,
)
from evaluation.core.registry import MetricRegistry
from evaluation.runtime.observer import EvaluationObserver


class _StubMetric:
    """A metric with configurable outcome, latency and concurrency tracking."""

    def __init__(
        self,
        name: str = "stub",
        *,
        passed: bool = True,
        delay: float = 0.0,
        raises: Exception | None = None,
        tracker: "_InFlight | None" = None,
    ) -> None:
        self.name = name
        self.canonical_name = name
        self._passed = passed
        self._delay = delay
        self._raises = raises
        self._inflight = tracker
        self.calls = 0

    def measure(self, context: EvaluationContext) -> MetricResult:
        self.calls += 1
        if self._inflight is not None:
            self._inflight.enter()
        try:
            if self._delay:
                time.sleep(self._delay)
            if self._raises is not None:
                raise self._raises
            return MetricResult(
                metric_name=self.name,
                agent_name=context.agent_name,
                execution_id=context.execution_id,
                session_id=context.session_id,
                score=1.0 if self._passed else 0.0,
                threshold=1.0,
                passed=self._passed,
                reason="ok" if self._passed else "failed",
                status=MetricStatus.OK,
                evaluation_mode=context.mode,
                timestamp=utc_now(),
            )
        finally:
            if self._inflight is not None:
                self._inflight.exit()


class _InFlight:
    """Tracks how many metrics are executing at once."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0

    def enter(self) -> None:
        self.current += 1
        self.peak = max(self.peak, self.current)

    def exit(self) -> None:
        self.current -= 1


class _StubTracker:
    def __init__(self, *, raises: bool = False) -> None:
        self.results: list[EvaluationResult] = []
        self._raises = raises

    @property
    def enabled(self) -> bool:
        return True

    def log_execution(self, result, **tags) -> None:
        if self._raises:
            raise RuntimeError("tracker exploded")
        self.results.append(result)


def _context(execution_id: str = "exec-1", agent_name: str = "intake") -> EvaluationContext:
    return EvaluationContext(
        agent_name=agent_name,
        input="I have a headache.",
        output="How severe is it?",
        execution_id=execution_id,
        session_id="sess-1",
        mode=EvaluationMode.REALTIME,
    )


def _engine(*metrics: _StubMetric) -> EvaluationEngine:
    registry = MetricRegistry()
    for metric in metrics:
        registry.register(metric)
    return EvaluationEngine(
        registry, MetricSelection(common=tuple(m.name for m in metrics))
    )


class _Recorder:
    """Collects observer results."""

    def __init__(self, *, raises: bool = False) -> None:
        self.results = []
        self._raises = raises

    def __call__(self, result) -> None:
        if self._raises:
            raise RuntimeError("callback exploded")
        self.results.append(result)


class TestScheduling:
    def test_scheduling_does_not_block_the_caller(self):
        async def scenario():
            metric = _StubMetric(delay=0.25)
            observer = EvaluationObserver(_engine(metric))

            began = time.perf_counter()
            scheduled = observer.schedule(_context())
            elapsed = time.perf_counter() - began

            await observer.drain(timeout=2)
            return scheduled, elapsed, metric.calls

        scheduled, elapsed, calls = asyncio.run(scenario())

        assert scheduled is True
        assert elapsed < 0.05, f"schedule() blocked for {elapsed:.3f}s"
        assert calls == 1

    def test_result_reaches_the_tracker_and_the_callback(self):
        async def scenario():
            tracker = _StubTracker()
            recorder = _Recorder()
            observer = EvaluationObserver(
                _engine(_StubMetric()), tracker=tracker, on_result=recorder
            )

            observer.schedule(_context())
            await observer.drain(timeout=2)
            return tracker.results, recorder.results

        tracked, recorded = asyncio.run(scenario())

        assert len(tracked) == 1
        assert len(recorded) == 1
        assert tracked[0].metrics[0].passed is True
        assert tracked[0].context.mode == EvaluationMode.REALTIME

    def test_disabled_observer_schedules_nothing(self):
        async def scenario():
            metric = _StubMetric()
            observer = EvaluationObserver(_engine(metric), enabled=False)

            scheduled = observer.schedule(_context())
            await observer.drain(timeout=1)
            return scheduled, metric.calls

        scheduled, calls = asyncio.run(scenario())

        assert scheduled is False
        assert calls == 0

    def test_agent_without_metrics_is_not_scheduled(self):
        async def scenario():
            observer = EvaluationObserver(_engine())

            scheduled = observer.schedule(_context(agent_name="unknown_agent"))
            await observer.drain(timeout=1)
            return scheduled

        assert asyncio.run(scenario()) is False

    def test_scheduling_outside_an_event_loop_is_ignored(self):
        observer = EvaluationObserver(_engine(_StubMetric()))

        # Called synchronously from a test with no running loop: must not raise,
        # and must not evaluate.
        assert observer.schedule(_context()) is False

    def test_stats_are_reported(self):
        async def scenario():
            observer = EvaluationObserver(_engine(_StubMetric()))
            observer.schedule(_context("a"))
            await observer.drain(timeout=2)
            return observer.stats

        stats = asyncio.run(scenario())

        assert stats["scheduled"] == 1
        assert stats["duplicates"] == 0
        assert stats["failures"] == 0
        assert stats["pending"] == 0


class TestDuplicatePrevention:
    def test_the_same_execution_is_evaluated_once(self):
        async def scenario():
            metric = _StubMetric()
            observer = EvaluationObserver(_engine(metric))

            first = observer.schedule(_context("exec-1"))
            second = observer.schedule(_context("exec-1"))
            third = observer.schedule(_context("exec-1"))
            await observer.drain(timeout=2)
            return first, second, third, metric.calls, observer.stats

        first, second, third, calls, stats = asyncio.run(scenario())

        assert (first, second, third) == (True, False, False)
        assert calls == 1
        assert stats["duplicates"] == 2

    def test_distinct_executions_are_all_evaluated(self):
        async def scenario():
            metric = _StubMetric()
            observer = EvaluationObserver(_engine(metric))

            for i in range(3):
                observer.schedule(_context(f"exec-{i}"))
            await observer.drain(timeout=2)
            return metric.calls

        assert asyncio.run(scenario()) == 3

    def test_duplicate_memory_is_bounded(self):
        async def scenario():
            metric = _StubMetric()
            observer = EvaluationObserver(_engine(metric), max_remembered=2)

            for execution_id in ("a", "b", "c"):
                observer.schedule(_context(execution_id))
            # 'a' has been evicted from the bounded memory, so it is allowed again.
            again = observer.schedule(_context("a"))
            await observer.drain(timeout=2)
            return again, metric.calls

        again, calls = asyncio.run(scenario())

        assert again is True
        assert calls == 4

    def test_executions_without_an_id_are_never_deduplicated(self):
        async def scenario():
            metric = _StubMetric()
            observer = EvaluationObserver(_engine(metric))

            observer.schedule(_context(""))
            observer.schedule(_context(""))
            await observer.drain(timeout=2)
            return metric.calls

        assert asyncio.run(scenario()) == 2


class TestConcurrencyAndTimeouts:
    def test_concurrency_limit_is_respected(self):
        async def scenario():
            inflight = _InFlight()
            metric = _StubMetric(delay=0.1, tracker=inflight)
            observer = EvaluationObserver(
                _engine(metric), max_concurrency=1, timeout=5
            )

            for i in range(3):
                observer.schedule(_context(f"exec-{i}"))
            await observer.drain(timeout=5)
            return inflight.peak, metric.calls

        peak, calls = asyncio.run(scenario())

        assert calls == 3
        assert peak == 1, "max_concurrency=1 must serialise evaluations"

    def test_higher_concurrency_allows_overlap(self):
        async def scenario():
            inflight = _InFlight()
            metric = _StubMetric(delay=0.15, tracker=inflight)
            observer = EvaluationObserver(
                _engine(metric), max_concurrency=3, timeout=5
            )

            for i in range(3):
                observer.schedule(_context(f"exec-{i}"))
            await observer.drain(timeout=5)
            return inflight.peak

        assert asyncio.run(scenario()) >= 2

    def test_timeout_is_counted_and_never_propagates(self):
        async def scenario():
            recorder = _Recorder()
            observer = EvaluationObserver(
                _engine(_StubMetric(delay=0.4)), on_result=recorder, timeout=0.05
            )

            observer.schedule(_context())
            await observer.drain(timeout=3)
            return observer.stats, recorder.results

        stats, results = asyncio.run(scenario())

        assert stats["failures"] == 1
        assert results == []

    def test_one_slow_evaluation_does_not_stop_the_others(self):
        async def scenario():
            recorder = _Recorder()
            observer = EvaluationObserver(
                _engine(_StubMetric()), on_result=recorder, timeout=2
            )

            observer.schedule(_context("ok-1"))
            observer.schedule(_context("ok-2"))
            await observer.drain(timeout=3)
            return len(recorder.results)

        assert asyncio.run(scenario()) == 2


class TestFailureIsolation:
    def test_a_raising_metric_is_reported_not_raised(self):
        async def scenario():
            recorder = _Recorder()
            observer = EvaluationObserver(
                _engine(_StubMetric(raises=RuntimeError("judge down"))),
                on_result=recorder,
                timeout=2,
            )

            observer.schedule(_context())
            await observer.drain(timeout=3)
            return recorder.results, observer.stats

        results, stats = asyncio.run(scenario())

        assert len(results) == 1
        assert results[0].errored[0].metric_name == "stub"
        assert stats["failures"] == 0

    def test_a_failing_tracker_does_not_break_evaluation(self):
        async def scenario():
            recorder = _Recorder()
            observer = EvaluationObserver(
                _engine(_StubMetric()), tracker=_StubTracker(raises=True),
                on_result=recorder, timeout=2,
            )

            observer.schedule(_context())
            await observer.drain(timeout=3)
            return recorder.results

        assert len(asyncio.run(scenario())) == 1

    def test_a_failing_callback_does_not_break_evaluation(self):
        async def scenario():
            observer = EvaluationObserver(
                _engine(_StubMetric()), on_result=_Recorder(raises=True), timeout=2
            )

            observer.schedule(_context())
            await observer.drain(timeout=3)
            return observer.stats

        stats = asyncio.run(scenario())

        assert stats["failures"] == 0
        assert stats["scheduled"] == 1


class TestDrain:
    def test_drain_waits_for_pending_work(self):
        async def scenario():
            recorder = _Recorder()
            observer = EvaluationObserver(
                _engine(_StubMetric(delay=0.2)), on_result=recorder, timeout=5
            )

            observer.schedule(_context())
            assert observer.pending == 1

            await observer.drain(timeout=5)
            return observer.pending, len(recorder.results)

        pending, results = asyncio.run(scenario())

        assert pending == 0
        assert results == 1

    def test_drain_abandons_stragglers_after_its_timeout(self):
        async def scenario():
            observer = EvaluationObserver(
                _engine(_StubMetric(delay=5)), timeout=30
            )

            observer.schedule(_context())
            await observer.drain(timeout=0.05)
            return observer.pending

        assert asyncio.run(scenario()) == 0

    def test_drain_with_nothing_pending_returns_immediately(self):
        async def scenario():
            observer = EvaluationObserver(_engine(_StubMetric()))
            await observer.drain(timeout=1)
            return True

        assert asyncio.run(scenario()) is True
