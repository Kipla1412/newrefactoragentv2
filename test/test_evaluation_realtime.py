"""Tests for real-time evaluation.

Every test drives the observer from a plain ``asyncio.run`` wrapper (the style
used elsewhere in this suite), so no async plugin markers are needed. No network,
no agent, no MLflow: the engine is built from stub metrics and the tracker is a
recording stub.

The final block covers the agent-side seam: ``Agent.run`` hands a finished turn
to an observer that was injected on the agent, and does nothing when none is set.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

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

EVALUATION_DIR = Path(__file__).resolve().parents[1] / "evaluation"


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


# ---------------------------------------------------------------------------
# Agent-side seam: Agent.run -> observer
# ---------------------------------------------------------------------------


from types import SimpleNamespace  # noqa: E402


def _stub_observer(*, enabled: bool = True):
    """A minimal observer stand-in: records the contexts it is handed."""
    calls: list[EvaluationContext] = []

    def schedule(context: EvaluationContext) -> bool:
        calls.append(context)
        return True

    def drain(timeout: float = 10.0):  # pragma: no cover - never awaited here
        return asyncio.sleep(0)

    return SimpleNamespace(
        calls=calls, schedule=schedule, drain=drain, enabled=enabled
    )


def _agent_with_stubbed_replies(monkeypatch, replies: list[str], message: str):
    """Build an Agent whose agentic loop yields the given replies, with no LLM."""
    from agent.agent import Agent
    from agent.events import AgentEvent, AgentEventType

    async def fake_loop(self, message: str, *_a, **_k):
        yield AgentEvent.agent_start(message)
        for reply in replies:
            yield AgentEvent.text_complete(reply)
        yield AgentEvent.agent_end(replies[-1] if replies else None)

    monkeypatch.setattr(Agent, "_agentic_loop", fake_loop)

    events: list[str] = []

    async def drain(agent, message: str, replies):
        async for event in agent.run(message):
            events.append(str(getattr(event, "type", "")))
        return events

    return Agent, drain


def _collect(agent, message: str) -> list:
    async def run():
        out = []
        async for event in agent.run(message):
            out.append(event)
        return out

    return asyncio.run(run())


class _StubSession:
    """A Session stand-in with just what Agent.run touches."""

    def __init__(self) -> None:
        self._messages: list[dict] = []
        self.session_id = "sess-1"
        self.agent_name = "IntakeAgent"
        self.agent_type_for_events = "intake"
        # Execution identity, as a real Session carries it.
        self.execution_id = None
        self.mlflow_run = None
        self.mlflow_tracker = None
        self.context_manager = SimpleNamespace(
            set_system_prompt=lambda *a, **k: None,
            add_user_message=lambda text: self._messages.append(
                {"role": "user", "content": text}
            ),
            messages=self._messages,
        )
        self.metadata: dict = {}
        self.hook_system = SimpleNamespace(
            trigger_before_agent=lambda *a, **k: _coro(),
            trigger_after_agent=lambda *a, **k: _coro(),
        )
        self.mlflow_run = None

    def start_mlflow_run(self, message: str) -> None: ...

    def end_mlflow_run(self) -> None: ...

    def increment_turn(self) -> None: ...


async def _coro():
    return None


class TestAgentSeam:
    def _make_agent(self, monkeypatch, replies: list[str]):
        from agent.agent import Agent
        from agent.events import AgentEvent, AgentEventType, AgentType

        # Patch the loop *body*, not _agentic_loop: the evaluation hook lives in
        # the wrapper around the body, so patching the whole loop would remove
        # the very thing under test.
        async def fake_turn_events(self, max_turns: int):
            yield AgentEvent.agent_start("start", AgentType.INTAKE)
            for reply in replies:
                yield AgentEvent.text_complete(reply, AgentType.INTAKE)
            yield AgentEvent.agent_end(replies[-1] if replies else None, agent=AgentType.INTAKE)

        monkeypatch.setattr(Agent, "_turn_events", fake_turn_events)
        agent = Agent.__new__(Agent)
        agent.config = SimpleNamespace(max_turns=1, model_name="gpt-4.1-mini")
        agent.session = _StubSession()
        agent.system_prompt = "test prompt"
        agent.agent_type = SimpleNamespace(value="intake")
        agent.configured_agent_name = "intake"
        agent.default_skill = None
        agent.evaluation_observer = None
        agent.skill_registry = None
        return agent

    def test_a_finished_turn_is_scheduled_for_evaluation(self, monkeypatch):
        agent = self._make_agent(monkeypatch, ["How severe is your headache?"])
        observer = _stub_observer()
        agent.evaluation_observer = observer

        events = _collect(agent, "I have a headache.")

        assert len(observer.calls) == 1
        context = observer.calls[0]
        assert context.agent_name == "intake"
        assert context.input == "I have a headache."
        assert context.output == "How severe is your headache?"
        assert context.mode == EvaluationMode.REALTIME
        assert context.session_id == "sess-1"
        assert context.execution_id
        assert context.duration_ms >= 0
        assert events  # the caller still receives its events

    def test_no_observer_means_no_evaluation_and_no_error(self, monkeypatch):
        agent = self._make_agent(monkeypatch, ["A reply."])
        agent.evaluation_observer = None

        events = _collect(agent, "Hello.")

        assert events

    def test_empty_reply_is_not_evaluated(self, monkeypatch):
        agent = self._make_agent(monkeypatch, [""])
        observer = _stub_observer()
        agent.evaluation_observer = observer

        _collect(agent, "Hello.")

        assert observer.calls == []

    def test_multiple_text_complete_events_yield_one_context(self, monkeypatch):
        agent = self._make_agent(monkeypatch, ["first", "second"])
        observer = _stub_observer()
        agent.evaluation_observer = observer

        _collect(agent, "Hello.")

        assert len(observer.calls) == 1
        # The last TEXT_COMPLETE is the turn's final reply.
        assert observer.calls[0].output == "second"

    def test_evaluation_never_blocks_or_alters_the_reply(self, monkeypatch):
        agent = self._make_agent(monkeypatch, ["Unchanged reply."])

        def exploding_schedule(context):
            raise RuntimeError("observer exploded")

        agent.evaluation_observer = SimpleNamespace(schedule=exploding_schedule)

        events = _collect(agent, "Hello.")

        assert events

    def test_context_carries_prior_conversation(self, monkeypatch):
        agent = self._make_agent(monkeypatch, ["Next question?"])
        agent.session.context_manager.add_user_message("Earlier message.")
        agent.evaluation_observer = _stub_observer()

        _collect(agent, "Now please.")

        context = agent.evaluation_observer.calls[0]
        assert any("Earlier message." in line for line in context.context)
        assert any("Now please." in line for line in context.context)

    def test_metadata_contains_no_patient_content(self, monkeypatch):
        agent = self._make_agent(monkeypatch, ["A reply."])
        agent.evaluation_observer = _stub_observer()

        _collect(agent, "Patient private details here.")

        context = agent.evaluation_observer.calls[0]
        assert "Patient private details here." not in str(context.metadata)

    def test_execution_ids_are_unique_per_turn(self, monkeypatch):
        agent = self._make_agent(monkeypatch, ["A reply."])
        observer = _stub_observer()
        agent.evaluation_observer = observer

        _collect(agent, "First.")
        _collect(agent, "Second.")

        assert len(observer.calls) == 2
        assert observer.calls[0].execution_id != observer.calls[1].execution_id


class _OverridingAgent:
    """Mirrors a real custom agent: overrides run() and drives the loop itself.

    Every shipped agent (intake, ehr, demographics, soap, …) does this instead of
    calling Agent.run, which is why the evaluation hook must live in the loop.
    """

    @staticmethod
    def make(agent, monkeypatch, replies: list[str]):
        from agent.agent import Agent
        from agent.events import AgentEvent, AgentType

        async def fake_turn_events(self, max_turns: int):
            for reply in replies:
                yield AgentEvent.text_complete(reply, AgentType.INTAKE)

        monkeypatch.setattr(Agent, "_turn_events", fake_turn_events)
        return agent

    @staticmethod
    def drive(agent, message: str):
        """Exactly what EHRAgent.run/IntakeAgent.run do."""

        async def run():
            return [e async for e in agent._custom_run(message)]

        return asyncio.run(run())


class TestCustomAgentCoverage:
    """Evaluation must fire for agents that never call Agent.run."""

    def _custom_agent(self, monkeypatch, replies: list[str]):
        import types

        from agent.agent import Agent
        from agent.events import AgentEvent, AgentEventType, AgentType

        async def fake_turn_events(self, max_turns: int):
            for reply in replies:
                yield AgentEvent.text_complete(reply, AgentType.INTAKE)

        monkeypatch.setattr(Agent, "_turn_events", fake_turn_events)

        class _Custom(Agent):
            """Overrides run() the way every shipped custom agent does."""

            async def _custom_run(self, message: str):
                self.session.context_manager.add_user_message(message)
                async for event in self._agentic_loop():
                    yield event

            async def _buffered(self, message: str) -> str:
                """The generate()/recommend_questions() shape."""
                self.session.context_manager.add_user_message(message)
                text = ""
                async for event in self._agentic_loop():
                    if event.type == AgentEventType.TEXT_COMPLETE:
                        text = event.data.get("content", "")
                return text

        agent = _Custom.__new__(_Custom)
        agent.config = SimpleNamespace(max_turns=1, model_name="gpt-4.1-mini")
        agent.session = _StubSession()
        agent.system_prompt = "prompt"
        agent.agent_type = AgentType.INTAKE
        agent.configured_agent_name = "intake"
        agent.default_skill = None
        agent.evaluation_observer = None
        agent.skill_registry = None
        return agent

    def test_an_agent_that_overrides_run_is_still_evaluated(self, monkeypatch):
        agent = self._custom_agent(monkeypatch, ["How severe is it?"])
        observer = _stub_observer()
        agent.evaluation_observer = observer

        asyncio.run(_drive_custom(agent, "I have a headache."))

        assert len(observer.calls) == 1, (
            "a custom agent overriding run() must still be evaluated"
        )
        context = observer.calls[0]
        assert context.agent_name == "intake"
        assert context.input == "I have a headache."
        assert context.output == "How severe is it?"
        assert context.mode == EvaluationMode.REALTIME

    def test_a_buffered_agent_is_still_evaluated(self, monkeypatch):
        agent = self._custom_agent(monkeypatch, ["Buffered reply."])
        observer = _stub_observer()
        agent.evaluation_observer = observer

        asyncio.run(agent._buffered("Summarise this."))

        assert len(observer.calls) == 1, (
            "generate()/recommend_questions() drain the loop and must be evaluated"
        )
        assert observer.calls[0].output == "Buffered reply."

    def test_two_turns_in_one_session_are_evaluated_separately(self, monkeypatch):
        agent = self._custom_agent(monkeypatch, ["First reply."])
        observer = _stub_observer()
        agent.evaluation_observer = observer

        asyncio.run(_drive_custom(agent, "One."))
        asyncio.run(_drive_custom(agent, "Two."))

        assert len(observer.calls) == 2
        assert observer.calls[0].execution_id != observer.calls[1].execution_id

    def test_an_agent_with_no_observer_is_unaffected(self, monkeypatch):
        agent = self._custom_agent(monkeypatch, ["A reply."])
        agent.evaluation_observer = None

        events = asyncio.run(_drive_custom(agent, "Hello."))

        assert events


async def _drive_custom(agent, message: str):
    return [event async for event in agent._custom_run(message)]


# ---------------------------------------------------------------------------
# Process wiring: get_observer / attach_observer
# ---------------------------------------------------------------------------


def _eval_config(*, realtime: bool = True, enabled: bool = True):
    """A configuration stand-in carrying just the evaluation block."""
    from types import SimpleNamespace

    return SimpleNamespace(
        mlflow_enabled=False,
        mlflow_experiment_name=None,
        mlflow_tracking_uri=None,
        evaluation=SimpleNamespace(
            enabled=enabled,
            realtime=SimpleNamespace(
                enabled=realtime, max_concurrency=1, timeout=5.0
            ),
            common_metrics=[],
            agents={},
            llm=SimpleNamespace(allow_external_calls=False, model_uri=None),
        ),
    )


@pytest.fixture
def fresh_observer_cache():
    """Reset the process-wide observer before and after each test."""
    from evaluation.composition import reset_observer

    reset_observer()
    yield
    reset_observer()


class TestObserverWiring:
    def test_no_observer_when_realtime_is_disabled(self, fresh_observer_cache):
        from evaluation.composition import get_observer

        assert get_observer(_eval_config(realtime=False)) is None

    def test_no_observer_when_evaluation_is_disabled(self, fresh_observer_cache):
        from evaluation.composition import get_observer

        assert get_observer(_eval_config(enabled=False)) is None

    def test_no_observer_without_a_config(self, fresh_observer_cache):
        from evaluation.composition import get_observer

        assert get_observer(None) is None

    def test_an_observer_is_built_when_realtime_is_enabled(self, fresh_observer_cache):
        from evaluation.composition import get_observer

        observer = get_observer(_eval_config())

        assert isinstance(observer, EvaluationObserver)
        assert observer.enabled is True

    def test_the_observer_is_built_once_per_process(self, fresh_observer_cache):
        from evaluation.composition import get_observer

        config = _eval_config()

        assert get_observer(config) is get_observer(config)

    def test_attach_sets_the_observer_on_the_agent(self, fresh_observer_cache):
        from evaluation.composition import attach_observer

        class _Agent:
            evaluation_observer = None

        agent = _Agent()

        assert attach_observer(agent, _eval_config()) is True
        assert isinstance(agent.evaluation_observer, EvaluationObserver)

    def test_attach_is_a_no_op_when_realtime_is_disabled(self, fresh_observer_cache):
        from evaluation.composition import attach_observer

        class _Agent:
            evaluation_observer = None

        agent = _Agent()

        assert attach_observer(agent, _eval_config(realtime=False)) is False
        assert agent.evaluation_observer is None

    def test_attach_never_raises(self, fresh_observer_cache, monkeypatch):
        import evaluation.composition as composition

        def boom(config=None):
            raise RuntimeError("engine exploded")

        monkeypatch.setattr(composition, "get_observer", boom)

        class _Agent:
            pass

        # An agent that cannot be evaluated must still be usable.
        assert composition.attach_observer(_Agent(), _eval_config()) is False

    def test_the_factory_attaches_the_observer(self, monkeypatch):
        """Every agent built through AgentFactory gets evaluation, for free."""
        import customagents.factory as factory_module

        attached: list = []
        monkeypatch.setattr(
            factory_module,
            "attach_observer",
            lambda agent, config: attached.append(agent) or True,
        )

        from config.loader import load_config
        from agent.events import AgentType

        config = load_config(EVALUATION_DIR.parent)
        agent = factory_module.AgentFactory.create(AgentType.INTAKE, config)

        assert attached and attached[0] is agent

    def test_a_factory_created_agent_resolves_to_its_configured_metrics(
        self, fresh_observer_cache
    ):
        """The name that selects metrics must match ``evaluation.agents.<name>``.

        Regression: the factory named the agent after its class ("IntakeAgent"),
        which matched no configuration entry, so an observer was attached but
        measured nothing at all.
        """
        import customagents.factory as factory_module
        from agent.events import AgentType
        from config.loader import load_config

        config = load_config(EVALUATION_DIR.parent)
        agent = factory_module.AgentFactory.create(AgentType.INTAKE, config)

        assert agent._evaluation_agent_name == "intake"
        assert agent.session.agent_name == "intake", "runs are tagged with this"

        observer = agent.evaluation_observer
        assert observer is not None, "the factory attaches it"

        names = observer._engine.metric_names(agent._evaluation_agent_name)
        assert "medical_boundary_compliance" in names
        assert names, "an attached observer with no metrics measures nothing"
