"""Correlation tests: one execution, one run, one trace, one set of scores.

These run against a real MLflow tracking backend (a temporary SQLite database), so
they exercise the actual APIs rather than stubs:

- an execution is exactly one run, tagged with ``execution_id``/``session_id``
- starting an execution never terminates a run it does not own
- the root AGENT span links the trace to that run (``mlflow.sourceRun``)
- evaluation metrics written with ``run_id=`` land on that same run

The point is requirement 4: execution behaviour and evaluation results must be
discoverable together.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation.adapters.mlflow_tracker import EvaluationTracker
from evaluation.core.models import (
    EvaluationContext,
    EvaluationMode,
    EvaluationResult,
    MetricResult,
    MetricStatus,
)
from utils.mlflow_spans import agent_span, llm_span, tool_span
from utils.mlflow_tracker import (
    begin_execution,
    end_execution,
    execution_run_id,
)


def _end_any_active_run(mlflow) -> None:
    """Close leftover runs, tolerating a run that belongs to another backend.

    Ending a run touches the backend it was created in, so a run left active by
    an earlier test can raise "Run not found" once the tracking URI has moved.
    """
    while mlflow.active_run() is not None:
        try:
            mlflow.end_run()
        except Exception:  # noqa: BLE001 - leftover state must not fail a test
            break


@pytest.fixture
def tracking(tmp_path):
    """A working MLflow tracking backend, isolated per test."""
    mlflow = pytest.importorskip("mlflow")

    # Close leftovers *before* retargeting: a run from another test belongs to a
    # different backend and cannot be ended after the switch.
    _end_any_active_run(mlflow)

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("correlation-tests")

    yield mlflow

    _end_any_active_run(mlflow)


@pytest.fixture
def experiment(tracking):
    """The test's experiment, so runs can be searched by id."""
    from mlflow.tracking import MlflowClient

    return MlflowClient().get_experiment_by_name("correlation-tests")


def _result(execution_id: str = "exec-1", *, run_metrics: bool = True) -> EvaluationResult:
    context = EvaluationContext(
        agent_name="intake",
        input="I have a headache.",
        output="How severe is it?",
        execution_id=execution_id,
        session_id="sess-1",
        mode=EvaluationMode.REALTIME,
    )
    metrics = (
        (
            MetricResult(
                metric_name="answer_relevancy",
                agent_name="intake",
                execution_id=execution_id,
                session_id="sess-1",
                score=0.8,
                threshold=0.4,
                passed=True,
                reason="addresses the input",
                status=MetricStatus.OK,
                evaluation_mode=EvaluationMode.REALTIME,
                timestamp="2026-01-01T00:00:00+00:00",
            ),
            MetricResult(
                metric_name="intake_completion",
                agent_name="intake",
                execution_id=execution_id,
                session_id="sess-1",
                score=1.0,
                threshold=1.0,
                passed=True,
                reason="completes",
                status=MetricStatus.OK,
                evaluation_mode=EvaluationMode.REALTIME,
                timestamp="2026-01-01T00:00:00+00:00",
            ),
        )
        if run_metrics
        else ()
    )
    return EvaluationResult(context=context, metrics=metrics, duration_ms=12.0)


class TestExecutionRun:
    def test_one_execution_is_one_run_tagged_with_the_correlation_ids(self, tracking):
        run = begin_execution(
            tracking,
            agent_name="intake",
            session_id="sess-1",
            execution_id="exec-1",
        )

        assert run is not None
        # Re-read: the run object returned by start_run is a snapshot taken
        # before the tags were written.
        tags = tracking.get_run(run.info.run_id).data.tags
        assert tags["execution_id"] == "exec-1"
        assert tags["session_id"] == "sess-1"
        assert tags["agent_name"] == "intake"

        end_execution(tracking)
        assert tracking.active_run() is None

    def test_starting_an_execution_never_ends_a_run_it_does_not_own(self, tracking):
        outer = tracking.start_run(run_name="doctor-wrote-this-run")
        outer_id = outer.info.run_id

        run = begin_execution(
            tracking, agent_name="intake", session_id="s", execution_id="e"
        )

        # A different run, nested under the outer one rather than replacing it.
        assert run.info.run_id != outer_id
        assert run.data.tags.get("mlflow.parentRunId") == outer_id

        end_execution(tracking)

        # The outer run is still active: we only closed our own.
        assert tracking.active_run().info.run_id == outer_id
        tracking.end_run()

    def test_end_execution_is_safe_with_nothing_active(self, tracking):
        end_execution(tracking)  # must not raise
        assert tracking.active_run() is None

    def test_execution_run_id_reports_the_active_run(self, tracking):
        with tracking.start_run(run_name="x") as run:
            assert execution_run_id(tracking) == run.info.run_id

    def test_a_disabled_tracker_is_a_no_op(self):
        assert (
            begin_execution(None, agent_name="a", session_id="s", execution_id="e")
            is None
        )
        end_execution(None)
        assert execution_run_id(None) is None


class TestTraceRunCorrelation:
    def test_the_root_agent_span_links_the_trace_to_the_run(self, tracking):
        with tracking.start_run(run_name="exec") as run:
            run_id = run.info.run_id
            with agent_span(
                name="intake",
                run_id=run_id,
                execution_id="exec-1",
                session_id="sess-1",
                agent_name="intake",
            ) as span:
                assert span is not None
                trace_id = span.trace_id
                span.set_outputs({"ok": True})

        # flush=True waits for the async trace exporter, otherwise the trace may
        # not have reached the backend yet.
        traces = tracking.search_traces(run_id=run_id, return_type="list", flush=True)

        assert traces, "the trace must be findable by the execution's run id"
        matched = [t for t in traces if t.info.trace_id == trace_id]
        assert matched, "the trace must carry this run as its source run"
        assert matched[0].info.trace_metadata.get("mlflow.sourceRun") == run_id

    def test_the_agent_span_carries_the_correlation_attributes(self, tracking):
        with tracking.start_run(run_name="exec"):
            with agent_span(
                name="intake",
                execution_id="exec-9",
                session_id="sess-9",
                agent_name="intake",
            ) as span:
                assert span.span_type == "AGENT"
                assert span.get_attribute("execution_id") == "exec-9"
                assert span.get_attribute("session_id") == "sess-9"

    def test_llm_and_tool_spans_nest_inside_the_agent_span(self, tracking):
        with tracking.start_run(run_name="exec"):
            with agent_span(name="intake", execution_id="exec-1") as root:
                with llm_span(model="gpt-4.1-mini", messages=[{"role": "user"}]) as llm:
                    assert llm.span_type == "LLM"
                    assert llm.parent_id == root.span_id
                with tool_span(tool_name="fhir_search") as tool:
                    assert tool.span_type == "TOOL"
                    assert tool.parent_id == root.span_id

    def test_tool_spans_mask_arguments_and_withhold_content(self, tracking):
        with tracking.start_run(run_name="exec"):
            with tool_span(
                tool_name="fhir_search",
                arguments={"patient_id": "12345", "api_key": "secret"},
            ) as span:
                assert span.inputs["arguments"]["patient_id"] != "12345"
                assert span.inputs["arguments"]["api_key"] == "***"

    def test_spans_still_work_with_no_run_id(self, tracking):
        with agent_span(name="intake", execution_id="exec-1") as span:
            assert span is not None


class TestEvaluationLandsOnTheExecutionRun:
    @pytest.fixture
    def tracker(self, tracking, monkeypatch):
        """An EvaluationTracker whose MLflow backend is the test's."""
        import evaluation.adapters.mlflow_tracker as module

        monkeypatch.setattr(module, "get_mlflow_tracker", lambda config: tracking)
        config = SimpleNamespace(
            mlflow_enabled=True,
            mlflow_tracking_uri=tracking.get_tracking_uri(),
            mlflow_experiment_name="correlation-tests",
        )
        return EvaluationTracker(config)

    def test_metrics_are_written_to_the_execution_run(self, tracker, tracking):
        from mlflow.tracking import MlflowClient

        run = begin_execution(
            tracking, agent_name="intake", session_id="sess-1", execution_id="exec-1"
        )
        run_id = run.info.run_id
        end_execution(tracking)

        tracker.log_execution(_result("exec-1"), run_id=run_id)

        client = MlflowClient()
        metrics = client.get_run(run_id).data.metrics

        assert metrics["answer_relevancy.score"] == 0.8
        assert metrics["answer_relevancy.passed"] == 1.0
        assert metrics["intake_completion.score"] == 1.0
        assert metrics["passed"] == 2.0

    def test_no_second_run_is_created_for_the_same_execution(
        self, tracker, tracking, experiment
    ):
        from mlflow.tracking import MlflowClient

        run = begin_execution(
            tracking, agent_name="intake", session_id="sess-1", execution_id="exec-1"
        )
        run_id = run.info.run_id
        end_execution(tracking)

        client = MlflowClient()
        before = len(client.search_runs([experiment.experiment_id]))
        tracker.log_execution(_result("exec-1"), run_id=run_id)
        after = len(client.search_runs([experiment.experiment_id]))

        assert after == before, "evaluation must not add a rival run"

    def test_the_execution_id_is_tagged_on_the_run_it_writes_to(self, tracker, tracking):
        from mlflow.tracking import MlflowClient

        run = begin_execution(
            tracking, agent_name="intake", session_id="sess-1", execution_id="exec-7"
        )
        run_id = run.info.run_id
        end_execution(tracking)

        tracker.log_execution(_result("exec-7"), run_id=run_id)

        tags = MlflowClient().get_run(run_id).data.tags
        assert tags["execution_id"] == "exec-7"
        assert tags["agent_name"] == "intake"

    def test_missing_metrics_are_reported_as_skipped_tags(self, tracker, tracking):
        from mlflow.tracking import MlflowClient

        run = begin_execution(
            tracking, agent_name="intake", session_id="s", execution_id="exec-8"
        )
        run_id = run.info.run_id
        end_execution(tracking)

        skipped_result = EvaluationResult(
            context=_result("exec-8").context,
            metrics=(
                MetricResult(
                    metric_name="faithfulness",
                    agent_name="intake",
                    execution_id="exec-8",
                    session_id="s",
                    score=0.0,
                    threshold=1.0,
                    passed=False,
                    reason="requires an MLflow trace",
                    status=MetricStatus.SKIPPED,
                    evaluation_mode=EvaluationMode.REALTIME,
                    timestamp="2026-01-01T00:00:00+00:00",
                ),
            ),
        )
        tracker.log_execution(skipped_result, run_id=run_id)

        tags = MlflowClient().get_run(run_id).data.tags
        assert "skipped.faithfulness" in tags
        assert "trace" in tags["skipped.faithfulness"]


async def _coro():
    return None


class _RecordingObserver:
    """Stands in for EvaluationObserver: records the contexts it is handed."""

    def __init__(self) -> None:
        self.calls: list[EvaluationContext] = []

    def schedule(self, context: EvaluationContext) -> bool:
        self.calls.append(context)
        return True


class TestAgentExecutionCorrelation:
    """One agent turn: its run, its trace and its evaluation share one id."""

    @staticmethod
    def _session(tracking):
        """A Session stand-in that uses the framework's real run lifecycle."""
        import uuid
        from types import SimpleNamespace

        from utils.mlflow_tracker import begin_execution, end_execution

        class _Session:
            def __init__(self) -> None:
                self.session_id = "sess-42"
                self.agent_name = "IntakeAgent"
                self.execution_id: str | None = None
                self.mlflow_run = None
                self.mlflow_tracker = tracking
                self._messages: list[dict] = []
                self.context_manager = SimpleNamespace(
                    set_system_prompt=lambda *a, **k: None,
                    add_user_message=lambda text: self._messages.append(
                        {"role": "user", "content": text}
                    ),
                    messages=self._messages,
                )
                self.hook_system = SimpleNamespace(
                    trigger_before_agent=lambda *a, **k: _coro(),
                    trigger_after_agent=lambda *a, **k: _coro(),
                )

            def start_mlflow_run(self, message: str) -> None:
                if not self.execution_id:
                    self.execution_id = str(uuid.uuid4())
                self.mlflow_run = begin_execution(
                    tracking,
                    agent_name="intake",
                    session_id=self.session_id,
                    execution_id=self.execution_id,
                )

            def end_mlflow_run(self) -> None:
                end_execution(tracking)
                self.mlflow_run = None

        return _Session()

    def _run_agent(self, tracking, monkeypatch, reply: str = "How severe is it?"):
        """Run one agent turn and return the contexts evaluation was handed.

        The observer is a recorder, and it is invoked during the turn — which is
        when the execution run is still open. Asserting after the turn returns
        would be wrong: by then the run has been closed and the session no longer
        holds it.
        """
        import asyncio

        from agent.agent import Agent
        from agent.events import AgentEvent, AgentType

        async def fake_turn_events(self, max_turns: int):
            # Patch the loop *body*: evaluation is scheduled by the wrapper
            # around it (_agentic_loop), so patching the loop itself would remove
            # the behaviour under test.
            yield AgentEvent.text_complete(reply, AgentType.INTAKE)

        monkeypatch.setattr(Agent, "_turn_events", fake_turn_events)

        recorder = _RecordingObserver()

        agent = Agent.__new__(Agent)
        agent.config = SimpleNamespace(max_turns=1, model_name="gpt-4.1-mini")
        agent.session = self._session(tracking)
        agent.system_prompt = "intake prompt"
        agent.agent_type = AgentType.INTAKE
        agent.configured_agent_name = "intake"
        agent.default_skill = None
        agent.evaluation_observer = recorder
        agent.skill_registry = None

        async def drive():
            return [event async for event in agent.run("I have a headache.")]

        asyncio.run(drive())
        return agent, recorder

    def test_one_turn_produces_one_run(self, tracking, monkeypatch):
        agent, recorder = self._run_agent(tracking, monkeypatch)

        assert len(recorder.calls) == 1
        contexts = recorder.calls

        assert contexts[0].execution_id == agent.session.execution_id
        assert contexts[0].session_id == "sess-42"

    def test_the_evaluation_context_points_at_the_executions_run(
        self, tracking, monkeypatch
    ):
        from mlflow.tracking import MlflowClient

        _, recorder = self._run_agent(tracking, monkeypatch)
        context = recorder.calls[0]

        run_id = context.metadata["mlflow_run_id"]
        assert run_id, "the execution's run id must reach evaluation"

        tags = MlflowClient().get_run(run_id).data.tags
        assert tags["execution_id"] == context.execution_id
        assert tags["session_id"] == "sess-42"
        assert tags["agent_name"] == "intake"

    def test_scores_and_the_agent_run_end_up_on_one_run(self, tracking, monkeypatch):
        from mlflow.tracking import MlflowClient

        import evaluation.adapters.mlflow_tracker as module

        monkeypatch.setattr(module, "get_mlflow_tracker", lambda config: tracking)
        tracker = EvaluationTracker(
            SimpleNamespace(
                mlflow_enabled=True,
                mlflow_tracking_uri=tracking.get_tracking_uri(),
                mlflow_experiment_name="correlation-tests",
            )
        )

        _, recorder = self._run_agent(tracking, monkeypatch)
        context = recorder.calls[0]
        run_id = context.metadata["mlflow_run_id"]
        execution_id = context.execution_id

        # Exactly what EvaluationObserver does with a finished result.
        tracker.log_execution(_result(execution_id), run_id=run_id)

        run = MlflowClient().get_run(run_id)
        assert run.data.tags["execution_id"] == execution_id
        assert run.data.metrics["answer_relevancy.score"] == 0.8

    def test_one_execution_run_is_created_per_turn_not_two(self, tracking, monkeypatch):
        from mlflow.tracking import MlflowClient

        _, recorder = self._run_agent(tracking, monkeypatch)
        execution_id = recorder.calls[0].execution_id

        runs = MlflowClient().search_runs(
            [self._experiment_id(tracking)],
            filter_string=f"tags.execution_id = '{execution_id}'",
        )
        assert len(runs) == 1, "one execution must map to exactly one run"

    @staticmethod
    def _experiment_id(tracking) -> str:
        from mlflow.tracking import MlflowClient

        return MlflowClient().get_experiment_by_name("correlation-tests").experiment_id

    def test_a_finished_execution_run_is_closed(self, tracking, monkeypatch):
        agent, _ = self._run_agent(tracking, monkeypatch)

        assert agent.session.mlflow_run is None
        assert tracking.active_run() is None

    def test_no_run_is_started_before_the_turn_begins(self, tracking):
        """A session is not an execution: nothing is created at session build."""
        session = self._session(tracking)

        assert session.mlflow_run is None
        assert session.execution_id is None
        assert tracking.active_run() is None
