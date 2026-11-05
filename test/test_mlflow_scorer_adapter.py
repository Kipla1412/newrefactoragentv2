"""Tests for the MLflow DeepEval scorer adapter.

No network access: the scorer is injected as a stub. The one test that touches
MLflow only *constructs* a scorer, which resolves credentials lazily.
"""

from __future__ import annotations

import pytest

from evaluation.core.engine import EvaluationEngine, MetricSelection
from evaluation.adapters.mlflow_scorer import (
    MLFLOW_SCORERS,
    MLflowDeepEvalMetric,
    ScorerSpec,
    build_scorer,
    feedback_to_metric_result,
    mlflow_scorers_available,
    register,
)
from evaluation.core.models import (
    EvaluationContext,
    EvaluationMode,
    MetricStatus,
)
from evaluation.core.registry import MetricRegistry

pytestmark = pytest.mark.skipif(
    not mlflow_scorers_available(),
    reason="mlflow.genai.scorers.deepeval is not importable",
)


class _Feedback:
    """Stand-in for an MLflow ``Feedback`` object."""

    def __init__(
        self,
        value="yes",
        rationale="the reply addresses the question",
        metadata=None,
        error=None,
    ):
        self.value = value
        self.rationale = rationale
        self.metadata = metadata if metadata is not None else {"score": 0.9, "threshold": 0.5}
        self.error = error


class _StubScorer:
    """Records how it was called and returns a canned result."""

    def __init__(self, feedback=None, raises=None):
        self.calls: list[dict] = []
        self._feedback = feedback if feedback is not None else _Feedback()
        self._raises = raises

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return self._feedback


def _context(**overrides) -> EvaluationContext:
    values = {
        "agent_name": "intake",
        "input": "I have had a headache since yesterday.",
        "output": "How severe is your headache on a scale of 0 to 10?",
        "execution_id": "exec-1",
        "session_id": "sess-1",
        "mode": EvaluationMode.REALTIME,
    }
    values.update(overrides)
    return EvaluationContext(**values)


def _metric(name="answer_relevancy", spec=None, **kwargs) -> MLflowDeepEvalMetric:
    return MLflowDeepEvalMetric(
        name,
        spec if spec is not None else MLFLOW_SCORERS[name],
        **kwargs,
    )


class TestAvailability:
    def test_availability_is_a_bool(self):
        assert isinstance(mlflow_scorers_available(), bool)

    def test_registry_names_are_snake_case_and_map_to_mlflow_names(self):
        assert set(MLFLOW_SCORERS) == {
            "answer_relevancy",
            "faithfulness",
            "task_completion",
            "tool_correctness",
            "toxicity",
            "json_correctness",
        }
        assert MLFLOW_SCORERS["answer_relevancy"].scorer_name == "AnswerRelevancy"
        assert MLFLOW_SCORERS["json_correctness"].scorer_name == "JsonCorrectness"

    def test_requirements_are_declared_honestly(self):
        assert MLFLOW_SCORERS["faithfulness"].requires_trace is True
        assert MLFLOW_SCORERS["tool_correctness"].requires_trace is True
        assert MLFLOW_SCORERS["task_completion"].requires_context == ("expected_output",)
        assert MLFLOW_SCORERS["json_correctness"].requires_config == ("expected_schema",)

    def test_real_scorer_can_be_constructed_without_a_network_call(self):
        scorer = build_scorer("AnswerRelevancy", threshold=0.7)

        assert scorer.name == "AnswerRelevancy"


class TestFeedbackConversion:
    def test_successful_feedback_maps_to_a_passing_result(self):
        result = feedback_to_metric_result(
            _Feedback(), name="answer_relevancy", context=_context()
        )

        assert result.status == MetricStatus.OK
        assert result.passed is True
        assert result.score == 0.9
        assert result.threshold == 0.5
        assert result.reason == "the reply addresses the question"
        assert result.metric_name == "answer_relevancy"
        assert result.agent_name == "intake"
        assert result.execution_id == "exec-1"
        assert result.session_id == "sess-1"
        assert result.evaluation_mode == EvaluationMode.REALTIME

    def test_no_rating_maps_to_a_failing_result(self):
        result = feedback_to_metric_result(
            _Feedback(value="no", metadata={"score": 0.2, "threshold": 0.5}),
            name="answer_relevancy",
            context=_context(),
        )

        assert result.passed is False
        assert result.score == 0.2
        assert result.status == MetricStatus.OK

    @pytest.mark.parametrize("value", [True, "yes", "YES", "true", "pass"])
    def test_various_pass_values_are_accepted(self, value):
        result = feedback_to_metric_result(
            _Feedback(value=value), name="m", context=_context()
        )
        assert result.passed is True

    @pytest.mark.parametrize("value", [False, "no", "NO", "fail", None, ""])
    def test_various_fail_values_are_accepted(self, value):
        result = feedback_to_metric_result(
            _Feedback(value=value), name="m", context=_context()
        )
        assert result.passed is False

    def test_stringified_metadata_is_parsed(self):
        result = feedback_to_metric_result(
            _Feedback(metadata={"score": "0.75", "threshold": "0.8"}),
            name="m",
            context=_context(),
        )

        assert result.score == 0.75
        assert result.threshold == 0.8

    def test_missing_metadata_falls_back_safely(self):
        result = feedback_to_metric_result(
            _Feedback(metadata={}), name="m", context=_context(), threshold=0.6
        )

        assert result.score == 0.0
        assert result.threshold == 0.6
        assert result.status == MetricStatus.OK

    def test_garbage_metadata_never_raises(self):
        result = feedback_to_metric_result(
            _Feedback(metadata={"score": "not-a-number"}),
            name="m",
            context=_context(),
        )

        assert result.score == 0.0

    def test_feedback_error_becomes_an_error_result(self):
        result = feedback_to_metric_result(
            _Feedback(error=RuntimeError("provider unavailable"), metadata={}),
            name="answer_relevancy",
            context=_context(),
        )

        assert result.status == MetricStatus.ERROR
        assert result.passed is False
        assert "RuntimeError" in result.reason
        assert "provider unavailable" in result.reason

    def test_error_result_is_not_counted_as_a_pass(self):
        engine = EvaluationEngine(MetricRegistry(), MetricSelection())
        result = feedback_to_metric_result(
            _Feedback(error=ValueError("boom"), metadata={}),
            name="m",
            context=_context(),
        )

        assert result.status == MetricStatus.ERROR
        assert engine.registry.has("m") is False
        assert result.passed is False

    def test_rationale_is_truncated(self):
        result = feedback_to_metric_result(
            _Feedback(rationale="x" * 5000),
            name="m",
            context=_context(),
            max_reason_chars=100,
        )

        assert len(result.reason) == 103
        assert result.reason.endswith("...")

    def test_no_raw_patient_text_is_stored(self):
        context = _context(output="Patient has stage 4 carcinoma.")
        result = feedback_to_metric_result(
            _Feedback(rationale=""), name="m", context=context
        )

        assert "carcinoma" not in result.reason
        assert "input" not in result.to_dict()
        assert "output" not in result.to_dict()
        assert "carcinoma" not in str(result.to_dict())


class TestMetricAdapter:
    def test_calls_the_scorer_with_inputs_and_outputs(self):
        scorer = _StubScorer()
        metric = _metric(scorer=scorer)

        metric.measure(_context())

        assert scorer.calls == [
            {
                "inputs": "I have had a headache since yesterday.",
                "outputs": "How severe is your headache on a scale of 0 to 10?",
            }
        ]

    def test_expected_output_is_forwarded_as_expectations(self):
        scorer = _StubScorer()
        metric = _metric(scorer=scorer)

        metric.measure(_context(metadata={"expected_output": "asks about severity"}))

        assert scorer.calls[0]["expectations"] == {
            "expected_output": "asks about severity"
        }

    def test_no_expectations_key_when_metadata_is_empty(self):
        scorer = _StubScorer()
        metric = _metric(scorer=scorer)

        metric.measure(_context(metadata={}))

        assert "expectations" not in scorer.calls[0]

    def test_result_carries_the_registry_name_not_the_mlflow_name(self):
        metric = _metric("answer_relevancy", scorer=_StubScorer())

        result = metric.measure(_context())

        assert result.metric_name == "answer_relevancy"
        assert metric.canonical_name == "answer_relevancy"

    def test_trace_only_metrics_are_skipped_with_a_reason(self):
        scorer = _StubScorer()
        metric = _metric("faithfulness", scorer=scorer)

        result = metric.measure(_context(retrieval_context=("doc",)))

        assert result.status == MetricStatus.SKIPPED
        assert "trace" in result.reason
        assert scorer.calls == []

    def test_tool_correctness_is_skipped_without_a_trace(self):
        metric = _metric("tool_correctness", scorer=_StubScorer())

        result = metric.measure(_context(tool_calls=("search",)))

        assert result.status == MetricStatus.SKIPPED
        assert "trace" in result.reason

    def test_metric_needing_context_is_skipped_when_absent(self):
        scorer = _StubScorer()
        metric = _metric("task_completion", scorer=scorer)

        result = metric.measure(_context())

        assert result.status == MetricStatus.SKIPPED
        assert "expected_output" in result.reason
        assert scorer.calls == []

    def test_metric_needing_context_runs_when_present(self):
        scorer = _StubScorer()
        metric = _metric("task_completion", scorer=scorer)

        result = metric.measure(_context(metadata={"expected_output": "books a visit"}))

        assert result.status == MetricStatus.OK
        assert len(scorer.calls) == 1

    def test_metric_needing_configuration_is_skipped(self):
        metric = _metric("json_correctness", scorer=_StubScorer())

        result = metric.measure(_context())

        assert result.status == MetricStatus.SKIPPED
        assert "expected_schema" in result.reason

    def test_metric_needing_configuration_runs_once_supplied(self):
        scorer = _StubScorer()
        metric = _metric(
            "json_correctness",
            scorer=scorer,
            metric_kwargs={"expected_schema": {"type": "object"}},
        )

        assert metric.measure(_context()).status == MetricStatus.OK

    def test_a_raising_scorer_becomes_an_error_not_an_exception(self):
        metric = _metric(scorer=_StubScorer(raises=TimeoutError("judge timed out")))

        result = metric.measure(_context())

        assert result.status == MetricStatus.ERROR
        assert "TimeoutError" in result.reason

    def test_a_scorer_that_cannot_be_built_becomes_an_error(self):
        metric = MLflowDeepEvalMetric("broken", ScorerSpec("NotAMetric"))

        result = metric.measure(_context())

        assert result.status == MetricStatus.ERROR
        assert "scorer unavailable" in result.reason

    def test_threshold_override_is_reported_when_the_judge_omits_one(self):
        metric = _metric(scorer=_StubScorer(_Feedback(metadata={})), threshold=0.75)

        assert metric.measure(_context()).threshold == 0.75


class TestRegistration:
    def test_registers_every_name_without_building_a_scorer(self):
        registry = MetricRegistry()

        registered = register(registry)

        assert set(registered) == set(MLFLOW_SCORERS)
        for name in MLFLOW_SCORERS:
            assert registry.has(name)

    def test_factories_are_lazy_and_build_the_adapter(self):
        registry = MetricRegistry()
        register(registry, specs={"stub": ScorerSpec("AnswerRelevancy")})

        metric = registry.get("stub")

        assert isinstance(metric, MLflowDeepEvalMetric)
        assert metric.name == "stub"

    def test_registration_is_skipped_when_mlflow_is_unavailable(self, monkeypatch):
        import evaluation.adapters.mlflow_scorer.registration as module

        monkeypatch.setattr(module, "mlflow_scorers_available", lambda: False)
        registry = MetricRegistry()

        assert register(registry) == ()
        assert len(registry) == 0

    def test_selected_specs_only(self):
        registry = MetricRegistry()
        register(registry, specs={"only_this": ScorerSpec("Toxicity")})

        assert registry.names() == ("only_this",)

    def test_engine_runs_a_registered_scorer(self):
        registry = MetricRegistry()
        registry.register(_metric(scorer=_StubScorer()))
        engine = EvaluationEngine(
            registry, MetricSelection(common=("answer_relevancy",))
        )

        result = engine.evaluate(_context())

        assert result.passed == 1
        assert result.metrics[0].metric_name == "answer_relevancy"

    def test_engine_reports_skipped_for_unmet_requirements(self):
        registry = MetricRegistry()
        registry.register(_metric("faithfulness", scorer=_StubScorer()))
        engine = EvaluationEngine(registry, MetricSelection(common=("faithfulness",)))

        result = engine.evaluate(_context())

        assert result.metrics[0].status == MetricStatus.SKIPPED
        assert result.total == 0
        assert result.ok is True


class TestPrivacyGate:
    def test_disabled_by_default_blocks_the_external_call(self):
        scorer = _StubScorer()
        metric = _metric(scorer=scorer, external_calls_allowed=False)

        result = metric.measure(_context())

        assert result.status == MetricStatus.SKIPPED
        assert "allow_external_calls" in result.reason
        assert scorer.calls == []

    def test_gate_is_checked_before_other_requirements(self):
        metric = _metric("faithfulness", external_calls_allowed=False)

        assert "allow_external_calls" in metric.measure(_context()).reason

    def test_enabled_gate_allows_the_call(self):
        scorer = _StubScorer()
        metric = _metric(scorer=scorer, external_calls_allowed=True)

        assert metric.measure(_context()).status == MetricStatus.OK

    def test_registration_defaults_to_no_external_calls(self):
        registry = MetricRegistry()
        register(registry, specs={"stub": ScorerSpec("AnswerRelevancy")})

        metric = registry.get("stub")

        assert metric.external_calls_allowed is False
        assert metric.measure(_context()).status == MetricStatus.SKIPPED

    def test_registration_can_opt_in(self):
        registry = MetricRegistry()
        register(
            registry,
            specs={"stub": ScorerSpec("AnswerRelevancy")},
            allow_external_calls=True,
        )

        assert registry.get("stub").external_calls_allowed is True

    def test_default_registry_registers_scorers_but_keeps_them_off(self):
        from evaluation.composition import build_registry as build_default_registry

        registry = build_default_registry()
        metric = registry.get("answer_relevancy")

        assert metric is not None
        assert metric.external_calls_allowed is False
        assert metric.measure(_context()).status == MetricStatus.SKIPPED

    def test_default_registry_can_opt_in(self):
        from evaluation.composition import build_registry as build_default_registry

        registry = build_default_registry(allow_external_calls=True)

        assert registry.get("answer_relevancy").external_calls_allowed is True

    def test_the_gate_does_not_affect_deterministic_metrics(self):
        from evaluation.composition import build_registry as build_default_registry

        registry = build_default_registry()
        engine = EvaluationEngine(
            registry, MetricSelection(common=("intake_completion",))
        )

        result = engine.evaluate(_context(output="Thanks, your intake is complete."))

        assert result.metrics[0].status == MetricStatus.OK
