"""Tests for the shared evaluation registry and engine."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from evaluation.agents import intake as intake_metrics
from evaluation.agents.intake.rules import CHECKS, SUPPORTED_CHECKS
from evaluation.composition import build_registry as build_default_registry
from evaluation.core.engine import EvaluationEngine, MetricSelection, dedupe_metrics
from evaluation.core.models import (
    EvaluationContext,
    EvaluationMode,
    EvaluationResult,
    MetricResult,
    MetricStatus,
)
from evaluation.core.registry import (
    DeterministicMetric,
    MetricRegistry,
)


def _context(**overrides) -> EvaluationContext:
    values = {
        "agent_name": "intake",
        "input": "I have had a headache since yesterday.",
        "output": "Thank you for sharing that. Is the pain constant or throbbing?",
        "execution_id": "exec-1",
        "session_id": "sess-1",
        "mode": EvaluationMode.OFFLINE,
    }
    values.update(overrides)
    return EvaluationContext(**values)


class _AlwaysPasses:
    name = "always_passes"

    def measure(self, context: EvaluationContext) -> MetricResult:
        return MetricResult(
            metric_name=self.name,
            agent_name=context.agent_name,
            execution_id=context.execution_id,
            session_id=context.session_id,
            score=1.0,
            threshold=1.0,
            passed=True,
            reason="fine",
            status=MetricStatus.OK,
            evaluation_mode=context.mode,
            timestamp="2026-01-01T00:00:00+00:00",
        )


class _Boom:
    name = "boom"

    def measure(self, context: EvaluationContext) -> MetricResult:
        raise RuntimeError("provider unavailable")


class TestRegistry:
    def test_register_and_resolve(self):
        registry = MetricRegistry()
        registry.register(_AlwaysPasses())

        assert registry.has("always_passes")
        assert registry.names() == ("always_passes",)
        assert len(registry) == 1

        metrics, unknown = registry.resolve(("always_passes",))
        assert [m.name for m in metrics] == ["always_passes"]
        assert unknown == []

    def test_unknown_names_are_reported_not_raised(self):
        registry = MetricRegistry()
        metrics, unknown = registry.resolve(("nope",))

        assert metrics == []
        assert unknown == ["nope"]
        assert registry.get("nope") is None

    def test_factories_are_built_lazily(self):
        calls: list[int] = []
        registry = MetricRegistry()
        registry.register_factory(
            "lazy", lambda: (calls.append(1), _AlwaysPasses())[1]
        )

        assert calls == []
        assert registry.has("lazy")

        metrics, _ = registry.resolve(("lazy",))
        assert calls == [1]
        assert metrics[0].name == "always_passes"

    def test_broken_factory_degrades_to_unknown(self):
        def boom():
            raise ValueError("no judge available")

        registry = MetricRegistry()
        registry.register_factory("llm_metric", boom)

        metrics, unknown = registry.resolve(("llm_metric",))
        assert metrics == []
        assert unknown == ["llm_metric"]

    def test_register_replaces_and_clears_factory(self):
        registry = MetricRegistry()
        registry.register_factory("m", lambda: _AlwaysPasses())
        registry.register(DeterministicMetric("m", lambda c, o: (True, "ok")))

        assert len(registry) == 1
        metric = registry.get("m")
        assert isinstance(metric, DeterministicMetric)
        assert metric.name == "m"


class TestDeterministicMetric:
    def test_wraps_existing_check_without_reimplementing_it(self):
        check = CHECKS["completes_intake"]
        context = _context(output="Thanks, your intake is complete.")

        expected_passed, _ = check(context, context)
        result = DeterministicMetric("completes_intake", check).measure(context)

        assert result.passed is bool(expected_passed)
        assert result.status == MetricStatus.OK
        assert result.score == (1.0 if expected_passed else 0.0)

    def test_context_exposes_the_adapter_shape(self):
        context = _context(context=("Patient: hello", "Assistant: hi"))

        assert context.answer == context.output
        assert context.context == ("Patient: hello", "Assistant: hi")
        assert context.input

    def test_aliases_reuse_the_same_implementation(self):
        context = _context(output="Thanks, your intake is complete.")
        registry = build_default_registry()

        clinical = registry.get("intake_completion")
        raw = registry.get("completes_intake")

        assert isinstance(clinical, DeterministicMetric)
        assert clinical._check is raw._check

    def test_description_falls_back_to_docstring(self):
        check = CHECKS["no_repeated_questions"]
        metric = DeterministicMetric("doc_metric", check)

        assert metric.description == check.__doc__.strip().split("\n")[0]

    def test_docstringless_check_gives_empty_description(self):
        metric = DeterministicMetric("m", lambda c, o: (True, "ok"))
        assert metric.description == ""

    def test_explicit_description_wins(self):
        metric = DeterministicMetric(
            "m", CHECKS["no_repeated_questions"], description="clinical rule"
        )
        assert metric.description == "clinical rule"


class TestMetricSelection:
    def test_common_plus_agent_specific(self):
        selection = MetricSelection(
            common=("answer_relevancy",),
            per_agent={"intake": ("intake_completion",)},
        )

        assert selection.for_agent("intake") == (
            "answer_relevancy",
            "intake_completion",
        )
        assert selection.for_agent("pdf_chat") == ("answer_relevancy",)

    def test_duplicates_are_collapsed_preserving_order(self):
        selection = MetricSelection(
            common=("a", "b"),
            per_agent={"intake": ("b", "c")},
        )

        assert selection.for_agent("intake") == ("a", "b", "c")

    def test_agents_lists_configured_agents(self):
        selection = MetricSelection(per_agent={"intake": (), "pdf_chat": ()})
        assert selection.agents() == ("intake", "pdf_chat")


class TestEngine:
    def test_unconfigured_engine_evaluates_nothing(self):
        engine = EvaluationEngine(MetricRegistry())
        result = engine.evaluate(_context())

        assert result.metrics == ()
        assert result.total == 0
        assert result.ok is True

    def test_selects_metrics_per_agent(self):
        registry = MetricRegistry()
        registry.register(DeterministicMetric("passes", lambda c, o: (True, "ok")))
        engine = EvaluationEngine(
            registry,
            MetricSelection(common=("passes",), per_agent={"intake": ("passes",)}),
        )

        assert engine.metric_names("intake") == ("passes",)
        assert engine.metric_names("voice") == ("passes",)
        assert engine.evaluate(_context()).passed == 1

    def test_missing_metric_is_skipped_not_fatal(self):
        engine = EvaluationEngine(
            MetricRegistry(), MetricSelection(common=("retired_metric",))
        )
        result = engine.evaluate(_context())

        assert len(result.metrics) == 1
        assert result.metrics[0].status == MetricStatus.SKIPPED
        assert result.metrics[0].passed is False
        assert "not registered" in result.metrics[0].reason
        assert result.total == 0
        assert result.ok is True

    def test_failure_is_isolated_to_the_failing_metric(self):
        registry = MetricRegistry()
        registry.register(_Boom())
        registry.register(_AlwaysPasses())
        engine = EvaluationEngine(
            registry, MetricSelection(common=("boom", "always_passes"))
        )

        result = engine.evaluate(_context())

        assert [m.status for m in result.metrics] == [
            MetricStatus.ERROR,
            MetricStatus.OK,
        ]
        assert result.errored[0].metric_name == "boom"
        assert "RuntimeError" in result.errored[0].reason
        assert result.failed == ()
        assert result.ok is False

    def test_engine_never_raises_on_metric_failure(self):
        registry = MetricRegistry()
        registry.register(_Boom())
        engine = EvaluationEngine(registry, MetricSelection(common=("boom",)))

        assert engine.evaluate(_context()).errored
        registry = MetricRegistry()
        registry.register(_AlwaysPasses())
        engine = EvaluationEngine(registry, MetricSelection(common=("always_passes",)))

        results = engine.evaluate_many(
            [_context(execution_id="a"), _context(execution_id="b")]
        )

        assert [r.context.execution_id for r in results] == ["a", "b"]

    def test_mode_is_carried_into_results(self):
        registry = MetricRegistry()
        registry.register(_AlwaysPasses())
        engine = EvaluationEngine(registry, MetricSelection(common=("always_passes",)))

        realtime = engine.evaluate(_context(mode=EvaluationMode.REALTIME)).metrics[0]
        offline = engine.evaluate(_context(mode=EvaluationMode.OFFLINE)).metrics[0]

        assert realtime.evaluation_mode == EvaluationMode.REALTIME
        assert offline.evaluation_mode == EvaluationMode.OFFLINE


class TestBuiltinRegistry:
    def test_all_intake_checks_are_registered(self):
        registry = build_default_registry()

        for name in SUPPORTED_CHECKS:
            assert registry.has(name), name

    def test_clinical_aliases_are_registered(self):
        registry = build_default_registry()

        for alias in intake_metrics.ALIASES:
            assert registry.has(alias), alias

    def test_registry_is_independent_per_call(self):
        first = build_default_registry()
        first.register(_AlwaysPasses())

        second = build_default_registry()
        assert not second.has("always_passes")

    def test_default_registry_includes_mlflow_scorers(self):
        registry = build_default_registry()

        for name in ("answer_relevancy", "faithfulness", "task_completion"):
            assert registry.has(name), name


class TestDuplicatePrevention:
    def test_canonical_name_defaults_to_the_metric_name(self):
        metric = DeterministicMetric("m", lambda c, o: (True, "ok"))
        assert metric.canonical_name == "m"

    def test_alias_shares_its_targets_canonical_name(self):
        registry = build_default_registry()

        alias = registry.get("intake_completion")
        target = registry.get("completes_intake")

        assert alias.canonical_name == target.canonical_name == "completes_intake"

    def test_dedupe_metrics_splits_duplicates(self):
        shared = lambda c, o: (True, "ok")  # noqa: E731
        keeper = DeterministicMetric("a", shared)
        duplicate = DeterministicMetric("b", shared, canonical_name="a")

        measured, duplicates = dedupe_metrics([keeper, duplicate])

        assert [m.name for m in measured] == ["a"]
        assert duplicates == [("b", "a")]

    def test_configuring_alias_and_target_runs_the_rule_once(self):
        engine = EvaluationEngine(
            build_default_registry(),
            MetricSelection(common=("intake_completion", "completes_intake")),
        )

        result = engine.evaluate(_context(output="Thanks, your intake is complete."))

        statuses = {m.metric_name: m.status for m in result.metrics}
        assert statuses["intake_completion"] == MetricStatus.OK
        assert statuses["completes_intake"] == MetricStatus.SKIPPED
        assert result.total == 1

    def test_duplicate_is_reported_with_a_reason(self):
        engine = EvaluationEngine(
            build_default_registry(),
            MetricSelection(common=("intake_completion", "completes_intake")),
        )

        duplicated = [
            m for m in engine.evaluate(_context()).metrics if m.status == MetricStatus.SKIPPED
        ]

        assert len(duplicated) == 1
        assert "duplicate" in duplicated[0].reason
        assert "completes_intake" in duplicated[0].reason

    def test_distinct_metrics_are_not_deduplicated(self):
        engine = EvaluationEngine(
            build_default_registry(),
            MetricSelection(common=("intake_completion", "emergency_compliance")),
        )

        assert engine.evaluate(_context()).total == 2


class TestResultModel:
    def test_metric_result_has_the_standardized_fields(self):
        result = _AlwaysPasses().measure(_context())

        assert set(result.to_dict()) == {
            "metric_name",
            "agent_name",
            "execution_id",
            "session_id",
            "score",
            "threshold",
            "passed",
            "reason",
            "status",
            "evaluation_mode",
            "timestamp",
        }

    def test_evaluation_result_is_json_serializable(self):
        import json

        registry = MetricRegistry()
        registry.register(_AlwaysPasses())
        engine = EvaluationEngine(registry, MetricSelection(common=("always_passes",)))
        payload = engine.evaluate(_context()).to_dict()

        assert json.loads(json.dumps(payload))["agent_name"] == "intake"
        assert payload["passed"] == 1
        assert payload["total"] == 1

    def test_no_metrics_ran_means_ok(self):
        assert EvaluationResult(context=_context()).ok is True


class TestEnums:
    def test_status_is_a_str_enum_so_it_survives_every_boundary(self):
        assert isinstance(MetricStatus.OK, str)
        assert MetricStatus.OK == "ok"
        assert MetricStatus("ok") is MetricStatus.OK
        assert f"{MetricStatus.OK.value}" == "ok"

    def test_mode_is_a_str_enum(self):
        assert EvaluationMode.REALTIME == "realtime"
        assert EvaluationMode("offline") is EvaluationMode.OFFLINE

    def test_unknown_values_are_rejected_at_the_boundary(self):
        with pytest.raises(ValueError):
            MetricStatus("bogus")
        with pytest.raises(ValueError):
            EvaluationMode("batch")

    def test_serialization_emits_values_not_member_names(self):
        import json

        payload = _AlwaysPasses().measure(_context()).to_dict()

        assert payload["status"] == "ok"
        assert payload["evaluation_mode"] == "offline"
        assert "MetricStatus" not in json.dumps(payload)
        assert "EvaluationMode" not in json.dumps(payload)

    def test_reason_strings_use_explicit_values(self):
        engine = EvaluationEngine(
            MetricRegistry(), MetricSelection(common=("retired_metric",))
        )
        skipped = engine.evaluate(_context()).metrics[0]

        assert skipped.status is MetricStatus.SKIPPED
        assert "not registered" in skipped.reason
