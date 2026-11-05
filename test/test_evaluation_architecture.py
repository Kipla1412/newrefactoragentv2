"""Architecture tests: the layering is enforced, not just documented.

The restructure is only worth anything if it cannot quietly rot. These tests parse
the source of every evaluation module and assert the dependency direction, so a
future import that points the wrong way fails here rather than being discovered in
a review.

Also covers the configuration -> selection bridging (the Specification pattern).
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.composition import build_engine, build_registry
from evaluation.core.models import EvaluationMode
from evaluation.core.registry import MetricRegistry
from evaluation.core.selection import MetricSelection

EVALUATION_DIR = Path(__file__).resolve().parents[1] / "evaluation"

#: Which evaluation sub-packages each layer may import.
ALLOWED_IMPORTS: dict[str, set[str]] = {
    "root": {"core"},
    "core": {"core"},
    "ports": {"core", "ports"},
    "datasets": {"core", "datasets"},
    "reporting": {"core", "reporting"},
    "agents": {"core", "agents", "adapters"},
    "adapters": {"core", "adapters", "ports", "datasets", "reporting"},
    "runtime": {"core", "ports", "adapters", "datasets", "reporting", "runtime"},
    "composition": {"core", "ports", "adapters", "agents", "datasets", "reporting", "runtime", "composition"},
}


def _evaluation_modules() -> list[Path]:
    return sorted(
        path
        for path in EVALUATION_DIR.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _imported_evaluation_subpackages(path: Path) -> set[str]:
    """Return the ``evaluation.<x>`` sub-packages a module imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))

    found: set[str] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules = [node.module]

        for module in modules:
            parts = module.split(".")
            if len(parts) >= 2 and parts[0] == "evaluation":
                found.add(parts[1])

    return found


def _layer(path: Path) -> str:
    """Which architectural layer a module belongs to."""
    relative = path.relative_to(EVALUATION_DIR)
    if len(relative.parts) == 1:
        # ``composition.py`` sits at the package root but is its own layer: it is
        # the one module allowed to see every other one.
        return "composition" if relative.name == "composition.py" else "root"
    return relative.parts[0]


class TestLayering:
    @pytest.mark.parametrize("path", _evaluation_modules(), ids=lambda p: p.name)
    def test_module_only_imports_permitted_layers(self, path: Path):
        layer = _layer(path)
        allowed = ALLOWED_IMPORTS[layer]
        imported = _imported_evaluation_subpackages(path)
        violations = imported - allowed

        assert not violations, (
            f"{path.relative_to(EVALUATION_DIR)} (layer '{layer}') imports "
            f"{sorted(violations)}; allowed: {sorted(allowed)}"
        )

    def test_core_depends_on_nothing_but_core(self):
        """The load-bearing rule: core is the innermost layer."""
        for path in (EVALUATION_DIR / "core").rglob("*.py"):
            imported = _imported_evaluation_subpackages(path)
            assert imported <= {"core"}, f"{path.name} reaches outside core: {imported}"

    def test_core_does_not_import_vendors(self):
        """Vendors belong in adapters; core must stay replaceable."""
        forbidden = {"mlflow", "deepeval", "litellm", "fastapi"}

        for path in (EVALUATION_DIR / "core").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module.split(".")[0]]
                else:
                    continue
                assert not (set(names) & forbidden), f"{path.name} imports {names}"

    def test_ports_import_only_core(self):
        for path in (EVALUATION_DIR / "ports").rglob("*.py"):
            imported = _imported_evaluation_subpackages(path)
            assert imported <= {"core", "ports"}, f"{path.name}: {imported}"

    def test_reporting_imports_only_core(self):
        for path in (EVALUATION_DIR / "reporting").rglob("*.py"):
            imported = _imported_evaluation_subpackages(path)
            assert imported <= {"core", "reporting"}, f"{path.name}: {imported}"

    def test_only_composition_knows_every_layer(self):
        """The composition root is the single module allowed to see it all."""
        reach_everything = {"core", "adapters", "agents", "runtime"}

        for path in _evaluation_modules():
            if _layer(path) in {"composition", "root"}:
                continue
            imported = _imported_evaluation_subpackages(path)
            # runtime and agents legitimately touch two broad layers; nobody else
            # should span four.
            assert not (reach_everything <= imported), (
                f"{path.relative_to(EVALUATION_DIR)} spans every layer"
            )


class TestSelectionFromConfig:
    @staticmethod
    def _config(
        *,
        enabled: bool = True,
        common: tuple[str, ...] = (),
        metrics: tuple[str, ...] = (),
        llm_metrics: tuple[str, ...] = (),
        allow_external: bool = False,
        model_uri: str | None = None,
    ):
        return SimpleNamespace(
            evaluation=SimpleNamespace(
                enabled=enabled,
                common_metrics=list(common),
                agents={
                    "intake": SimpleNamespace(
                        metrics=list(metrics), llm_metrics=list(llm_metrics)
                    )
                },
                llm=SimpleNamespace(
                    allow_external_calls=allow_external, model_uri=model_uri
                ),
            )
        )

    def test_agent_metrics_and_llm_metrics_are_combined(self):
        selection = MetricSelection.from_config(
            self._config(
                common=("toxicity",),
                metrics=("intake_completion",),
                llm_metrics=("answer_relevancy",),
            )
        )

        assert selection.for_agent("intake") == (
            "toxicity",
            "intake_completion",
            "answer_relevancy",
        )

    def test_disabled_configuration_selects_nothing(self):
        selection = MetricSelection.from_config(
            self._config(enabled=False, metrics=("intake_completion",))
        )

        assert selection.for_agent("intake") == ()
        assert selection.is_empty()

    def test_missing_configuration_selects_nothing(self):
        assert MetricSelection.from_config(SimpleNamespace()).for_agent("intake") == ()
        assert MetricSelection.from_config(None).for_agent("intake") == ()

    def test_llm_metrics_are_selected_even_when_disabled(self):
        """They must still appear so the report can say why they did not run."""
        selection = MetricSelection.from_config(
            self._config(llm_metrics=("toxicity",), allow_external=False)
        )

        assert "toxicity" in selection.for_agent("intake")

    def test_privacy_gate_defaults_to_closed(self):
        assert MetricSelection.external_calls_allowed(SimpleNamespace()) is False
        assert MetricSelection.external_calls_allowed(None) is False
        assert MetricSelection.external_calls_allowed(self._config()) is False

    def test_privacy_gate_honours_configuration(self):
        assert (
            MetricSelection.external_calls_allowed(self._config(allow_external=True))
            is True
        )

    def test_judge_model_uri_defaults_to_none(self):
        assert MetricSelection.judge_model_uri(self._config()) is None
        assert MetricSelection.judge_model_uri(SimpleNamespace()) is None

    def test_judge_model_uri_is_read(self):
        config = self._config(model_uri="openai:/gpt-4.1-mini")
        assert MetricSelection.judge_model_uri(config) == "openai:/gpt-4.1-mini"


class TestCompositionRoot:
    def test_build_registry_is_closed_by_default(self):
        registry = build_registry()

        assert registry.has("intake_completion")
        assert registry.has("answer_relevancy")
        assert registry.get("answer_relevancy").external_calls_allowed is False

    def test_build_registry_can_open_the_gate(self):
        registry = build_registry(allow_external_calls=True)

        assert registry.get("answer_relevancy").external_calls_allowed is True

    def test_build_registry_reads_the_privacy_gate_from_config(self):
        config = SimpleNamespace(
            evaluation=SimpleNamespace(
                enabled=True,
                common_metrics=[],
                agents={},
                llm=SimpleNamespace(allow_external_calls=True, model_uri=None),
            )
        )

        assert build_registry(config).get("answer_relevancy").external_calls_allowed

    def test_build_engine_wires_registry_and_selection(self):
        config = SimpleNamespace(
            evaluation=SimpleNamespace(
                enabled=True,
                common_metrics=[],
                agents={
                    "intake": SimpleNamespace(
                        metrics=["intake_completion"], llm_metrics=[]
                    )
                },
                llm=SimpleNamespace(allow_external_calls=False, model_uri=None),
            )
        )

        engine = build_engine(config)

        assert engine.metric_names("intake") == ("intake_completion",)
        result = engine.evaluate(
            _context(output="Thanks, your intake is complete.")
        )
        assert result.metrics[0].passed is True

    def test_build_engine_accepts_overrides(self):
        registry = MetricRegistry()
        registry.register(
            SimpleNamespace(name="only", canonical_name="only", measure=lambda c: None)
        )

        engine = build_engine(
            None, registry=registry, selection=MetricSelection(common=("only",))
        )

        assert engine.metric_names("intake") == ("only",)


def _context(**overrides):
    from evaluation.core.models import EvaluationContext

    values = {
        "agent_name": "intake",
        "input": "I am Maya.",
        "output": "Thanks, your intake is complete.",
        "execution_id": "exec-1",
        "mode": EvaluationMode.OFFLINE,
    }
    values.update(overrides)
    return EvaluationContext(**values)


class TestShippedConfiguration:
    """The real ``agents.yaml`` block must load and mean what it says."""

    @staticmethod
    def _config():
        from config.loader import load_config

        return load_config(EVALUATION_DIR.parent)

    def test_evaluation_section_loads(self):
        config = self._config()

        assert config.evaluation.enabled is True
        assert config.evaluation.offline.enabled is True

    def test_external_calls_are_off_in_the_shipped_config(self):
        config = self._config()

        assert MetricSelection.external_calls_allowed(config) is False

    def test_intake_has_per_turn_deterministic_metrics(self):
        engine = build_engine(self._config())

        assert "medical_boundary_compliance" in engine.metric_names("intake")
        assert "no_repeated_questions" in engine.metric_names("intake")

    def test_intake_default_metrics_are_per_turn_only(self):
        """End-of-intake outcomes must not run on mid-intake turns.

        ``intake_completion`` fails every mid-intake reply by construction
        ("no completion acknowledgement"), so running it on a normal turn marks
        the agent wrong for not ending the conversation early. It belongs in
        ``end_of_intake_metrics``, selected per case through the dataset's
        ``critical_checks``.
        """
        config = self._config()
        engine = build_engine(config)

        default_names = engine.metric_names("intake")
        for name in ("intake_completion", "emergency_compliance"):
            assert name not in default_names, (
                f"'{name}' is an end-of-intake outcome and must not run per-turn"
            )

        configured = config.evaluation.agents["intake"]
        assert sorted(configured.end_of_intake_metrics) == [
            "emergency_compliance",
            "intake_completion",
        ]

    def test_intake_has_no_llm_metrics(self):
        """Answer Relevancy under-scores clarifying questions; it stays off."""
        engine = build_engine(self._config())

        for name in engine.metric_names("intake"):
            metric = engine.registry.get(name)
            assert getattr(metric, "mode", "") != "llm", name

    def test_pdf_chat_declares_a_trace_only_metric(self):
        engine = build_engine(self._config())

        assert "faithfulness" in engine.metric_names("pdf_chat")
