"""Run the offline intake evaluation.

Usage (from the project root):

    uv run python test/runnabletest/evaluation/evalintake.py --all
    uv run python test/runnabletest/evaluation/evalintake.py --case INTAKE_002
    uv run python test/runnabletest/evaluation/evalintake.py --all --report evaluation_report.json
    uv run python test/runnabletest/evaluation/evalintake.py --list

Offline only: it drives the real ``IntakeAgent`` with the dataset in
``evaluation/intake_dataset.json`` and scores the replies.

- Deterministic checks (``critical_checks``) need no LLM and always run.
- DeepEval metrics (Answer Relevancy, Task Completion) run only when
  ``deepeval`` is installed — ``uv pip install deepeval``. It is kept out of the
  project lockfile on purpose: it pins ``click<8.4`` and pulls pytest into the
  runtime environment.
- MLflow logging happens only when ``MLFLOW_ENABLED`` is true.

Talking to the model needs a configured provider in ``.env`` (``API_KEY`` etc.).

This lives under ``test/runnabletest/`` because it is a manual entry point, not a
pytest module: ``evalintake.py`` does not match ``test_*.py``, so the suite never
collects it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# .../test/runnabletest/evaluation/evalintake.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_config  # noqa: E402
from evaluation.dataset import DatasetError, load_dataset  # noqa: E402
from evaluation.runner import EvaluationRunner  # noqa: E402
from utils.logging_config import setup_logging  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline intake agent evaluation.",
    )
    parser.add_argument(
        "--case",
        nargs="?",
        const="",
        metavar="ID",
        help=(
            "Evaluate a single case, e.g. INTAKE_002. "
            "Run with --case but no id to list the available ids."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every case in the dataset.",
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        default="evaluation_report.json",
        help="Where to write the JSON report (default: evaluation_report.json).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the dataset cases and exit.",
    )
    parser.add_argument(
        "--no-llm-metrics",
        action="store_true",
        help="Skip the DeepEval metrics; run only the deterministic checks.",
    )
    parser.add_argument(
        "--judge-model",
        metavar="MODEL",
        default="clinical_model",
        help=(
            "Logical model the DeepEval judge uses (default: clinical_model, "
            "i.e. OpenAI). The judge does not use the fallback chain."
        ),
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Show each case step by step: context, agent, answer, checks, metrics.",
    )
    return parser


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _print_trace(case, result) -> None:
    """Print one case's pipeline, stage by stage."""
    print(f"\n── {result.case_id}  {result.scenario}")

    print(f"   1. context   : {result.history_turns} turn(s) replayed")
    for line in case.context:
        print(f"        │ {_short(line)}")

    print(f"   2. input     : {_short(case.input)}")
    print(
        f"   3. agent     : {result.duration_ms:.0f}ms · "
        f"{result.event_count} events · "
        f"tools: {', '.join(result.tool_calls) or 'none'}"
    )

    if result.greeting:
        print(f"   4. greeting  : {_short(result.greeting)}")
    print(f"   5. answer    : {_short(result.answer) or '(no answer)'}")

    print(f"   6. checks    :", end="")
    for check in result.checks:
        mark = "✓" if check["passed"] else "✗"
        print(f"\n        {mark} {check['name']:<24} {_short(check['detail'], 70)}", end="")
    print()

    if result.skipped_checks:
        print(f"   7. skipped   : {', '.join(result.skipped_checks)}")

    if result.llm_scores or result.llm_reasons:
        print(f"   8. deepeval  :", end="")
        for name, score in result.llm_scores.items():
            print(f"\n        {score:>5.2f}  {name}", end="")
            reason = result.llm_reasons.get(name)
            if reason:
                print(f"\n               {_short(reason, 100)}", end="")
        for name, reason in result.llm_reasons.items():
            if name not in result.llm_scores:
                print(f"\n        ✗     {name} — {_short(reason, 90)}", end="")
        print()

    print(f"   9. verdict   : {'PASS' if result.fully_passed else 'FAIL'}"
          f"  ({result.checks_passed}/{result.checks_total} checks)")


def _print_compact(result) -> None:
    status = "PASS" if result.fully_passed else "FAIL"
    scores = " ".join(f"{name.replace('Metric', '')}={score:.2f}"
                      for name, score in result.llm_scores.items())
    print(f"{status}  {result.case_id:12s} {result.checks_passed}/{result.checks_total} "
          f"checks  {result.duration_ms:7.0f}ms  {scores}  {result.answer[:50]!r}")
    for check in result.checks:
        if not check["passed"]:
            print(f"        ✗ {check['name']}: {check['detail']}")
    if result.error:
        print(f"        ✗ error: {result.error}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    setup_logging()

    try:
        dataset = load_dataset()
    except DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 2

    if args.list or args.case == "":
        for case in dataset.test_cases:
            print(f"{case.id:12s} {case.scenario}")
        print(f"\n{len(dataset.test_cases)} cases · {len(dataset.critical_checks())} checks")
        if args.case == "":
            print("\nPick one with: --case <ID>")
        return 0

    if not args.all and not args.case:
        _parser().print_help()
        return 2

    runner = EvaluationRunner(
        load_config(cwd=PROJECT_ROOT),
        llm_metrics=not args.no_llm_metrics,
        judge_model=args.judge_model,
    )

    if args.case:
        try:
            report = runner.run_case(args.case, dataset)
        except DatasetError as exc:
            print(f"{exc}", file=sys.stderr)
            return 2
    else:
        report = runner.run_all(dataset)

    for case_result in report.cases:
        if args.trace:
            _print_trace(dataset.by_id(case_result.case_id), case_result)
        else:
            _print_compact(case_result)

    summary = report.summary
    print(
        f"\ncases {summary['cases_passed']}/{summary['cases']} passed · "
        f"checks {summary['checks_passed']}/{summary['checks_total']} "
        f"({summary['check_score']:.0%}) · {summary['duration_ms']:.0f}ms"
    )

    if report.llm_metrics.get("status") != "ok":
        print(f"llm metrics: {report.llm_metrics['status']} — {report.llm_metrics.get('reason', '')}")
    if summary["skipped_checks"]:
        print(f"skipped checks: {', '.join(summary['skipped_checks'])}")

    path = EvaluationRunner.save(report, args.report)
    print(f"report: {path}")

    return 0 if summary["cases_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
