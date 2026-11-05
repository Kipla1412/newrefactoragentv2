"""Standalone Answer Relevancy demo.

Why this does not fail like a bare ``AnswerRelevancyMetric()``:

DeepEval builds its own OpenAI client from ``OPENAI_API_KEY``. This project
routes model access through ``LLMClient`` using ``API_KEY``, so the two keys can
differ — and here they do (the ``OPENAI_API_KEY`` has no credits). Passing
``model=build_judge(...)`` makes DeepEval call the framework's client instead,
on the same provider and credentials as the agents.

Run from anywhere:

    uv run python test/runnabletest/evaluation/answerrelevance.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from client.llm_client import LLMClient
from config.loader import load_config
from registries.model_registry import ModelRegistry

# Anchored to this file, not the working directory: agents.yaml and .env live at
# the project root, and a demo script should not depend on where it is run from.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

JUDGE_MODEL = "clinical_model"  # logical name, resolved by the ModelRegistry

QUESTION = "I have had a headache since yesterday."
ANSWER = (
    "Thank you for sharing that. "
    "How severe is your headache on a scale of 0 to 10?"
)


def main() -> int:
    try:
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
    except ImportError:
        print("deepeval is not installed: uv pip install deepeval", file=sys.stderr)
        return 2

    from evaluation.adapters.deepeval_judge import build_judge

    config = load_config(PROJECT_ROOT)

    if JUDGE_MODEL not in config.models.definitions:
        print(
            f"Model '{JUDGE_MODEL}' is not configured in agents.yaml.",
            file=sys.stderr,
        )
        return 2

    client = LLMClient(
        config,
        model_registry=ModelRegistry(config.models.definitions),
    )

    try:
        judge = build_judge(client, model=JUDGE_MODEL)
        if judge is None:
            print("Could not build the DeepEval judge.", file=sys.stderr)
            return 2

        metric = AnswerRelevancyMetric(
            model=judge,
            threshold=0.8,
            include_reason=True,
        )
        test_case = LLMTestCase(input=QUESTION, actual_output=ANSWER)

        metric.measure(test_case)

        print(f"input   : {QUESTION}")
        print(f"output  : {ANSWER}")
        print(f"score   : {metric.score}")
        print(f"threshold: {metric.threshold}")
        print(f"reason  : {metric.reason}")
        return 0
    finally:
        asyncio.run(client.close())


if __name__ == "__main__":
    raise SystemExit(main())
