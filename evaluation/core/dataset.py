"""Dataset domain models — agent-neutral, and part of the core vocabulary.

A dataset is authored as "given this conversation so far, what should the next
reply be?":

.. code-block:: json

    {
      "dataset_name": "intake_agent_offline_evaluation",
      "version": "1.0",
      "agent": "intake",
      "test_cases": [
        {
          "test_case_id": "INTAKE_001",
          "scenario": "Normal intake opening",
          "context": ["Assistant: What's your full name?"],
          "input": "I'm Maya Raman.",
          "expected_output": "Acknowledge and ask one next question.",
          "expected_behaviour": ["Ask only one question"],
          "critical_checks": ["one_question_only", "no_repeated_questions"]
        }
      ]
    }

``context`` carries the turns *before* ``input``. Lines prefixed ``Assistant:`` /
``Agent:`` replay as assistant turns and ``Patient:`` / ``User:`` as user turns,
so a case runner can rebuild the state the case assumes. Lines without a
recognised prefix are prose state descriptions and are skipped — such a case is
reported as not fully replayable rather than being silently mis-seeded.

``critical_checks`` is the metric vocabulary: the names are registry metric names
(see :mod:`evaluation.agents.intake.rules`), so a case and an agent's configuration
speak the same language.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from utils.errors import AgentError

#: Prefix recognised when replaying ``context`` into a session.
_ROLE_PREFIXES: dict[str, str] = {
    "assistant:": "assistant",
    "agent:": "assistant",
    "patient:": "user",
    "user:": "user",
}


class DatasetError(AgentError):
    """Raised when an evaluation dataset cannot be loaded or is invalid."""


class EvalCase(BaseModel):
    """One evaluation case: a conversation context plus the next patient turn.

    Attributes:
        test_case_id: Stable identifier, unique within the dataset.
        scenario: Short human label for what the case probes.
        context: The conversation before ``input``.
        input: The patient's next message — the turn being evaluated.
        expected_output: Natural-language description of a correct reply.
        expected_behaviour: Prose behaviours the reply should exhibit.
        critical_checks: Metric names to run (registry names).
        retrieval_context: Passages the reply should be grounded in, for
            retrieval-backed agents.
        tags: Free-form labels for filtering and reporting.
        metadata: Extra, non-content context (task description, model hints).
    """

    test_case_id: str
    scenario: str
    context: list[str] = Field(default_factory=list)
    input: str
    expected_output: str
    expected_behaviour: list[str] = Field(default_factory=list)
    critical_checks: list[str] = Field(default_factory=list)
    retrieval_context: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "EvalCase":
        if not self.test_case_id.strip():
            raise ValueError("test_case_id must not be empty")
        if not self.scenario.strip():
            raise ValueError(f"{self.test_case_id}: scenario must not be empty")
        if not self.input.strip():
            raise ValueError(f"{self.test_case_id}: input must not be empty")
        if not self.expected_output.strip():
            raise ValueError(f"{self.test_case_id}: expected_output must not be empty")
        if any(not line.strip() for line in self.context):
            raise ValueError(f"{self.test_case_id}: context contains an empty line")
        return self

    @property
    def id(self) -> str:
        """The case identifier."""
        return self.test_case_id

    @property
    def task(self) -> str:
        """Short task label, used as the LLM-judged task description."""
        return self.scenario

    @property
    def prompts(self) -> list[str]:
        """The user turn to send to the agent."""
        return [self.input]

    @property
    def metric_names(self) -> tuple[str, ...]:
        """Metric names to run for this case.

        ``critical_checks`` wins; ``metadata['metrics']`` is the fallback so a
        dataset can name metrics without using the check vocabulary.
        """
        if self.critical_checks:
            return tuple(self.critical_checks)
        return tuple(self.metadata.get("metrics", ()))

    @property
    def conversation(self) -> list[tuple[str, str]]:
        """Parse ``context`` into ``(role, text)`` pairs.

        Only lines with a recognised ``Assistant:`` / ``Patient:`` prefix are
        returned; prose state descriptions are skipped.
        """
        replayable: list[tuple[str, str]] = []

        for line in self.context:
            lowered = line.lower()
            for prefix, role in _ROLE_PREFIXES.items():
                if lowered.startswith(prefix):
                    text = line[len(prefix):].strip()
                    if text:
                        replayable.append((role, text))
                    break

        return replayable

    @property
    def is_replayable(self) -> bool:
        """Whether every non-empty context line can be replayed as a turn."""
        if not self.context:
            return True
        return len(self.conversation) == len(self.context)


class EvalDataset(BaseModel):
    """A named, versioned collection of evaluation cases.

    Attributes:
        dataset_name: Identifier for the dataset.
        version: Author-supplied version string.
        description: What the dataset covers.
        agent: Agent the dataset targets.
        test_cases: The evaluation cases.
        metadata: Extra dataset-level context.
    """

    dataset_name: str = "intake_offline_evaluation"
    version: str = "1.0"
    description: str | None = None
    agent: str = "intake"
    test_cases: list[EvalCase]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "EvalDataset":
        if not self.test_cases:
            raise ValueError("dataset contains no test cases")

        seen: set[str] = set()
        for case in self.test_cases:
            if case.id in seen:
                raise ValueError(f"duplicate test_case_id '{case.id}'")
            seen.add(case.id)

        return self

    @property
    def cases(self) -> list[EvalCase]:
        """Alias for :attr:`test_cases`."""
        return self.test_cases

    def by_id(self, case_id: str) -> EvalCase:
        """Return one case by id.

        Args:
            case_id: The case identifier.

        Returns:
            The matching case.

        Raises:
            DatasetError: If no case has that id.
        """
        for case in self.test_cases:
            if case.id == case_id:
                return case
        raise DatasetError(f"Unknown case '{case_id}'")

    def critical_checks(self) -> set[str]:
        """Return every distinct metric name named by the dataset."""
        return {name for case in self.test_cases for name in case.metric_names}

    def not_replayable(self) -> list[str]:
        """Return ids of cases whose context cannot be fully replayed."""
        return [case.id for case in self.test_cases if not case.is_replayable]
