"""Execute intake evaluation cases against the real intake agent.

The adapter is the only place that knows how the intake agent is invoked. It
reuses the production path — ``AgentFactory``, ``Session``, and the agentic loop
— and normalizes the resulting ``AgentEvent`` stream into a plain observation
that the deterministic validators and the DeepEval metrics can read.

Nothing here re-implements execution: the agent is built and run exactly as the
API does, and only the output normalization is new.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from agent.events import AgentEventType, AgentType
from agent.session import Session
from config.config import Config
from customagents.factory import AgentFactory
from evaluation.dataset import EvalCase
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class IntakeObservation:
    """Normalized result of running one evaluation case.

    Attributes:
        case_id: The case this observation belongs to.
        answers: Reply to each turn, in order. Empty string when the agent
            produced no text (for example an empty message that only greets).
        greeting: The one-time greeting emitted on a fresh session, if any.
        ended: Whether the agent emitted ``status_end`` (intake complete).
        error: Error text if the run failed, else None.
        turns: Number of turns executed.
        history_turns: How many conversation turns were replayed from the case
            context before the evaluated turn.
        event_count: Total ``AgentEvent`` objects produced.
        tool_calls: Distinct tool names the agent invoked.
        duration_ms: Wall-clock duration of the case.
    """

    case_id: str
    answers: tuple[str, ...]
    greeting: str | None
    ended: bool
    error: str | None
    turns: int
    history_turns: int
    event_count: int
    tool_calls: tuple[str, ...]
    duration_ms: float

    @property
    def answer(self) -> str:
        """The reply to the final turn."""
        return self.answers[-1] if self.answers else ""

    @property
    def full_text(self) -> str:
        """Greeting plus every reply, for cases that inspect the whole run."""
        parts = [self.greeting or "", *self.answers]
        return " ".join(part for part in parts if part).strip()

    @property
    def ok(self) -> bool:
        """Whether the case ran without raising an agent error."""
        return self.error is None


class IntakeAgentAdapter:
    """Run intake cases through the existing agent.

    Args:
        config: Runtime configuration, as the API would use it.
        session_factory: Optional factory returning a ``Session`` for one case.
            Defaults to a fresh ``Session(config)`` per case, so cases cannot
            leak history into each other. Tests inject a factory whose LLM
            client is stubbed.
    """

    def __init__(
        self,
        config: Config,
        *,
        session_factory: Callable[[Config], Session] | None = None,
    ) -> None:
        self._config = config
        self._session_factory = session_factory or Session

    async def run_case(self, case: EvalCase) -> IntakeObservation:
        """Run one case and return its normalized observation.

        A case failure is recorded on the observation rather than raised, so one
        bad case cannot abort a dataset run.

        Args:
            case: The case to execute.

        Returns:
            The normalized observation.
        """
        started = time.monotonic()

        session = self._session_factory(self._config)
        greeting: str | None = None
        answers: list[str] = []
        tool_calls: set[str] = set()
        event_count = 0
        ended = False
        error: str | None = None
        history_turns = 0

        try:
            agent = AgentFactory.create(AgentType.INTAKE, self._config, session)

            # Replay the conversation the case assumes before the evaluated
            # turn, so the agent answers from the same state the case describes.
            history = case.conversation
            if history:
                if not session.context_manager:
                    await session.initialize()
                for role, text in history:
                    if role == "assistant":
                        session.context_manager.add_assistant_message(text)
                    else:
                        session.context_manager.add_user_message(text)
                    history_turns += 1
                # A replayed conversation means the greeting already happened.
                session.metadata["greeted"] = True

            for prompt in case.prompts:
                # Read the flag before the turn: the greeting is emitted once
                # per session, on the first turn only.
                greeted = bool(session.metadata.get("greeted"))
                deltas: list[str] = []
                final_text: str | None = None

                async for event in agent.run(prompt):
                    event_count += 1
                    kind = event.type

                    if kind == AgentEventType.TEXT_DELTA:
                        deltas.append(str(event.data.get("content", "")))
                    elif kind == AgentEventType.TEXT_COMPLETE:
                        content = str(event.data.get("content", ""))
                        if not greeted and greeting is None:
                            greeting = content
                            greeted = True
                        else:
                            final_text = content
                    elif kind == AgentEventType.STATUS_END:
                        ended = True
                    elif kind == AgentEventType.AGENT_ERROR:
                        error = str(event.data.get("error", "unknown error"))
                    elif kind == AgentEventType.TOOL_CALL_START:
                        name = event.data.get("name")
                        if name:
                            tool_calls.add(str(name))

                answers.append("".join(deltas).strip() or (final_text or "").strip())

        except Exception as exc:  # noqa: BLE001 - never abort the dataset run
            logger.exception("Evaluation case %s failed", case.id)
            error = str(exc) or exc.__class__.__name__
            if not answers:
                answers.append("")
        finally:
            await self._close(session)

        duration_ms = round((time.monotonic() - started) * 1000, 1)

        observation = IntakeObservation(
            case_id=case.id,
            answers=tuple(answers),
            greeting=greeting,
            ended=ended,
            error=error,
            turns=len(case.prompts),
            history_turns=history_turns,
            event_count=event_count,
            tool_calls=tuple(sorted(tool_calls)),
            duration_ms=duration_ms,
        )

        logger.debug(
            "Case executed",
            extra={
                "fields": {
                    "case": case.id,
                    "turns": observation.turns,
                    "history_turns": history_turns,
                    "ended": ended,
                    "ok": observation.ok,
                    "duration_ms": duration_ms,
                }
            },
        )

        return observation

    @staticmethod
    async def _close(session: Session) -> None:
        """Release the session's LLM client and MCP connections.

        Mirrors ``Agent.__aexit__`` so a dataset run does not leak sockets.
        """
        try:
            if session.client:
                await session.client.close()
        except Exception as exc:  # noqa: BLE001 - cleanup must not raise
            logger.debug("Failed to close LLM client: %s", exc)

        try:
            mcp_manager = getattr(session, "mcp_manager", None)
            if mcp_manager:
                await mcp_manager.shutdown()
        except Exception as exc:  # noqa: BLE001 - cleanup must not raise
            logger.debug("Failed to shut down MCP manager: %s", exc)
