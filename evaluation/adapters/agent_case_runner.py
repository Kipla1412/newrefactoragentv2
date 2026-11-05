"""Adapter: run a dataset case through the real agent.

This is the only place that knows how an agent is invoked. It reuses the
production path — ``AgentFactory``, ``Session`` and the agentic loop — and
normalizes the resulting ``AgentEvent`` stream into the framework's single
evaluation value object, :class:`~evaluation.core.models.EvaluationContext`.

Nothing here re-implements execution: the agent is built and run exactly as the
API does, and only the output normalization is new. A case that fails records the
failure on the context instead of raising, so one bad case cannot abort a run.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable

from agent.events import AgentEventType, AgentType
from agent.session import Session
from config.config import Config
from customagents.factory import AgentFactory
from evaluation.core.dataset import EvalCase
from evaluation.core.models import EvaluationContext, EvaluationMode
from utils.logging_config import get_logger

logger = get_logger(__name__)


class AgentCaseRunner:
    """Execute evaluation cases against one agent.

    Implements :class:`~evaluation.ports.case_runner.CaseRunner`.

    Args:
        config: Runtime configuration, as the API would use it.
        agent_name: Name recorded on produced contexts; must match the key used
            for metric configuration.
        agent_type: Which agent to build.
        session_factory: Optional factory returning a ``Session`` for one case.
            Defaults to a fresh ``Session(config)`` per case, so cases cannot leak
            history into each other. Tests inject a factory with a stubbed client.
        skill: Optional skill to activate for each case.
    """

    def __init__(
        self,
        config: Config,
        *,
        agent_name: str = "intake",
        agent_type: AgentType = AgentType.INTAKE,
        session_factory: Callable[[Config], Session] | None = None,
        skill: str | None = None,
    ) -> None:
        self.agent_name = agent_name
        self._config = config
        self._agent_type = agent_type
        self._session_factory = session_factory or Session
        self._skill = skill

    async def run_case(self, case: EvalCase) -> EvaluationContext:
        """Run one case and return its evaluation context.

        Args:
            case: The case to execute.

        Returns:
            The normalized context. Failures appear on ``error``, never as an
            exception.
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
            agent = AgentFactory.create(self._agent_type, self._config, session)

            # Replay the conversation the case assumes before the evaluated turn,
            # so the agent answers from the same state the case describes.
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
                # Read the flag before the turn: the greeting is emitted once per
                # session, on the first turn only.
                greeted = bool(session.metadata.get("greeted"))
                deltas: list[str] = []
                final_text: str | None = None

                async for event in agent.run(prompt, **self._run_kwargs()):
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
        answer = answers[-1] if answers else ""

        logger.debug(
            "Case executed",
            extra={
                "fields": {
                    "case": case.id,
                    "turns": len(case.prompts),
                    "history_turns": history_turns,
                    "ended": ended,
                    "ok": error is None,
                    "duration_ms": duration_ms,
                }
            },
        )

        return EvaluationContext(
            agent_name=self.agent_name,
            input=case.input,
            output=answer,
            context=tuple(case.context),
            ended=ended,
            error=error,
            execution_id=case.id,
            retrieval_context=tuple(case.retrieval_context),
            tool_calls=tuple(sorted(tool_calls)),
            duration_ms=duration_ms,
            metadata={
                "greeting": greeting,
                "answer": answer,
                "history_turns": history_turns,
                "event_count": event_count,
                "expected_output": case.expected_output,
                "task": case.task,
                "scenario": case.scenario,
                "tags": tuple(case.tags),
                **case.metadata,
            },
            mode=EvaluationMode.OFFLINE,
        )

    def _run_kwargs(self) -> dict[str, Any]:
        """Extra arguments for ``agent.run`` (skill activation, when configured)."""
        return {"skill": self._skill} if self._skill else {}

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


def default_case_runners() -> dict[str, Iterable[Any]]:
    """Agent name -> ``(agent_type, skill)`` for the agents that have datasets.

    One entry per agent with a golden dataset, so the CLI can resolve a runner
    from an agent name without a lookup chain.
    """
    return {
        "intake": (AgentType.INTAKE, None),
    }
