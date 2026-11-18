from __future__ import annotations
from collections.abc import Iterator
from contextlib import contextmanager
from typing import AsyncGenerator, Awaitable, Callable
from agent.events import AgentEvent, AgentEventType, AgentType
from agent.session import Session
from client.response import StreamEventType, TokenUsage, ToolCall, ToolResultMessage, parse_tool_call_arguments
from config.config import Config
from prompts.system import create_loop_breaker_prompt
from skills import Skill, SkillRegistry, default_skill_registry
from tools.base import ToolConfirmation
from utils.errors import AgentError
from utils.logging_config import get_logger
from utils.mlflow_spans import (
    agent_span,
    finish_llm_span,
    finish_tool_span,
    llm_span,
    tool_span,
)
import json
import time
import uuid
from datetime import datetime
import asyncio
from typing import Any

if False:  # type-checking only for the two names used in annotations
    from evaluation.core.models import EvaluationContext  # noqa: F401
    from evaluation.runtime.observer import EvaluationObserver  # noqa: F401

logger = get_logger(__name__)

class Agent:
    def __init__(
        self,
        config: Config,
        system_prompt: str,
        agent_type: AgentType,
        confirmation_callback: Callable[[ToolConfirmation], bool] | None = None,
        skill_registry: SkillRegistry | None = None,
    ):
        self.config = config
        self.system_prompt = system_prompt
        self.session: Session = Session(self.config)
        self.agent_type = agent_type
        self.session.agent_name = self.__class__.__name__
        # Reusable source of per-request instruction sets (skills). Injected so
        # tests — and alternative skill libraries — can be swapped in.
        self.skill_registry: SkillRegistry = skill_registry or default_skill_registry()
        # Real-time evaluation hook. Set by the evaluation composition root (or
        # the API bootstrap) to an EvaluationObserver. None in tests, CLI and
        # offline evaluation, which means no per-turn evaluation happens.
        # The hook is optional on purpose: evaluation must never change agent
        # behaviour, so an unset hook is a complete no-op.
        # Typed as Any so the base agent does not import the evaluation package;
        # see evaluation/composition.py for the wiring.
        self.evaluation_observer: Any | None = None
        # Override this on subclasses that must use tools on every turn
        # (e.g. document retrieval must search before answering).
        self.force_tool_choice = False
        # Restrict which tools the agent may see. None means all tools.
        self.allowed_tool_names: list[str] | None = None
        # Tools resolved from configuration by the AgentFactory. None means
        # no per-agent tool configuration was applied.
        self.configured_tools: list | None = None
        # Skill resolved from configuration by the AgentFactory. Used when a
        # caller does not name one, so an agent "knows" its own skill without
        # the request having to carry it.
        self.default_skill: str | None = None
        # Human-readable name resolved from configuration by the AgentFactory,
        # used for MLflow tagging and error messages. Only needed by agents
        # that are not distinct classes (see GenericAgent).
        self.configured_agent_name: str | None = None
        
        if confirmation_callback is not None:
            self.session.approval_manager.confirmation_callback = confirmation_callback

    def active_skill(self, skill: str | None) -> str | None:
        """Resolve which skill to activate for a call.

        This is how an agent "knows" its skill: the configured
        :attr:`default_skill` is used unless the caller overrides it.

        Args:
            skill: Skill name from the caller. ``None`` means "use this
                agent's configured skill"; an empty string means "no skill
                for this call", which is how a caller opts out of the default.

        Returns:
            The skill name to activate, or None for no skill.
        """
        if skill is None:
            return self.default_skill
        return skill or None

    @contextmanager
    def use_skill(self, skill: str | None) -> Iterator[Skill | None]:
        """Activate a named skill for the duration of a block.

        The reusable entry point for skills: any agent can wrap part of its
        execution in this block, and the skill's instructions are handed to the
        ContextManager, which appends them to the system message for the
        current request.

        Activation is scoped — whatever was active before is restored on exit,
        including when the block raises — so instructions cannot leak into a
        later request or a concurrent execution.

        Args:
            skill: Skill name from :attr:`skill_registry`, or None for no
                skill.

        Yields:
            The loaded skill, or None when no skill was requested.

        Raises:
            AgentError: If the session context is not initialized yet.
            SkillNotFoundError: If the requested skill is unknown.
            SkillValidationError: If the skill file is malformed.
        """
        if not skill:
            yield None
            return

        if not self.session.context_manager:
            raise AgentError(
                "Session must be initialized before activating a skill "
                "(call session.initialize() first)."
            )

        loaded = self.skill_registry.load(skill)
        logger.debug(
            "Skill activated for this execution",
            extra={
                "fields": {
                    "skill": loaded.name,
                    # A config-driven agent may carry a plain string rather than
                    # an AgentType member.
                    "agent": getattr(self.agent_type, "value", self.agent_type),
                }
            },
        )
        with self.session.context_manager.use_skill(loaded.instructions):
            yield loaded

    async def run(self, message: str):
        """Run one turn, traced as a single MLflow execution.

        The whole turn is wrapped in one root AGENT span, created with the
        execution's ``run_id``. That links the trace to the run, so the agent's
        behaviour (LLM calls, tool calls, tokens) and the evaluation scores for
        this execution are discoverable together in MLflow.
        """
        self.session.start_mlflow_run(message)

        await self.session.hook_system.trigger_before_agent(message)

        async for event in self._run_turn(message):
            yield event

    def _execution_run_id(self) -> str | None:
        """This execution's MLflow run id, when tracking is on."""
        run = getattr(self.session, "mlflow_run", None)
        return getattr(getattr(run, "info", None), "run_id", None)

    async def _run_turn(self, message: str):
        """The turn itself: events in, events out, no run lifecycle."""
        yield AgentEvent.agent_start(message, self.agent_type_for_events())

        self.session.context_manager.set_system_prompt(self.system_prompt)
        
        self.session.context_manager.add_user_message(message)

        started = time.perf_counter()
        final_response: str | None = None

        async for event in self._agentic_loop():
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")

        duration_ms = (time.perf_counter() - started) * 1000

        # Real-time evaluation is scheduled at the end of _agentic_loop, not here:
        # this method is the base-class path only, and every custom agent
        # overrides run(). See Agent._agentic_loop.

        await self.session.hook_system.trigger_after_agent(message, final_response)

        yield AgentEvent.agent_end(final_response, agent=self.agent_type_for_events())

        self.session.end_mlflow_run()

    def agent_type_for_events(self) -> AgentType | str:
        """The identity to tag events with, tolerating config-driven names.

        ``agent_type`` may be a plain string (e.g. for custom agents), so events
        are tagged with the string value rather than requiring an enum member.
        """
        return getattr(self.agent_type, "value", self.agent_type) or self.agent_type

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        """Run the turn's reasoning loop.

        This is the boundary every agent path shares: agents that override
        ``run()``, agents that drain the loop for a buffered reply (``generate``,
        ``recommend_questions``) and the base ``Agent.run`` all come through here.
        Real-time evaluation is therefore scheduled on exit from this method
        rather than from ``run()``, which only the base class uses — putting it in
        ``run()`` silently skipped every custom agent.
        """
        max_turns = self.config.max_turns
        started = time.perf_counter()
        final_response: str | None = None
        self._last_trace_id = None

        # The root AGENT span lives here, not in run(): every custom agent
        # overrides run(), so a root span there would leave their spans
        # fragmented across several traces — and MLflow reads tool calls from the
        # trace, which tool metrics then cannot see.
        with agent_span(
            name=self._evaluation_agent_name,
            run_id=self._execution_run_id(),
            execution_id=self.session.execution_id or "",
            session_id=self.session.session_id,
            agent_name=self._evaluation_agent_name,
        ) as span:
            self._last_trace_id = getattr(span, "trace_id", None)

            try:
                async for event in self._turn_events(max_turns):
                    if event.type == AgentEventType.TEXT_COMPLETE:
                        # Last one wins: a turn can emit several while iterating
                        # through tool calls, and the final reply is the one to judge.
                        final_response = event.data.get("content") or final_response
                    yield event
            finally:
                # Runs on every exit, including the early return when no tools are
                # called and the max-turns path. A GeneratorExit (client
                # disconnect) leaves final_response as None, which is not
                # evaluated.
                self._schedule_real_time_evaluation(
                    self._current_user_input(),
                    final_response,
                    (time.perf_counter() - started) * 1000,
                )

    def _current_user_input(self) -> str:
        """The most recent user message, i.e. the input being evaluated."""
        manager = self.session.context_manager
        if manager is None:
            return ""

        for message in reversed(getattr(manager, "messages", []) or []):
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).lower() == "user":
                content = message.get("content")
                return content if isinstance(content, str) else ""
        return ""

    async def _turn_events(self, max_turns: int) -> AsyncGenerator[AgentEvent, None]:
        """The loop body, split out so the caller can wrap it in try/finally."""
        for turn_num in range(max_turns):
            self.session.increment_turn()
            
            while not self.session.event_queue.empty():

                event = await self.session.event_queue.get()

                data = event["data"]

                logger.debug(
                    "Approval request pending",
                    extra={
                        "fields": {
                            "approval_id": data.get("approval_id"),
                            "tool": data.get("tool_name"),
                        }
                    },
                )

                yield AgentEvent.approval_request(
                    approval_id=data["approval_id"],
                    tool_name=data["tool_name"],
                    description=data["description"],
                    params=data.get("params"),
                    agent=self.agent_type
                )
            
            response_text = ""

            # check for context overflow
            if self.session.context_manager.needs_compression():
                summary, usage = await self.session.chat_compactor.compress(
                    self.session.context_manager
                )

                if summary:
                    self.session.context_manager.replace_with_summary(summary)
                    self.session.context_manager.set_latest_usage(usage)
                    self.session.context_manager.add_usage(usage)

            tool_schemas = self.session.tool_registry.get_schemas()

            if self.allowed_tool_names is not None:
                allowed = set(self.allowed_tool_names)
                tool_schemas = [
                    schema for schema in tool_schemas
                    if schema.get("name") in allowed
                ]

            tool_calls: list[ToolCall] = []
            usage: TokenUsage | None = None

            messages = self.session.context_manager.get_messages()
            # Agent → ModelRouter → ModelRegistry → LLMClient → LiteLLM.
            # Returns None when routing is disabled (configured default model).
            model = self.session.route_model(messages)

            # One span per model call, holding the prompt shape and the token
            # counts. A child of the execution's AGENT span, so the whole
            # reasoning path of a turn is visible as one trace.
            with llm_span(model=model or self.config.model_name, messages=messages) as llm:
                async for event in self.session.client.chat_completion(
                    messages,
                    model=model,
                    tools=tool_schemas if tool_schemas else None,
                    tool_choice=(
                        "required"
                        if self.force_tool_choice and turn_num == 0
                        else None
                    ),
                ):
                    if event.type == StreamEventType.TEXT_DELTA:
                        if event.text_delta:
                            content = event.text_delta.content
                            response_text += content
                            yield AgentEvent.text_delta(content, agent=self.agent_type)
                    elif event.type == StreamEventType.TOOL_CALL_COMPLETE:
                        if event.tool_call:
                            tool_calls.append(event.tool_call)
                    elif event.type == StreamEventType.ERROR:
                        yield AgentEvent.agent_error(
                            event.error or "Unknown error occurred.",
                        )
                    elif event.type == StreamEventType.MESSAGE_COMPLETE:
                        usage = event.usage

                        if usage and self.session.mlflow_tracker:
                            self.session.mlflow_tracker.log_metrics({
                                "prompt_tokens": usage.prompt_tokens,
                                "completion_tokens": usage.completion_tokens,
                                "total_tokens": usage.total_tokens
                            })

                finish_llm_span(
                    llm, usage=usage, response_chars=len(response_text or "")
                )

            self.session.context_manager.add_assistant_message(
                response_text or "",
                (
                    [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in tool_calls
                    ]
                    if tool_calls
                    else None
                ),
            )
            
            if response_text:
                yield AgentEvent.text_complete(response_text, agent=self.agent_type)
                self.session.loop_detector.record_action(
                    "response",
                    text=response_text,
                )

            if not tool_calls:
                if usage:
                    self.session.context_manager.set_latest_usage(usage)
                    self.session.context_manager.add_usage(usage)

                self.session.context_manager.prune_tool_outputs()
                return

            tool_call_results: list[ToolResultMessage] = []

            for tool_call in tool_calls:
                yield AgentEvent.tool_call_start(
                    tool_call.call_id,
                    tool_call.name,
                    tool_call.arguments,
                    agent=self.agent_type
                )

                self.session.loop_detector.record_action(
                    "tool_call",
                    tool_name=tool_call.name,
                    args=tool_call.arguments,
                )

                logger.debug(
                    "Tool call started",
                    extra={
                        "fields": {
                            "tool": tool_call.name,
                            "argument_keys": sorted((tool_call.arguments or {}).keys()),
                        }
                    },
                )

                with tool_span(
                    tool_name=tool_call.name, arguments=tool_call.arguments
                ) as tspan:
                    invocation_task = asyncio.create_task(
                        self.session.tool_registry.invoke(
                            tool_call.name,
                            tool_call.arguments,
                            self.config.cwd,
                            self.session.hook_system,
                            self.session.approval_manager,
                        )
                    )

                    # While the tool is running (it might be paused waiting for
                    # approval), drain the event queue and yield approval requests.
                    while not invocation_task.done():
                        try:
                            event_data = await asyncio.wait_for(
                                self.session.event_queue.get(),
                                timeout=0.1
                            )
                            data = event_data["data"]
                            yield AgentEvent.approval_request(
                                approval_id=data["approval_id"],
                                tool_name=data["tool_name"],
                                description=data["description"],
                                params=data.get("params"),
                                agent=self.agent_type
                            )
                        except asyncio.TimeoutError:
                            continue

                    result = await invocation_task
                    # Records success plus a content-safe result summary, so tool
                    # behaviour is auditable without storing patient data.
                    finish_tool_span(tspan, result)

                logger.debug(
                    "Tool call finished",
                    extra={
                        "fields": {
                            "tool": tool_call.name,
                            "success": result.success,
                        }
                    },
                )

                yield AgentEvent.tool_call_complete(
                    tool_call.call_id,
                    tool_call.name,
                    result,
                    agent=self.agent_type
                )

                tool_call_results.append(
                    ToolResultMessage(
                        tool_call_id=tool_call.call_id,
                        content=result.to_model_output(),
                        is_error=not result.success,
                    )
                )

            for tool_result in tool_call_results:
                self.session.context_manager.add_tool_result(
                    tool_result.tool_call_id,
                    tool_result.content,
                )

            loop_detection_error = self.session.loop_detector.check_for_loop()
            if loop_detection_error:
                loop_prompt = create_loop_breaker_prompt(loop_detection_error)
                self.session.context_manager.add_user_message(loop_prompt)

            if usage:
                self.session.context_manager.set_latest_usage(usage)
                self.session.context_manager.add_usage(usage)

            self.session.context_manager.prune_tool_outputs()
        logger.error("Maximum turns (%s) reached", max_turns)
        yield AgentEvent.agent_error(f"Maximum turns ({max_turns}) reached")

    @property
    def _evaluation_agent_name(self) -> str:
        """The name used to look up which metrics apply to this agent.

        Prefers the human-readable configured name, falling back to the class
        name the session already carries, so generic agents are evaluated under
        their configured identity rather than a shared class.
        """
        return (
            self.configured_agent_name
            or self.session.agent_name
            or self.__class__.__name__
        )

    def _schedule_real_time_evaluation(
        self,
        message: str,
        final_response: str | None,
        duration_ms: float,
    ) -> None:
        """Hand a finished turn to the real-time evaluation observer, if any.

        Never blocks, never mutates the reply, never raises. Two reasons for the
        guards below:

        - An empty reply is not evidence of anything, so it is not evaluated.
        - The observer is optional infrastructure; a failure while scheduling must
          never reach the caller of ``run``.

        Args:
            message: The user input for this turn.
            final_response: The agent's reply, when it produced one.
            duration_ms: How long this turn took.
        """
        observer = getattr(self, "evaluation_observer", None)
        if observer is None or not final_response:
            return

        try:
            context = self._build_evaluation_context(
                message, final_response, duration_ms
            )
            logger.debug(
                "Scheduling real-time evaluation for agent '%s'",
                context.agent_name,
                extra={
                    "fields": {
                        "agent_name": context.agent_name,
                        "execution_id": context.execution_id,
                        "session_id": context.session_id,
                        "duration_ms": round(duration_ms, 1),
                    }
                },
            )
            observer.schedule(context)
        except Exception as exc:  # noqa: BLE001 - evaluation must never break a turn
            logger.debug("Real-time evaluation was not scheduled: %s", exc)

    def _build_evaluation_context(
        self,
        message: str,
        final_response: str,
        duration_ms: float,
    ) -> EvaluationContext:
        """Assemble one evaluation context for a finished turn.

        Uses the same value object the offline runner produces, with the
        session's conversation history as the context. Message *content* is
        deliberately not copied into the metadata, so results stored in MLflow
        contain no patient text.
        """
        from evaluation.core.models import EvaluationContext, EvaluationMode

        metadata: dict[str, Any] = {"agent_class": self.__class__.__name__}
        if self.default_skill:
            metadata["skill"] = self.default_skill
        # Lets the observer attach scores to the same run the trace lives on,
        # instead of creating a second run for the same execution.
        run_id = self._execution_run_id()
        if run_id:
            metadata["mlflow_run_id"] = run_id
        # The trace id lets a tool metric read the tool calls MLflow recorded on
        # this execution's spans.
        trace_id = getattr(self, "_last_trace_id", None)
        if trace_id:
            metadata["mlflow_trace_id"] = trace_id

        return EvaluationContext(
            agent_name=self._evaluation_agent_name,
            input=message,
            output=final_response,
            context=tuple(self._conversation_history()),
            # The execution's own id, not a fresh one: the run, the trace and the
            # evaluation results must all be correlatable by the same value.
            execution_id=self.session.execution_id or str(uuid.uuid4()),
            session_id=self.session.session_id,
            model=self._evaluation_model_name(),
            duration_ms=duration_ms,
            metadata=metadata,
            mode=EvaluationMode.REALTIME,
        )

    def _evaluation_model_name(self) -> str | None:
        """Best available model label for this turn, when one is known."""
        try:
            model = getattr(self.config, "model_name", None)
        except Exception:  # noqa: BLE001 - this is enrichment, never critical
            return None
        return str(model) if model else None

    def _conversation_history(self) -> list[str]:
        """Prior conversation lines in the dataset's ``Role: text`` format.

        The intake rules read prior questions out of ``case.context``, so the
        real-time path has to speak the same format or
        ``no_repeated_questions`` would silently do nothing live.
        """
        manager = self.session.context_manager
        if manager is None:
            return []

        lines: list[str] = []
        for message in getattr(manager, "messages", []) or []:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            role = str(message.get("role", "user")).lower()
            if role == "system" or not isinstance(content, str) or not content.strip():
                continue
            label = "Assistant" if role in {"assistant", "agent"} else "Patient"
            lines.append(f"{label}: {content.strip()}")
        return lines


    async def __aenter__(self) :
        
        await self.session.initialize()
        # for tool in self.session.tool_registry.get_tools():
        #     # Check if the tool class has a 'session' attribute
        #     if hasattr(tool, 'session'):
        #         tool.session = self.session
        #         print(f"DEBUG: Linked Session to Tool: {tool.name}")
        return self

    async def __aexit__(
        self,
        exc_type,
        exc_val,
        exc_tb,  
    ) -> None:
        if self.session and self.session.client and self.session.mcp_manager:
            await self.session.client.close()
            await self.session.mcp_manager.shutdown()
            # Cleanup MLflow run
            self.session.cleanup()
            self.session = None
