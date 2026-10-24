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
import json
import time
from datetime import datetime
import asyncio

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
        
        # self.session.context_manager.clear()

        self.session.start_mlflow_run(message)
        
        await self.session.hook_system.trigger_before_agent(message)

        yield AgentEvent.agent_start(message)

        self.session.context_manager.set_system_prompt(self.system_prompt)
        
        self.session.context_manager.add_user_message(message)

        final_response: str | None = None

        async for event in self._agentic_loop():
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")

        await self.session.hook_system.trigger_after_agent(message, final_response)

        # Extract tools used and token usage from the session
        tools_used = []  # You can populate this from the session context
        token_usage = None  # You can extract this from the LLM client response
        
        yield AgentEvent.agent_end(final_response)

        self.session.end_mlflow_run()

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        max_turns = self.config.max_turns

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
