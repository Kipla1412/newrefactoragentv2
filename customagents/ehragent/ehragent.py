# customagents/mcpagent/mcpagent.py

from __future__ import annotations
from typing import AsyncGenerator

from agent.agent import Agent
from agent.events import AgentEvent, AgentType
from .ehrprompt import EHR_AGENT_PROMPT
from prompts.system import get_system_prompt


class EHRAgent(Agent):

    def __init__(self, config, session=None, skill_registry=None):

        super().__init__(
            config=config,
            system_prompt=EHR_AGENT_PROMPT,
            agent_type=AgentType.EHR,
            skill_registry=skill_registry,
        )

        # optional shared session
        if session:
            self.session = session

    async def run(
        self,
        message: str = "",
        skill: str | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:

        # initialize session
        if not self.session.context_manager:
            await self.session.initialize()

        self.session.agent_name = "EHRAgent"

        # build final prompt
        full_system_prompt = get_system_prompt(
            config=self.config,
            role_prompt=self.system_prompt,
        )

        self.session.context_manager.set_system_prompt(
            full_system_prompt
        )

        # add user message
        if message:
            self.session.context_manager.add_user_message(message)

        # tracking
        self.session.start_mlflow_run(
            "EHR Agent Request"
        )

        try:

            # Optional reusable skill, scoped to this request only. Falls back
            # to the skill this agent configures in its own config.yaml.
            with self.use_skill(self.active_skill(skill)):
                # use inherited framework loop
                async for event in self._agentic_loop():
                    yield event

        except Exception as e:

            yield AgentEvent.agent_error(
                error=f"EHR Agent Error: {str(e)}"
            )

        finally:
            self.session.end_mlflow_run()

    async def __aenter__(self):

        if not self.session.context_manager:
            await self.session.initialize()

        return self