"""Patient demographics agent.

Same shape as the EHR and consult agents: a concrete class, its own prompt, and
the shared agentic loop. The field-by-field workflow lives in the
``patient-demographics-summary`` skill, which this agent activates by default.
"""

from __future__ import annotations

from typing import AsyncGenerator

from agent.agent import Agent
from agent.events import AgentEvent, AgentType
from prompts.system import get_system_prompt
from skills import SkillRegistry

from .demographicsprompt import DEMOGRAPHICS_PROMPT


class DemographicsAgent(Agent):

    def __init__(self, config, session=None, skill_registry: SkillRegistry | None = None):
        """Initialize the demographics agent.

        Args:
            config: Application configuration.
            session: Optional existing session to preserve history and reuse
                the shared ToolRegistry, MCP connections, and LLM client.
            skill_registry: Skill source. Defaults to the bundled library.
        """
        super().__init__(
            config=config,
            system_prompt=DEMOGRAPHICS_PROMPT,
            agent_type=AgentType.DEMOGRAPHICS,
            skill_registry=skill_registry,
        )
        if session:
            self.session = session

    async def run(
        self,
        message: str = "",
        skill: str | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run one turn of the standard agentic loop.

        Args:
            message: The user's request.
            skill: Optional skill name; falls back to the skill configured for
                this agent in customagents/demographicsagent/config.yaml.

        Yields:
            AgentEvent objects produced by the agentic loop.
        """
        if not self.session.context_manager:
            await self.session.initialize()

        self.session.agent_name = "DemographicsAgent"

        full_system_prompt = get_system_prompt(
            config=self.config,
            role_prompt=self.system_prompt,
        )

        self.session.context_manager.set_system_prompt(full_system_prompt)

        if message:
            self.session.context_manager.add_user_message(message)

        self.session.start_mlflow_run("Demographics Agent Request")

        try:
            with self.use_skill(self.active_skill(skill)):
                async for event in self._agentic_loop():
                    yield event

        except Exception as e:
            yield AgentEvent.agent_error(
                error=f"Demographics Agent Error: {str(e)}"
            )

        finally:
            self.session.end_mlflow_run()

    async def __aenter__(self):
        if not self.session.context_manager:
            await self.session.initialize()
        return self
