"""A configuration-driven agent.

Every aspect of this agent's behaviour comes from its own ``config.yaml``:

===========  ==================  ============================================
Key          Resolved by         Applied as
===========  ==================  ============================================
name         —                   the agent's identity in agents.yaml
agent_class  AgentTypeRegistry   ``"generic"`` — which class builds it
prompt       PromptRegistry      the system prompt (via get_system_prompt)
model        ModelRegistry       the runtime Config for this agent
tools        ToolRegistry        ``allowed_tool_names``
skill        SkillRegistry       the default skill for each request
agent_name   —                   display name for MLflow and errors
===========  ==================  ============================================

It exists so a new agent of this shape needs **no new Python**: any number of
agents share this one class, each declaring ``agent_class: generic`` in its own
``config.yaml``:

.. code-block:: yaml

    name: triage_nurse
    agent_class: generic
    model: clinical_model
    prompt: triage
    skill: triage-questions
    agent_name: TriageNurseAgent

There is deliberately no prompt text and no domain logic here — that belongs to
the configured prompt and skill, so this class never needs editing. Agents with
behaviour beyond "prompt + tools + skill" still need their own class; see
``docs/customagents/new-agent.md``.
"""

from __future__ import annotations

from typing import AsyncGenerator

from agent.agent import Agent
from agent.events import AgentEvent, AgentType
from prompts.system import get_system_prompt
from skills import SkillRegistry
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Placeholder prompt. AgentFactory replaces it with the prompt resolved from the
# agent's config.yaml. run() refuses to start while it is still empty, rather
# than sending an unconfigured agent to the model.
PENDING_PROMPT = ""


class GenericAgent(Agent):
    """An agent whose prompt, model, tools, and skill all come from config."""

    def __init__(self, config, session=None, skill_registry: SkillRegistry | None = None):
        """Initialize the agent.

        Args:
            config: Application configuration (already carrying this agent's
                configured model).
            session: Optional existing session to preserve history and reuse
                the shared ToolRegistry, MCP connections, and LLM client.
            skill_registry: Skill source. Defaults to the bundled library.
        """
        super().__init__(
            config,
            PENDING_PROMPT,
            AgentType.GENERIC,
            skill_registry=skill_registry,
        )
        if session:
            self.session = session

    @property
    def display_name(self) -> str:
        """Name used for MLflow tagging and error messages.

        Falls back to the class name when no ``name`` is configured.
        """
        return self.configured_agent_name or self.__class__.__name__

    async def run(
        self, message: str = "", skill: str | None = None
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run one turn of the standard agentic loop.

        Args:
            message: The user's request.
            skill: Optional skill name; falls back to this agent's configured
                default.

        Yields:
            AgentEvent objects produced by the agentic loop.
        """
        prompt = (self.system_prompt or "").strip()
        if not prompt:
            # Fail before touching the session: an unconfigured agent must not
            # reach the model.
            yield AgentEvent.agent_error(
                error=(
                    f"{self.display_name} has no prompt configured. Set 'prompt:' "
                    "in its config.yaml to a logical name registered in "
                    "PromptRegistry."
                )
            )
            return

        if not self.session.context_manager:
            await self.session.initialize()

        self.session.agent_name = self.display_name

        self.session.context_manager.set_system_prompt(
            get_system_prompt(config=self.config, role_prompt=prompt)
        )

        if message:
            self.session.context_manager.add_user_message(message)

        self.session.start_mlflow_run(message or self.display_name)

        try:
            with self.use_skill(self.active_skill(skill)):
                async for event in self._agentic_loop():
                    if hasattr(event, "data") and event.data.get("agent") is None:
                        event.data["agent"] = self.agent_type
                    yield event
        except Exception as e:
            logger.exception("%s run failed", self.display_name)
            yield AgentEvent.agent_error(error=f"{self.display_name} error: {e}")
        finally:
            self.session.end_mlflow_run()

    async def __aenter__(self):
        if not self.session.context_manager:
            await self.session.initialize()
        return self
