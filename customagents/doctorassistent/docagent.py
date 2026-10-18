from __future__ import annotations
from agent.agent import Agent
from .docprompt import DOC_PROMPT
from agent.events import AgentType, AgentEventType, AgentEvent
from skills import SkillRegistry
import json
import re
from typing import Any, AsyncGenerator, List

class DoctorAssistantAgent(Agent):
    def __init__(self, config, session=None, skill_registry: SkillRegistry | None = None):
        """
        Initializes the Assistant with the DOC_PROMPT.
        Reuses the base Agent infrastructure for tracking and session management.

        Args:
            config: Application configuration.
            session: Optional existing session to preserve history.
            skill_registry: Skill source used when a skill is requested by
                name. Defaults to the framework's bundled skills library.
        """
        super().__init__(
            config,
            DOC_PROMPT,
            AgentType.DOC,
            skill_registry=skill_registry,
        )
        if session:
            self.session = session

    async def recommend_questions(
        self, conversation: List[str], skill: str | None = None
    ) -> dict:
        """
        NON-STREAMING version for the 'Magic Button' router.
        Consumes the async generator internally and returns a clean dictionary.

        Args:
            conversation: Conversation history to review.
            skill: Optional skill name. When given, the skill's instructions
                are activated on the ContextManager for this call only, so they
                reach the LLM as part of the system message and are removed
                again when the call returns.

        Returns:
            A dict with the validated ``questions`` list, or an ``error``.
        """
        # 1. Initialize session and context
        if not self.session.context_manager:
            await self.session.initialize()

        self.session.agent_name = "DoctorAssistant"
        
        # Clear context to ensure we only focus on the current recommendation task
        self.session.context_manager.clear()
        self.session.context_manager.set_system_prompt(self.system_prompt)
        
        # Convert the list of strings into a single block or add them individually
        history_block = "\n".join(conversation)
        self.session.context_manager.add_user_message(
            f"Review this conversation and suggest 3 questions:\n{history_block}"
        )

        self.session.start_mlflow_run("Generate Clinical Questions")

        response_text = ""
        try:
            # 2. CONSUME the stream internally
            # We use 'async for' because self._agentic_loop() is an AsyncGenerator
            with self.use_skill(self.active_skill(skill)):
                async for event in self._agentic_loop():
                    if event.type == AgentEventType.TEXT_DELTA:
                        response_text += event.data.get("content", "")
                    elif event.type == AgentEventType.TEXT_COMPLETE:
                        response_text = event.data.get("content", "")

            # 3. Parse and return the final JSON
            return self.parse_questions(response_text)

        except Exception as e:
            return {"questions": [], "error": str(e)}
        finally:
            self.session.end_mlflow_run()

    @staticmethod
    def validate_questions(payload: Any, expected: int = 3) -> List[str]:
        """Validate a follow-up question payload.

        The prompt (and the matching skill) ask the model for strict JSON
        containing exactly three questions; this enforces that contract in
        application code rather than trusting the model's formatting. It runs
        whether or not a skill was activated.

        Args:
            payload: Parsed JSON returned by the model.
            expected: Required number of questions.

        Returns:
            The cleaned question strings.

        Raises:
            ValueError: If the payload is not an object with a ``questions``
                list of non-empty strings, or the count does not match.
        """
        if not isinstance(payload, dict):
            raise ValueError("Response must be a JSON object")

        questions = payload.get("questions")
        if not isinstance(questions, list):
            raise ValueError("'questions' must be a list")

        cleaned: List[str] = []
        for question in questions:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Each question must be a non-empty string")
            cleaned.append(question.strip())

        if len(cleaned) != expected:
            raise ValueError(f"Expected exactly {expected} questions, got {len(cleaned)}")

        return cleaned

    async def run(
        self, message: str = "", skill: str | None = None
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        STREAMING version.
        Use this if you ever want to show the assistant 'typing' in the UI.

        Args:
            message: The user's message.
            skill: Optional skill name, activated for this request only.

        Yields:
            AgentEvent objects produced by the agentic loop.
        """
        if not self.session.context_manager:
            await self.session.initialize()

        self.session.agent_name = "DoctorAssistant"

        # Same raw-prompt convention as the report agents: DOC_PROMPT is the
        # complete instruction set for this agent.
        self.session.context_manager.set_system_prompt(self.system_prompt)

        if message:
            self.session.context_manager.add_user_message(message)

        self.session.start_mlflow_run("Streaming Clinical Questions")

        try:
            with self.use_skill(self.active_skill(skill)):
                async for event in self._agentic_loop():
                    if hasattr(event, "data") and event.data.get("agent") is None:
                        event.data["agent"] = self.agent_type
                    yield event
        except Exception as e:
            yield AgentEvent.agent_error(error=f"Assistant Loop Error: {str(e)}")
        finally:
            self.session.end_mlflow_run()

    @staticmethod
    def parse_questions(raw_content: str) -> dict:
        """
        Utility method to clean markdown and parse JSON.

        The parsed payload is validated before it is returned, so callers
        always get either exactly the expected number of questions or an
        error — never an unchecked model payload.

        Args:
            raw_content: Raw model output.

        Returns:
            A dict containing ``questions``, or ``questions`` plus ``error``
            when the output could not be used.
        """
        try:
            # Strip backticks (e.g., ```json ... ```)
            clean_content = re.sub(r"```json\s?|```", "", raw_content).strip()
            payload = json.loads(clean_content)
        except json.JSONDecodeError:
            return {"questions": [], "error": "Invalid JSON format received", "raw": raw_content[:100]}

        try:
            return {"questions": DoctorAssistantAgent.validate_questions(payload)}
        except ValueError as exc:
            return {"questions": [], "error": str(exc)}

