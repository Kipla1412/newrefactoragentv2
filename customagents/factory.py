"""Agent composition root.

``AgentFactory`` is the *only* place that knows the concrete agent classes
and how the pieces are wired. Everything it does is delegated to focused
collaborators from :mod:`customagents.composition`, so each concern has a
single reason to change (SRP), new agents/providers are added without
editing the factory (OCP), and the collaborators can be swapped wholesale
(DIP via :meth:`AgentFactory.configure`).

Adding an agent:

    from customagents.factory import AgentFactory
    AgentFactory.components().types.register("triage", TriageAgent)
"""

from customagents.composition import AgentComponents
from customagents.consultagent.consultagent import ConsultingAgent
from customagents.diarizeagent.diarizeagent import DiarizeAgent
from customagents.doctorassistent.docagent import DoctorAssistantAgent
from customagents.documentagent.documentagent import DocumentRetrievalAgent
from customagents.ehragent.ehragent import EHRAgent
from customagents.medicalagent.medicalagent import MedicalAgent
from customagents.pdfagent.pdfagent import PDFChatAgent
from customagents.previsitagent.intakeagent import IntakeAgent
from customagents.reportagent.assesment import AssessmentAgent
from customagents.reportagent.clinicalextraction import ClinicalExtractionAgent
from customagents.reportagent.soapagent import SOAPAgent
from customagents.voiceagent.voiceconsult import VoiceConsultAgent
from customagents.voiceagent.voiceintake import VoiceIntakeAgent
from config.config import Config
from config.schemas import AgentYaml
from agent.session import Session

# Default name → class mapping. This is the ONE place concretions are
# referenced; new agents are registered here (or at runtime through the
# registry) without touching the factory logic.
DEFAULT_AGENT_CLASSES: dict[str, type] = {
    "doc": DoctorAssistantAgent,
    "soap": SOAPAgent,
    "assessment": AssessmentAgent,
    "consult": ConsultingAgent,
    "intake": IntakeAgent,
    "ehr": EHRAgent,
    "clinical_extraction": ClinicalExtractionAgent,
    "voice_intake": VoiceIntakeAgent,
    "voice_consult": VoiceConsultAgent,
    "diarize": DiarizeAgent,
    "medical": MedicalAgent,
    "document_retrieval": DocumentRetrievalAgent,
    "pdf_chat": PDFChatAgent,
}


class AgentFactory:
    """Compose agents from configuration.

    Resolves the two-level agent configuration (application-level enabled
    flags + agent-specific config.yaml) through the ModelRegistry, the
    PromptRegistry, and the existing ToolRegistry, then constructs the
    concrete agent. Callers request agents by type only; they never see
    model/prompt/tool selection.

    Attributes:
        _agent_registry: Backward-compatible view of the default agent map.
    """

    # Backward-compatible alias for the default registry contents.
    _agent_registry = DEFAULT_AGENT_CLASSES

    _components: AgentComponents | None = None

    # Collaborator wiring (DIP)

    @classmethod
    def components(cls) -> AgentComponents:
        """Return the active collaborator bundle, building the default.

        Returns:
            The AgentComponents used by :meth:`create`.
        """
        if cls._components is None:
            cls._components = AgentComponents.default(DEFAULT_AGENT_CLASSES)
        return cls._components

    @classmethod
    def configure(cls, components: AgentComponents) -> None:
        """Replace the collaborator bundle.

        Injection seam: tests and alternative deployments can supply their
        own collaborators without touching module state.

        Args:
            components: The bundle to use.
        """
        cls._components = components

    @classmethod
    def reset_components(cls) -> None:
        """Restore the default collaborator bundle."""
        cls._components = None

    @classmethod
    def known_agent_names(cls) -> set[str]:
        """Return the set of registered agent type names.

        Returns:
            The names in the agent type registry.
        """
        return cls.components().types.known_names()


    # Public API

    @classmethod
    def create(
        cls,
        agent_type: str,
        config: Config,
        session: Session | None = None,
        resolved=None,
    ):
        """Resolve and construct the concrete Agent instance.

        Model, prompt, and tools are resolved from the agent-specific
        config.yaml through the ModelRegistry, PromptRegistry, and the
        existing ToolRegistry when the agent is configured; otherwise the
        agent's hardcoded defaults are used.

        An agent explicitly disabled in agents.yaml is rejected. An agent
        with no application-level entry at all keeps its hardcoded
        defaults (backward compatible).

        Args:
            agent_type: Agent identifier matching an AgentType string.
            config: Runtime configuration.
            session: Optional shared session to inject.
            resolved: Optional pre-resolved ResolvedAgentConfig (used at
                startup so no YAML is read per request).

        Returns:
            The constructed Agent instance.

        Raises:
            ValueError: If the agent type is not registered.
            ConfigError: If the agent is disabled in agents.yaml, or a
                configured model, prompt, or tool cannot be resolved.
        """
        components = cls.components()
        name = agent_type.lower()

        agent_class = components.types.get(name)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}. Check your registry.")

        # 1. Configuration (enforces the enabled flag).
        agent_yaml, model_registry = components.configs.provide(name, config, resolved)

        # 2. Derived configuration: model, then framework settings.
        agent_config = components.models.resolve(agent_yaml, config, model_registry)
        agent_config = components.settings.apply(agent_config, agent_yaml)

        # 3. Prompt content.
        prompt = components.prompts.resolve(name, agent_yaml, agent_config)

        # 4. Construct, then apply the resolved overrides.
        agent = agent_class(agent_config, session=session)
        tools = components.tools.resolve(name, agent_yaml, agent_config, session)
        components.overrides.apply(agent, prompt, tools)

        return agent

   
    # Backward-compatible shim
   
    def _resolve_tools(
        cls,
        agent_name: str,
        agent_yaml: AgentYaml | None,
        config: Config,
        session: Session | None,
    ) -> list | None:
        """Resolve configured tool names (delegates to ToolResolver).

        Kept for backward compatibility; prefer
        ``AgentFactory.components().tools.resolve(...)``.

        Args:
            agent_name: Agent identifier used in error messages.
            agent_yaml: The agent's resolved configuration, or None.
            config: Runtime configuration.
            session: Optional shared session carrying the ToolRegistry.

        Returns:
            The resolved tools, or None if the agent has no tool config.
        """
        return cls.components().tools.resolve(agent_name, agent_yaml, config, session)