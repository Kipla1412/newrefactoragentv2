from customagents.doctorassistent.docagent import DoctorAssistantAgent
from customagents.reportagent.soapagent import SOAPAgent
from customagents.reportagent.assesment import AssessmentAgent
from customagents.previsitagent.intakeagent import IntakeAgent
from customagents.consultagent.consultagent import ConsultingAgent
from customagents.ehragent.ehragent import EHRAgent
from customagents.reportagent.clinicalextraction import ClinicalExtractionAgent
from customagents.voiceagent.voiceintake import VoiceIntakeAgent
from customagents.voiceagent.voiceconsult import VoiceConsultAgent
from customagents.diarizeagent.diarizeagent import DiarizeAgent
from customagents.medicalagent.medicalagent import MedicalAgent
from customagents.documentagent.documentagent import DocumentRetrievalAgent
from customagents.pdfagent.pdfagent import PDFChatAgent
from config.config import Config
from config.schemas import AgentYaml
from agent.session import Session
from registries.model_registry import ModelRegistry
from registries.prompt_registry import PromptRegistry
from utils.errors import ConfigError

# Module-level cache of the two-level agent config loader. Built lazily on
# first use and reused across factory calls so agent config files are not
# re-scanned from disk on every request.
_CACHED_LOADER = None


def _get_loader(config: Config):
    """Return the cached AgentConfigLoader, building it on first use.

    Args:
        config: Runtime configuration.

    Returns:
        The cached AgentConfigLoader with its configuration resolved.
    """
    global _CACHED_LOADER
    if _CACHED_LOADER is None:
        from config.agent_config_loader import AgentConfigLoader

        _CACHED_LOADER = AgentConfigLoader(
            config,
            known_agents=AgentFactory.known_agent_names(),
        )
        # Force one discovery pass so config files are cached from the start.
        _CACHED_LOADER.load()
    return _CACHED_LOADER


class AgentFactory:
    """
    Composition root for agent construction.

    Resolves the two-level agent configuration (application-level enabled
    flags + agent-specific config.yaml) through the ModelRegistry, the
    PromptRegistry, and the existing ToolRegistry, then constructs the
    concrete agent. Callers request agents by type only; they never see
    model/prompt/tool selection.
    """

    # Map the strings/enums to classes
    _agent_registry = {
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

    @classmethod
    def known_agent_names(cls) -> set[str]:
        """Return the set of registered agent type names.

        Returns:
            The keys of the agent registry.
        """
        return set(cls._agent_registry.keys())

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
            ConfigError: If a configured model, prompt, or tool cannot be
                resolved.
        """
        agent_class = cls._agent_registry.get(agent_type.lower())

        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}. Check your registry.")

        # Use the cached startup configuration when available; otherwise fall
        # back to the cached loader (no per-request filesystem re-scan).
        agent_yaml = None
        model_registry = None
        if resolved is not None:
            agent_yaml = resolved.enabled_agents.get(agent_type.lower())
            model_registry = resolved.model_registry
        else:
            loader = _get_loader(config)
            agent_yaml = loader.load_agent(agent_type.lower())
            if loader.resolved is not None:
                model_registry = loader.resolved.model_registry

        # Model: derived config copy so the LLM client reads it at call time.
        if agent_yaml and agent_yaml.model:
            registry = model_registry or ModelRegistry(config.models.definitions)
            if registry.has(agent_yaml.model):
                config = registry.apply(agent_yaml.model, config)

        # Prompt: registry wins; fallback to the agent class's hardcoded prompt.
        prompt = None
        if agent_yaml and agent_yaml.prompt:
            config.resolve_prompt_definition(agent_type, agent_yaml.prompt)
            prompt = PromptRegistry.from_builtin().get(agent_yaml.prompt)

        # Design Pattern Principle: Constructor Injection
        # We inject 'config' and 'session' immediately.
        agent = agent_class(config, session=session)

        if prompt is not None:
            agent.system_prompt = prompt

        # Tools: resolve each logical name through the EXISTING ToolRegistry
        # and restrict what the agent may use via its allowed_tool_names hook.
        resolved_tools = cls._resolve_tools(agent_type, agent_yaml, config, session)
        if resolved_tools is not None:
            agent.configured_tools = resolved_tools
            agent.allowed_tool_names = [tool.name for tool in resolved_tools]

        return agent

    @classmethod
    def _resolve_tools(
        cls,
        agent_name: str,
        agent_yaml: AgentYaml | None,
        config: Config,
        session: Session | None,
    ) -> list | None:
        """Resolve configured logical tool names via the existing ToolRegistry.

        Each name in ``agent_yaml.tools`` is looked up through the existing
        ToolRegistry. Unresolvable names raise a ConfigError identifying the
        agent and the missing tool. Returns None when the agent has no tool
        configuration (which leaves the agent's tools unrestricted). An
        explicitly empty ``tools`` list yields an empty list (no tools).

        Args:
            agent_name: Agent identifier used in error messages.
            agent_yaml: The agent's resolved configuration, or None.
            config: Runtime configuration.
            session: Optional shared session carrying the ToolRegistry.

        Returns:
            List of resolved Tool instances, or None if no tool configuration.

        Raises:
            ConfigError: If a configured tool name is not registered.
        """
        if agent_yaml is None or agent_yaml.tools is None:
            return None

        registry = session.tool_registry if session is not None else None
        if registry is None:
            from tools.registry import create_default_registry
            registry = create_default_registry(config)

        resolved = []
        for tool_name in agent_yaml.tools:
            tool = registry.get(tool_name)
            if tool is None:
                raise ConfigError(
                    f"Agent '{agent_name}' references unknown tool '{tool_name}'"
                )
            resolved.append(tool)
        return resolved