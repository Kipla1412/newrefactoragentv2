from __future__ import annotations

from typing import Any

from utils.errors import ConfigError


class PromptRegistry:
    """Resolve logical prompt names to registered prompt content.

    Maps configured logical names to existing prompt constants (e.g.
    ``"clinical_extraction"`` → ``CLINICAL_EXTRACTION_PROMPT``). Prompt
    content is not duplicated here; the registry imports the existing
    constants from the agent packages.

    Attributes:
        prompts: Mapping of logical prompt names to prompt content.
    """

    def __init__(self, prompts: dict[str, str] | None = None) -> None:
        """Initialize the registry with an optional prompt mapping.

        Args:
            prompts: Mapping of logical prompt names to prompt content.
        """
        self.prompts: dict[str, str] = dict(prompts or {})

    def get(self, name: str) -> str:
        """Return the prompt content for a logical prompt name.

        Args:
            name: Logical prompt identifier configured for an agent.

        Returns:
            The resolved prompt content.

        Raises:
            ConfigError: If the prompt is not registered.
        """
        prompt = self.prompts.get(name)
        if prompt is None:
            raise ConfigError(f"Prompt '{name}' is not registered")
        return prompt

    def has(self, name: str) -> bool:
        """Return whether a logical prompt name is registered.

        Args:
            name: Logical prompt identifier.

        Returns:
            True if the name resolves to prompt content.
        """
        return name in self.prompts

    @classmethod
    def from_builtin(cls) -> "PromptRegistry":
        """Build a registry pre-populated with every agent prompt constant.

        Each built-in prompt is keyed by its logical name (e.g.
        ``"clinical_extraction"``, ``"soap"``, ``"medical"``). The logical
        names are what the YAML agent configuration references — they are
        not coupled to the AgentType enum.

        Returns:
            A PromptRegistry containing all built-in agent prompts.
        """
        from customagents.consultagent.consultprompt import CONSULT_PROMPT
        from customagents.demographicsagent.demographicsprompt import DEMOGRAPHICS_PROMPT
        from customagents.diarizeagent.diarizeprompt import DIARIZE_ROLE_MAP_PROMPT
        from customagents.doctorassistent.docprompt import DOC_PROMPT
        from customagents.documentagent.documentprompt import DOCUMENT_RETRIEVAL_PROMPT
        from customagents.ehragent.ehrprompt import EHR_AGENT_PROMPT
        from customagents.medicalagent.medicalprompt import MEDICAL_PROMPT
        from customagents.pdfagent.pdfprompt import PDF_CHAT_PROMPT
        from customagents.previsitagent.intakeprompt import INTAKE_PROMPT
        from customagents.reportagent.assessmentprompt import ASSESSMENT_PROMPT
        from customagents.reportagent.clinicalprompt import CLINICAL_EXTRACTION_PROMPT
        from customagents.reportagent.soapprompt import SOAP_PROMPT
        from customagents.voiceagent.voiceconsultprompt import VOICE_CONSULT_PROMPT
        from customagents.voiceagent.voiceintakeprompt import VOICE_INTAKE_PROMPT

        return cls({
            "doc": DOC_PROMPT,
            "soap": SOAP_PROMPT,
            "assessment": ASSESSMENT_PROMPT,
            "consult": CONSULT_PROMPT,
            "intake": INTAKE_PROMPT,
            "ehr": EHR_AGENT_PROMPT,
            "demographics": DEMOGRAPHICS_PROMPT,
            "clinical_extraction": CLINICAL_EXTRACTION_PROMPT,
            "voice_intake": VOICE_INTAKE_PROMPT,
            "voice_consult": VOICE_CONSULT_PROMPT,
            "diarize": DIARIZE_ROLE_MAP_PROMPT,
            "medical": MEDICAL_PROMPT,
            "document_retrieval": DOCUMENT_RETRIEVAL_PROMPT,
            "pdf_chat": PDF_CHAT_PROMPT,
        })