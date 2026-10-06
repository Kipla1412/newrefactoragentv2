from __future__ import annotations
from enum import Enum
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from pydantic import BaseModel, Field, model_validator
from speechtospeech.providers.stt.factory import create_stt_provider
from speechtospeech.providers.tts.factory import create_tts_provider
from speechtospeech.audioprocessor import AudioProcessor
from speechtospeech.speechtotext.sttengine import TranscriptionEngine
from speechtospeech.texttospeech.ttsengine import TTSEngine
from dotenv import load_dotenv
from utils.errors import ConfigError
from config.schemas import ApplicationAgentConfig
load_dotenv()


def _opensearch_url_host(url: str) -> str:
    return urlparse(url).hostname or "localhost"


def _opensearch_url_port(url: str) -> int:
    parsed = urlparse(url)
    return parsed.port or (443 if parsed.scheme == "https" else 9200)


def _opensearch_url_user(url: str) -> str:
    return urlparse(url).username or "admin"


def _opensearch_url_password(url: str) -> str | None:
    return urlparse(url).password

class ModelConfig(BaseModel):
    """Runtime model configuration.

    Used both as the default runtime model and as a named entry resolved
    by the ModelRegistry from configuration.

    Attributes:
        provider: Provider identifier (e.g. "openai", "anthropic",
            "gemini", "ollama"). Used to build the LiteLLM model string
            when ``litellm_model`` is not set.
        name: Model identifier passed to the LLM client.
        temperature: Sampling temperature (0.0-2.0).
        context_window: Maximum context window in tokens.
        litellm_model: Explicit LiteLLM model string (e.g.
            "openai/gpt-4.1", "anthropic/claude-3-5-sonnet"). When set it
            takes precedence over provider/name.
    """

    provider: str = "openai"
    name: str = "gpt-4.1" #"gpt-4o-mini" #"mistralai/devstral-2512:free"
    temperature: float = Field(default=1, ge=0.0, le=2.0)
    context_window: int = 256_000
    litellm_model: str | None = None

    def to_litellm_model(self) -> str:
        """Return the LiteLLM model string for this entry.

        Falls back to ``"<provider>/<name>"`` when no explicit
        ``litellm_model`` is configured.

        Returns:
            The LiteLLM model identifier.
        """
        return self.litellm_model or f"{self.provider}/{self.name}"


class ModelsSection(BaseModel):
    """Container for named model definitions loaded from configuration."""

    definitions: dict[str, ModelConfig] = Field(default_factory=dict)


class AgentsSection(BaseModel):
    """Container for application-level agent entries (enabled flags).

    The per-agent model/prompt/tools configuration lives in each
    customagents/<agent>/config.yaml, parsed by the AgentConfigLoader
    into AgentYaml. This section only tracks which agents are enabled.
    """

    definitions: dict[str, "ApplicationAgentConfig"] = Field(default_factory=dict)

class ShellEnvironmentPolicy(BaseModel):
    ignore_default_excludes: bool = False
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["*KEY*", "*TOKEN*", "*SECRET*"]
    )
    set_vars: dict[str, str] = Field(default_factory=dict)

class MCPServerConfig(BaseModel):
    enabled: bool = True
    startup_timeout_sec: float = 10

    # stdio transport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: Path | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    # http/sse transport
    url: str | None = None
    transport: str = "streamable-http"

    @model_validator(mode="after")
    def validate_transport(self) -> MCPServerConfig:
        has_command = self.command is not None
        has_url = self.url is not None

        if not has_command and not has_url:
            raise ValueError(
                "MCP Server must have either 'command' (stdio) or 'url' (http/sse)"
            )

        if has_command and has_url:
            raise ValueError(
                "MCP Server cannot have both 'command' (stdio) and 'url' (http/sse)"
            )

        return self


class ApprovalPolicy(str, Enum):
    ON_REQUEST = "on-request"
    ON_FAILURE = "on-failure"
    AUTO = "auto"
    AUTO_EDIT = "auto-edit"
    NEVER = "never"
    YOLO = "yolo"


class HookTrigger(str, Enum):
    BEFORE_AGENT = "before_agent"
    AFTER_AGENT = "after_agent"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    ON_ERROR = "on_error"


class HookConfig(BaseModel):
    name: str
    trigger: HookTrigger
    command: str | None = None  # python3 tests.py
    script: str | None = None  # *.sh
    timeout_sec: float = 30
    enabled: bool = True

    @model_validator(mode="after")
    def validate_hook(self) -> HookConfig:
        if not self.command and not self.script:
            raise ValueError("Hook must either have 'command' or 'script'")
        return self

class WikiConfig(BaseModel):
    enabled: bool = True
    root_path: str = "/home/kipla/aipipeline/aiplatform/storage/wiki"

class Neo4jConfig(BaseModel):
    enabled: bool = True
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password123"

class KnowledgeConfig(BaseModel):
    wiki: WikiConfig = Field(default_factory=WikiConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)

class Config(BaseModel):
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    cwd: Path = Field(default_factory=Path.cwd)
    shell_environment: ShellEnvironmentPolicy = Field(
        default_factory=ShellEnvironmentPolicy
    )
    hooks_enabled: bool = False
    hooks: list[HookConfig] = Field(default_factory=list)
    approval: ApprovalPolicy = ApprovalPolicy.AUTO_EDIT   # approval: ApprovalPolicy = ApprovalPolicy.ON_REQUEST
    max_turns: int = 100
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    allowed_tools: list[str] | None = Field(
        None,
        description="If set, only these tools will be available to the agent",
    )

    developer_instructions: str | None = None
    user_instructions: str | None = None

    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)

    models: ModelsSection = Field(default_factory=ModelsSection)
    agents: AgentsSection = Field(default_factory=AgentsSection)

    debug: bool = False

    def resolve_agent_config(self, name: str) -> ApplicationAgentConfig | None:
        """Return the application-level entry for an agent name, if any.

        Args:
            name: Agent identifier (matches the AgentType string).

        Returns:
            The resolved ApplicationAgentConfig (enabled flag), or None if
            the agent has no application-level entry. Per-agent model,
            prompt, and tools are resolved by the AgentConfigLoader.
        """
        return self.agents.definitions.get(name.lower())

    def resolve_model_definition(self, name: str) -> ModelConfig:
        """Return the ModelConfig for a logical model name.

        Args:
            name: Logical model identifier configured for an agent.

        Returns:
            The resolved ModelConfig.

        Raises:
            ConfigError: If the model is not registered.
        """
        definition = self.models.definitions.get(name)
        if definition is None:
            raise ConfigError(f"Model '{name}' is not registered")
        return definition

    def resolve_prompt_definition(self, agent: str, prompt: str) -> str:
        """Validate that a referenced prompt is registered for an agent.

        Args:
            agent: Agent identifier used in error messages.
            prompt: Logical prompt name configured for the agent.

        Returns:
            The prompt name, if registered.

        Raises:
            ConfigError: If the prompt is not registered.
        """
        from registries.prompt_registry import PromptRegistry
        registry = PromptRegistry.from_builtin()
        if not registry.has(prompt):
            raise ConfigError(
                f"Agent '{agent}' references unknown prompt '{prompt}'"
            )
        return prompt

    @property
    def api_key(self) -> str | None:
        return os.environ.get("API_KEY")

    @property
    def base_url(self) -> str | None:
        return os.environ.get("BASE_URL")

    @property
    def llm_routing_enabled(self) -> bool:
        """Whether the ModelRouter selects the model per request.

        Disabled by default so existing behaviour (single configured
        model) is preserved unless explicitly enabled.
        """
        return os.environ.get("LLM_ROUTING_ENABLED", "false").lower() == "true"

    @property
    def litellm_fallback_models(self) -> list[str]:
        """Ordered fallback models tried when the primary model fails.

        Read from ``LLM_FALLBACK_MODELS`` as a comma-separated list of
        logical or LiteLLM model names.
        """
        raw = os.environ.get("LLM_FALLBACK_MODELS", "")
        return [m.strip() for m in raw.split(",") if m.strip()]

    @property
    def litellm_timeout(self) -> float:
        """Per-request LiteLLM timeout in seconds."""
        return float(os.environ.get("LLM_TIMEOUT", "60"))

    @property
    def litellm_num_retries(self) -> int:
        """Number of retries for a single model before falling back."""
        return int(os.environ.get("LLM_NUM_RETRIES", "3"))

    @property
    def default_litellm_model(self) -> str:
        """LiteLLM model string for the runtime default model."""
        return self.model.to_litellm_model()

    @property
    def vision_model_name(self) -> str | None:
        """Vision-capable model used for OCR. Defaults to the main model."""
        return os.environ.get("VISION_MODEL_NAME") or self.model.name

    @property
    def model_name(self) -> str:
        return self.model.name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.model.name = value

    @property
    def temperature(self) -> float:
        return self.model.temperature

    # @model_name.setter
    # def temperature(self, value: str) -> None:
    #     self.model.temperature = value

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.model.temperature = value

    def validate(self) -> list[str]:
        errors: list[str] = []

        if not self.api_key:
            errors.append("No API key found. Set API_KEY environment variable")

        if not self.cwd.exists():
            errors.append(f"Working directory does not exist: {self.cwd}")

        return errors

    
    @property
    def iam_jwks_url(self):
        return os.environ.get("IAM_JWKS_URL")

    @property
    def iam_issuer(self):
        return os.environ.get("IAM_ISSUER")

    @property
    def mcp_servers_config(self) -> dict[str, MCPServerConfig]:
        base_url = self.fhir_base_url
        if not base_url:
            return {}
        return {
            "fhir": MCPServerConfig(
                url=base_url,
                enabled=True,
                transport="streamable-http",
            )
        }
        
    @property
    def fhir_base_url(self):
        return os.environ.get("FHIR_BASE_URL")

    @property
    def wiki_root_path(self) -> str:
        return os.environ.get("WIKI_ROOT_PATH", self.knowledge.wiki.root_path)

    @property
    def neo4j_uri(self) -> str:
        return os.environ.get("NEO4J_URI", self.knowledge.neo4j.uri)

    @property
    def neo4j_user(self) -> str:
        return os.environ.get("NEO4J_USER", self.knowledge.neo4j.user)

    @property
    def neo4j_password(self) -> str:
        return os.environ.get("NEO4J_PASSWORD", self.knowledge.neo4j.password)
    
    @property
    def mlflow_enabled(self) -> bool:
        return os.environ.get("MLFLOW_ENABLED", "false").lower() == "true"

    @property
    def mlflow_tracking_uri(self) -> str:
        return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


    @property
    def mlflow_experiment_name(self) -> str:
        return os.environ.get("MLFLOW_EXPERIMENT_NAME")
    
    
    @property
    def vad_enabled(self):
        return os.environ.get("VAD_ENABLED", "true") == "true"
    
    @property
    def sarvam_api_key(self):
        return os.environ.get("SARVAM_API_KEY")
    
    @property
    def sarvam_stt_model(self):
        return os.environ.get(
            "SARVAM_STT_MODEL",
            "saaras:v3"
        )

    @property
    def sarvam_tts_model(self):
        return os.environ.get(
            "SARVAM_TTS_MODEL",
            "bulbul:v3"
        )

    @property
    def sarvam_speaker(self):
        return os.environ.get(
            "SARVAM_SPEAKER",
            "neha"
        )

    @property
    def stt_engine(self):

        if not hasattr(self, "_stt_engine") or self._stt_engine is None:

            api_key = self.openai_api_key
            if self.stt_provider == "huggingface":
                api_key = self.hf_api_key
            elif self.stt_provider == "sarvam":
                api_key = self.sarvam_api_key

            provider = create_stt_provider(
                self.stt_provider,
                api_key=api_key,
                model=self.stt_model,
                endpoint_url=self.stt_endpoint,
                language=self.stt_language,
                debug=self.debug
            )

            processor = AudioProcessor(target_rate=self.stt_sample_rate)

            self._stt_engine = TranscriptionEngine(provider, processor)

        return self._stt_engine

    @property
    def hf_api_key(self):
        return os.environ.get("HF_API_KEY")

    @property
    def openai_api_key(self):
        return os.environ.get("API_KEY")
    
    @property
    def stt_provider(self) -> str:
        return os.environ.get("STT_PROVIDER", "huggingface")

    @property
    def stt_model(self) -> str:
        return os.environ.get("STT_MODEL", "openai/whisper-large-v3")

    @property
    def stt_endpoint(self) -> str | None:
        return os.environ.get("STT_ENDPOINT")

    @property
    def stt_sample_rate(self) -> int:
        return int(os.environ.get("STT_SAMPLE_RATE", "16000"))
    
    @property
    def stt_language(self) -> str:
        return os.environ.get("STT_LANGUAGE", "en")
    
    # TTS CONFIG
# --------------------------------------------------

    @property
    def tts_provider(self):
        return os.environ.get("TTS_PROVIDER", "openai")

    @property
    def tts_model(self):
        return os.environ.get("TTS_MODEL", "gpt-4o-mini-tts")

    @property
    def tts_endpoint(self):
        return os.environ.get("TTS_ENDPOINT")

    @property
    def tts_sample_rate(self):
        return int(os.environ.get("TTS_SAMPLE_RATE", "22050"))

    @property
    def groq_api_key(self):
        return os.environ.get("GROQ_API_KEY")

    @property
    def tts_language(self):
        return os.environ.get("TTS_LANGUAGE", "en-IN")

    @property
    def tts_speaker(self):
        return os.environ.get("TTS_SPEAKER", "anushka")

    @property
    def tts_engine(self):

        if not hasattr(self, "_tts_engine"):

            api_key = self.openai_api_key
            if self.tts_provider == "groq":
                api_key = self.groq_api_key
            elif self.tts_provider == "sarvam":
                api_key = self.sarvam_api_key

            provider = create_tts_provider(
                self.tts_provider,
                api_key=api_key,
                model=self.tts_model,
                endpoint_url=self.tts_endpoint,
                language=self.tts_language,
                speaker=self.tts_speaker,
            )

            processor = AudioProcessor(target_rate=self.tts_sample_rate)

            self._tts_engine = TTSEngine(provider, processor)

        return self._tts_engine

    
    @property
    def jina_api_key(self) -> str | None:
        return os.environ.get("JINA_API_KEY")

    @property
    def jina_api_url(self) -> str:
        return os.environ.get("JINA_BASE_URL","https://api.jina.ai/v1/embeddings")

    @property
    def jina_model(self) -> str:
        return os.environ.get("JINA_MODEL", "jina-embeddings-v3")

    @property
    def jina_dimensions(self) -> int:
        return int(os.environ.get("JINA_DIMENSIONS", "1024"))

    @property
    def opensearch_url(self) -> str | None:
        """Full OpenSearch URL (e.g. ``http://admin:pass@host:9200``).

        When set, it takes precedence over OPENSEARCH_HOST/PORT/USER/PASSWORD.
        """
        return os.environ.get("OPENSEARCH_URL")

    @property
    def opensearch_host(self) -> str:
        url = self.opensearch_url
        if url:
            return _opensearch_url_host(url)
        return os.environ.get("OPENSEARCH_HOST", "localhost")

    @property
    def opensearch_port(self) -> int:
        url = self.opensearch_url
        if url:
            return _opensearch_url_port(url)
        return int(os.environ.get("OPENSEARCH_PORT", "9200"))

    @property
    def opensearch_user(self) -> str:
        url = self.opensearch_url
        if url:
            return _opensearch_url_user(url)
        return os.environ.get("OPENSEARCH_USER", "admin")

    @property
    def opensearch_password(self) -> str:
        url = self.opensearch_url
        if url:
            return _opensearch_url_password(url)
        return os.environ.get("OPENSEARCH_PASSWORD", None)

    @property
    def opensearch_ssl(self) -> bool:
        if self.opensearch_url:
            return self.opensearch_url.startswith("https://")
        return os.environ.get("OPENSEARCH_SSL", "false").lower() == "true"

    @property
    def medical_document_index(self) -> str:
        """Return the OpenSearch index used for medical document retrieval."""
        return os.environ.get(
            "MEDICAL_DOCUMENT_INDEX",
            "clinical_documents",
        )  
    @property
    def opensearch_search_pipeline(self) -> str:
        """Return the OpenSearch hybrid search pipeline."""
        return os.environ.get(
            "OPENSEARCH_SEARCH_PIPELINE",
            "hybrid-rrf-pipeline",
        ) 
     
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
