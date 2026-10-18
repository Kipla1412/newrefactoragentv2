"""
FastAPI application module
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.auth import decode_token
from api.routers.assessment import router as assessment_router
from api.routers.clinical import router as clinical_router
from api.routers.consult import router as consult_router
from api.routers.doc import router as doc_router
from api.routers.document import router as document_router
from api.routers.ehr import router as ehr_router
from api.routers.pdf import router as pdf_router
from api.routers.soap import router as soap_router
from api.routers.workflow import router as workflow_router
from api.wsrouters.diarize import router as diarize_router
from api.wsrouters.webs2s import router as webs2s_router
from config.application_bootstrap import ApplicationBootstrap, RuntimeState
from config.config import Config
from customagents.factory import AgentFactory
from utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# (router, required agents). A router is mounted only when *all* of its
# required agents are enabled in agents.yaml.
ROUTER_AGENT_REQUIREMENTS: list[tuple[object, tuple[str, ...]]] = [
    (doc_router, ("doc",)),
    (soap_router, ("soap",)),
    (assessment_router, ("assessment",)),
    (clinical_router, ("clinical_extraction",)),
    (workflow_router, ("soap", "assessment", "clinical_extraction")),
    (document_router, ("document_retrieval",)),
    (pdf_router, ("pdf_chat",)),
    (ehr_router, ("ehr",)),
]


def _mount_agent_routers(app: FastAPI, enabled_agents: set[str]) -> None:
    """Mount the agent sub-routers whose required agents are enabled.

    Args:
        app: The FastAPI application.
        enabled_agents: Set of enabled agent names from the bootstrap.
    """
    for router, required in ROUTER_AGENT_REQUIREMENTS:
        if set(required).issubset(enabled_agents):
            app.include_router(router, prefix="/api")
            logger.info("Mounted router (requires %s)", ", ".join(required))
        else:
            logger.info(
                "Skipped router (requires %s; enabled: %s)",
                ", ".join(required),
                ", ".join(sorted(enabled_agents)),
            )


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting Alees AI Agent Runtime")

    # The runtime state is computed once at import (below) so the OpenAPI
    # schema reflects the enabled agents before the server starts. The
    # lifespan only publishes it onto app.state.
    state: RuntimeState = app.state.runtime

    app.state.config = state.config
    app.state.agent_config = state.agent_config
    app.state.model_registry = state.model_registry
    app.state.prompt_registry = state.prompt_registry
    app.state.session_manager = state.session_manager

    logger.info(
        "System initialized successfully — %d agents enabled: %s",
        len(state.enabled_agents),
        ", ".join(state.enabled_agents),
    )

    yield

    logger.info("Shutting down runtime")

# Create FastAPI application
def create_app(state: RuntimeState):

    app = FastAPI(
        title="AI Agent API",
        version="1.0",
        lifespan=lifespan
    )

    app.state.runtime = state

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):

        public_paths = ["/docs", "/openapi.json", "/health", "/ws"]

        if request.url.path.startswith(tuple(public_paths)):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing authentication token"}
            )

        token = auth_header.split(" ")[1]

        try:
            user = decode_token(token)

            request.state.user = user
            request.state.token = token

        except Exception as e:
            logger.warning("JWT middleware decode failed: %s", e)

            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired token"}
            )

        response = await call_next(request)

        return response

    enabled = set(state.enabled_agents)

    # Mount the HTTP agent routers only when their agents are enabled.
    _mount_agent_routers(app, enabled)

    # The consult router exposes both /consult and /intake, so it is
    # mounted when either agent is enabled.
    if {"consult", "intake"} & enabled:
        app.include_router(consult_router, prefix="/api")

    # WebSocket routers — only mount when the backing agent is enabled.
    if {"voice_intake", "voice_consult"} & enabled:
        app.include_router(webs2s_router, prefix="/ws")
    if "diarize" in enabled:
        app.include_router(diarize_router, prefix="/ws")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# Global app instance — bootstrap once at import so the OpenAPI schema
# reflects the enabled agents before the server starts.
_bootstrap = ApplicationBootstrap(
    base_dir=Path.cwd(),
    known_agents=AgentFactory.known_agent_names(),
)
_runtime_state = _bootstrap.run()
app = create_app(_runtime_state)


def get_config(request: Request) -> Config:
    return request.app.state.config