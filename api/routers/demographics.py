"""Patient demographics summary routes (AgentType.DEMOGRAPHICS)."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.events import AgentType
from api.auth import require_permission
from customagents.factory import AgentFactory

AGENT_NAME = AgentType.DEMOGRAPHICS.value

router = APIRouter(prefix="/agent")


class DemographicsRequest(BaseModel):
    message: str
    session_id: str | None = None
    skill: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Summarise the demographics for patient 10008",
                "session_id": "session-123",
                "skill": "patient-demographics-summary",
            }
        }
    }


class ErrorResponse(BaseModel):
    detail: str

    model_config = {
        "json_schema_extra": {"example": {"detail": "Authentication failed"}}
    }


@router.post(
    "/demographics",
    summary="Patient Demographics Summary",
    description="""
Build a structured demographics summary for one patient from FHIR Patient,
RelatedPerson, and Coverage resources.

**Features:**
- Patient identifiers, name, birth date, gender, addresses, contacts, language
- Related persons with relationship and period
- Insurance coverage, when the connected server exposes it
- An explicit "missing or unavailable" section — nothing is invented or inferred

**Behaviour:**
- Uses the FHIR MCP tools, authorized with the caller's token.
- The `patient-demographics-summary` skill is activated by default; pass
  `skill` to override it, or `skill: ""` to disable it.
- Never assumes a contact is an emergency contact, nor that coverage is active.

**Authentication:**
Requires a valid user session with consultagent:chat permission.
""",
    responses={
        200: {"description": "NDJSON streaming response"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    dependencies=[Depends(require_permission("consultagent", "chat"))],
)
async def demographics_api(req: DemographicsRequest, request: Request):

    config = request.app.state.config
    user_id = request.state.user.get("sub")
    session_id = req.session_id or str(uuid.uuid4())

    session_manager = request.app.state.session_manager
    session = await session_manager.get_session(
        user_id=user_id,
        session_id=session_id,
        config=config,
    )

    # The FHIR tools are MCP tools, so the MCP connection is (re)initialized
    # with this request's token. The FHIR server enforces access from it.
    current_token = request.state.token
    session.auth_token = current_token

    await session.initialize()
    await session.mcp_manager.initialize(auth_token=current_token)
    session.mcp_manager.register_tools(session.tool_registry)

    # A demographics summary is a one-shot report over one patient, so the
    # conversation is reset per request. Without this, a client that reuses a
    # session_id would carry the previous patient's details into this turn's
    # context — both a correctness and a privacy problem.
    session.context_manager.clear()

    agent = AgentFactory.create(AgentType.DEMOGRAPHICS, config, session)

    async def event_stream():
        async for event in agent.run(req.message, skill=req.skill):

            event_type = (
                event.type.value
                if hasattr(event.type, "value")
                else str(event.type)
            )

            if event_type == "tool_call_start":
                yield json.dumps({
                    "type": "tool_start",
                    "tool": event.data.get("name"),
                    "arguments": event.data.get("arguments", {}),
                    "agent": AGENT_NAME,
                }) + "\n"

            elif event_type == "tool_call_complete":
                yield json.dumps({
                    "type": "tool_result",
                    "tool": event.data.get("name"),
                    "success": event.data.get("success"),
                    "output": event.data.get("output"),
                    "agent": AGENT_NAME,
                }) + "\n"

            elif event_type == "text_delta":
                yield json.dumps({
                    "type": "text_delta",
                    "message": event.data.get("content", ""),
                    "agent": AGENT_NAME,
                }) + "\n"

            elif event_type == "text_complete":
                yield json.dumps({
                    "type": "text_complete",
                    "message": event.data.get("content", ""),
                    "agent": AGENT_NAME,
                }) + "\n"

            elif event_type == "agent_error":
                yield json.dumps({
                    "type": "error",
                    "message": event.data.get("error", "Unknown error"),
                }) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"X-Session-ID": session_id},
    )
