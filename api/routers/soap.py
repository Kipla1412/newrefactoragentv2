"""SOAP report generation routes (AgentType.SOAP)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from agent.events import AgentType
from api.auth import require_permission
from api.routers.schemas import ConversationRequest, ErrorResponse, SOAPResponse
from customagents.factory import AgentFactory

router = APIRouter(prefix="/agent")



@router.post(
    "/soap",
    summary="SOAP Report Generation",
    description="""
Generate professional clinical notes in SOAP format from doctor-patient conversation.

This endpoint processes the complete conversation and creates structured clinical documentation
following the SOAP (Subjective, Objective, Assessment, Plan) format used in medical practice.

**Features:**
- Standard SOAP format documentation
- Professional medical note generation
- Conversation-to-clinical-notes conversion

**Authentication:**
Requires valid user session with consultagent:chat permission.
""",
    response_model=SOAPResponse,
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Failed to generate SOAP report"
        }
    },
    dependencies=[Depends(require_permission("consultagent", "chat"))]
)
async def generate_soap_api(request: Request, data: ConversationRequest):

    if not data.conversation:
        return {"error": "Conversation is empty"}

    user_id = request.state.user["sub"]
    config = request.app.state.config
    session = await request.app.state.session_manager.get_session(user_id, config)

    agent = AgentFactory.create(AgentType.SOAP, config, session)

    try:
        # await agent.session.initialize()

        result = await agent.generate(data.conversation)

        return result

    except Exception as e:
        return {
            "error": "Failed to generate SOAP report",
            "details": str(e)
        }
