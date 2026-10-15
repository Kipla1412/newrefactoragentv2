"""Doctor assistant agent routes (AgentType.DOC)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from agent.events import AgentType
from api.auth import require_permission
from api.routers.schemas import ConversationRequest, DoctorAgentResponse, ErrorResponse
from customagents.factory import AgentFactory

router = APIRouter(prefix="/agent")



@router.post(
    "/doctoragent",
    summary="Doctor Assistant - Medical Question Recommendations",
    description="""
Generate intelligent medical question recommendations for doctors based on patient conversation.

This endpoint analyzes the conversation history and provides targeted follow-up questions
that help doctors gather essential patient information for better medical care.

**Features:**
- Medical question recommendations based on symptoms
- Targeted follow-up questions for clinical assessment
- Context-aware conversation analysis

**Authentication:**
Requires valid user session with consultagent:chat permission.
""",
    response_model=DoctorAgentResponse,
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Failed to generate recommendations"
        }
    },
    dependencies=[Depends(require_permission("consultagent", "chat"))]
)
async def recommend_questions_api(request: Request, data: ConversationRequest):

    if not data.conversation:
        return {"questions": [], "status": "empty_input"}

    user_id = request.state.user["sub"]
    config = request.app.state.config
    session = await request.app.state.session_manager.get_session(user_id, config)

    agent = AgentFactory.create(AgentType.DOC, config, session)

    try:
        # await agent.session.initialize()

        result = await agent.recommend_questions(data.conversation)

        return result

    except Exception as e:
        return {
            "questions": [],
            "error": "Failed to generate recommendations",
            "details": str(e)
        }
