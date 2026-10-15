"""Assessment & Plan generation routes (AgentType.ASSESSMENT)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from agent.events import AgentType
from api.auth import require_permission
from api.routers.schemas import AssessmentResponse, ConversationRequest, ErrorResponse
from customagents.factory import AgentFactory

router = APIRouter(prefix="/agent")



# Assessment Plan
@router.post(
    "/assessment",
    summary="Medical Assessment and Plan Generation",
    description="""
Generate comprehensive medical assessment and treatment plan from patient conversation.

This endpoint analyzes the complete patient consultation and creates detailed clinical
documentation including differential diagnosis, assessment findings, and recommended
treatment plans for healthcare providers.

**Features:**
- Comprehensive medical assessment generation
- Differential diagnosis analysis
- Treatment plan recommendations
- Clinical decision support
- Evidence-based suggestions

**Authentication:**
Requires valid user session with consultagent:chat permission.
""",
    response_model=AssessmentResponse,
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Failed to generate assessment"
        }
    },
    dependencies=[Depends(require_permission("consultagent", "chat"))]
)
async def generate_assessment_api(request: Request, data: ConversationRequest):

    if not data.conversation:
        return {"error": "Conversation is empty"}

    user_id = request.state.user["sub"]
    config = request.app.state.config
    session = await request.app.state.session_manager.get_session(user_id, config)

    agent = AgentFactory.create(AgentType.ASSESSMENT, config, session)

    try:
        # await agent.session.initialize()

        result = await agent.generate(data.conversation)

        return result

    except Exception as e:
        return {
            "error": "Failed to generate assessment",
            "details": str(e)
        }
