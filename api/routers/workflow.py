"""Composite consultation workflow route.

Runs SOAP + Assessment + Clinical Extraction as a single pipeline.
Only mounted when all three agents are enabled.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from agent.events import AgentType
from api.auth import require_permission
from api.routers.schemas import (
    ConsultationWorkflowResponse,
    ConversationRequest,
    ErrorResponse,
)
from customagents.factory import AgentFactory

router = APIRouter(prefix="/agent")


# Agents required for this composite workflow to be mounted.
REQUIRED_AGENTS = (
    AgentType.SOAP.value,
    AgentType.ASSESSMENT.value,
    AgentType.CLINICAL_EXTRACTION.value,
)


@router.post(
    "/consultation-workflow",
    summary="SOAP + Assessment + Clinical Extraction Workflow",
    description="""
Runs the complete clinical workflow:

1. SOAP Generation
2. Assessment & Plan Generation
3. Clinical Concept Extraction

Returns all outputs in a single response.
""",
    response_model=ConsultationWorkflowResponse,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Conversation is empty"
        },
        500: {
            "model": ErrorResponse,
            "description": "Workflow execution failed"
        }
    },
    dependencies=[Depends(require_permission("consultagent", "chat"))]
)
async def consultation_workflow(
    request: Request,
    data: ConversationRequest
):

    if not data.conversation:
        raise HTTPException(
            status_code=400,
            detail="Conversation is empty"
        )

    user_id = request.state.user["sub"]

    config = request.app.state.config

    session = await request.app.state.session_manager.get_session(
        user_id,
        config
    )

    try:
        # SOAP AGENT
        soap_agent = AgentFactory.create(
            AgentType.SOAP,
            config,
            session
        )

        soap_result = await soap_agent.generate(
            data.conversation
        )

        # ASSESSMENT AGENT
        assessment_agent = AgentFactory.create(
            AgentType.ASSESSMENT,
            config,
            session
        )

        assessment_result = await assessment_agent.generate(
            data.conversation
        )

        # CLINICAL EXTRACTION AGENT
        clinical_agent = AgentFactory.create(
            AgentType.CLINICAL_EXTRACTION,
            config,
            session
        )

        clinical_result = await clinical_agent.generate(
            soap_note=soap_result,
            assessment_report=assessment_result
        )

        return {
            "soap": soap_result,
            "assessment": assessment_result,
            "clinicalExtraction": clinical_result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Consultation workflow failed: {str(e)}"
        )
