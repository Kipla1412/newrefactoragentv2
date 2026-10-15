"""Clinical concept extraction routes (AgentType.CLINICAL_EXTRACTION)."""

from __future__ import annotations

import traceback

from fastapi import APIRouter, Depends, HTTPException, Request

from agent.events import AgentType
from api.auth import require_permission
from api.routers.schemas import (
    ClinicalExtractionRequest,
    ClinicalExtractionResponse,
)
from customagents.factory import AgentFactory

router = APIRouter(prefix="/agent")



@router.post(
    "/clinical-extraction",
    summary="Clinical Concept Extraction",
    response_model=ClinicalExtractionResponse,
    responses={
        500: {
            "model": dict,
            "description": "Clinical extraction failed"
        }
    },
    dependencies=[
        Depends(
            require_permission(
                "consultagent",
                "chat"
            )
        )
    ]
)
async def generate_clinical_extraction(
    request: Request,
    data: ClinicalExtractionRequest
):

    user_id = request.state.user["sub"]

    config = request.app.state.config

    session = await request.app.state.session_manager.get_session(
        user_id,
        config
    )

    try:

        agent = AgentFactory.create(
            AgentType.CLINICAL_EXTRACTION,
            config,
            session
        )

        result = await agent.generate(
            soap_note=data.soap,
            assessment_report=data.assessment
        )

        return ClinicalExtractionResponse(
            conditions=result.get("conditions") or [],
            observations=result.get("observations") or [],
            medicationRequests=result.get("medicationRequests") or [],
            serviceRequests=result.get("serviceRequests") or [],
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
