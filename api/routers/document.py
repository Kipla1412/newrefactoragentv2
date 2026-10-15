import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.events import AgentType
from api.auth import require_permission
from customagents.factory import AgentFactory

router = APIRouter(prefix="/agent")



class DocumentRequest(BaseModel):
    message: str = Field(
        ...,
        description="Question to ask about the medical document.",
    )
    patient_id: str = Field(
        ...,
        description="Patient ID used to restrict document retrieval.",
    )
    file_id: str | None = Field(
        default=None,
        description=(
            "Optional file ID. When provided, retrieval is "
            "restricted to this document."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What is the patient's current medical condition?",
                "patient_id": "10008",
                "file_id": "e024e9c5-d5ff-4dbf-8d0c-dd13eb210bd8",
                "session_id": "document-session-123",
            }
        }
    }



class DocumentResponse(BaseModel):
    type: str
    message: str | None = None
    agent: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "text_complete",
                "message": (
                    "The patient is currently managing "
                    "Type 2 Diabetes Mellitus."
                ),
                "agent": "document_retrieval",
            }
        }
    }


class ErrorResponse(BaseModel):
    error: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "Message is empty",
            }
        }
    }


# Document Retrieval API

@router.post(
    "/document",
    summary="Medical Document Retrieval Assistant",
    description="""
Medical document retrieval assistant for querying indexed
clinical documents.

The endpoint retrieves information from the medical document
collection using semantic search.

Retrieval flow:
1. User provides a medical question.
2. Patient ID is used to restrict document retrieval.
3. Optional file ID can further restrict retrieval to one document.
4. jina_embedding generates the query embedding.
5. medical_document_search searches OpenSearch.
6. The retrieved document chunks are provided to the LLM.
7. The agent generates a grounded answer.

The agent must not invent information that is not present
in the retrieved medical documents.

Authentication:
Requires valid user session with documentagent:chat permission.
""",
    responses={
        200: {
            "model": DocumentResponse,
            "description": "NDJSON streaming response",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request",
        },
    },
    dependencies=[
        Depends(require_permission("consultagent", "chat"))
    ],
)
async def document_api(
    request: Request,
    data: DocumentRequest,
):
    # ========================================================
    # 1. Validate message
    # ========================================================
    if not data.message or not data.message.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Message is empty"},
        )

    # 2. Get authenticated user

    user_id = request.state.user["sub"]


    # 3. Get application config
 
    config = request.app.state.config

    # 4. Get shared session

    session = await request.app.state.session_manager.get_session(user_id, config)


    # 5. Store document scope in session

    session.metadata["patient_id"] = data.patient_id

    if data.file_id:
        session.metadata["file_id"] = data.file_id
    else:
        session.metadata.pop("file_id", None)

   
    # 6. Create Document Retrieval Agent

    agent = AgentFactory.create(
        AgentType.DOCUMENT_RETRIEVAL,
        config,
        session,
    )

    # 7. Stream agent events

    async def event_stream():
        try:
            async for event in agent.run(data.message):
                # Convert event into JSON-safe data
                event_data = jsonable_encoder(event)
                yield json.dumps(event_data) + "\n"

        except Exception as e:
            yield json.dumps({
                "type": "error",
                "data": str(e),
                "agent": "document_retrieval",
            }) + "\n"

    # 8. Return NDJSON stream

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "X-Session-ID": session.session_id,
        },
    )