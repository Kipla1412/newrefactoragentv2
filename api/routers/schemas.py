"""Shared request/response schemas for the API routers.

These models are shared across the per-agent sub-routers so each router
reuses them without duplication.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConversationRequest(BaseModel):
    conversation: List[str] = Field(
        ...,
        description="Doctor patient conversation transcript"
    )
    model_config = {
        "json_schema_extra": {
            "example": {
                "conversation": [
                    "Doctor: What brings you in today?",
                    "Patient: I have had a fever for 3 days.",
                    "Doctor: Any cough or breathing difficulty?"
                ]
            }
        }
    }


class ErrorResponse(BaseModel):
    error: str
    details: str | None = None


class SOAPSubjective(BaseModel):
    chief_complaint: str | None = None
    history_of_present_illness: str | None = None
    review_of_systems: List[Any] = Field(default_factory=list)
    past_medical_history: List[Any] = Field(default_factory=list)
    past_surgical_history: List[Any] = Field(default_factory=list)
    family_history: List[Any] = Field(default_factory=list)
    social_history: List[Any] = Field(default_factory=list)
    allergies: Dict[str, Any] = Field(default_factory=dict)
    current_medications: List[Any] = Field(default_factory=list)
    relevant_history: List[Any] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class SOAPObjective(BaseModel):
    vital_signs: List[Any] = Field(default_factory=list)
    physical_examination: List[Any] = Field(default_factory=list)
    investigation_results: List[Any] = Field(default_factory=list)
    other_observations: List[Any] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class SOAPAssessment(BaseModel):
    primary_diagnosis: Any = None
    secondary_diagnoses: List[Any] = Field(default_factory=list)
    active_problems: List[Any] = Field(default_factory=list)
    differential_diagnoses: List[Any] = Field(default_factory=list)
    clinical_reasoning: str | None = None
    uncertainties: List[Any] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class SOAPPlan(BaseModel):
    medications: List[Any] = Field(default_factory=list)
    investigations: List[Any] = Field(default_factory=list)
    procedures: List[Any] = Field(default_factory=list)
    referrals: List[Any] = Field(default_factory=list)
    non_pharmacological_management: List[Any] = Field(default_factory=list)
    counselling: List[Any] = Field(default_factory=list)
    follow_up: List[Any] = Field(default_factory=list)
    safety_netting: List[Any] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class SOAPNote(BaseModel):
    subjective: SOAPSubjective = Field(default_factory=SOAPSubjective)
    objective: SOAPObjective = Field(default_factory=SOAPObjective)
    assessment: SOAPAssessment = Field(default_factory=SOAPAssessment)
    plan: SOAPPlan = Field(default_factory=SOAPPlan)

    model_config = {"extra": "allow"}


class SOAPResponse(BaseModel):
    """Mirrors the JSON contract in customagents/reportagent/soapprompt.py.

    Every field has a default and unknown keys are preserved, so partial or
    specialty-extended output still validates instead of failing the response.
    """

    document_type: str | None = None
    document_version: str | None = None
    patient: Dict[str, Any] = Field(default_factory=dict)
    encounter: Dict[str, Any] = Field(default_factory=dict)
    documentation_metadata: Dict[str, Any] = Field(default_factory=dict)
    soap_note: SOAPNote = Field(default_factory=SOAPNote)
    clinical_actions: List[Dict[str, Any]] = Field(default_factory=list)
    review_flags: List[Dict[str, Any]] = Field(default_factory=list)
    completeness: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "example": {
                "document_type": "SOAP_NOTE",
                "document_version": "1.0",
                "patient": {"patient_id": "10008", "display_name": "Jane Doe"},
                "encounter": {"encounter_id": "enc-1", "documentation_status": "DRAFT"},
                "documentation_metadata": {
                    "source_types": ["consultation_transcript"],
                    "requires_clinician_review": True,
                },
                "soap_note": {
                    "subjective": {
                        "chief_complaint": "Fever and cough",
                        "history_of_present_illness": "Fever and cough for 3 days",
                    },
                    "objective": {"vital_signs": [{"name": "Temperature", "value": "101.2", "unit": "F"}]},
                    "assessment": {"primary_diagnosis": {"display": "Upper Respiratory Tract Infection"}},
                    "plan": {"investigations": [{"name": "CBC"}]},
                },
                "clinical_actions": [
                    {"action_id": "a1", "action_type": "Investigation", "description": "CBC"}
                ],
                "review_flags": [],
                "completeness": {"subjective_complete": True, "missing_sections": []},
            }
        },
    }


class AssessmentResponse(BaseModel):
    clinical_overview: str
    differential_diagnosis: List[Dict[str, Any]]
    diagnostic_plan: Dict[str, Any]
    treatment_plan: List[Dict[str, Any]]
    procedures: str
    risk_level: str
    red_flags: List[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "clinical_overview": "Patient presents with fever, cough, and sore throat for 4 days. Findings suggest mild upper respiratory tract infection.",
                "differential_diagnosis": [
                    {
                        "condition": "Viral Upper Respiratory Tract Infection",
                        "likelihood": "high",
                        "rationale": "Most symptoms are consistent with viral URI."
                    },
                    {
                        "condition": "Bacterial Pharyngitis",
                        "likelihood": "moderate",
                        "rationale": "Fever and sore throat are present."
                    }
                ],
                "diagnostic_plan": {
                    "laboratory_tests": [
                        {
                            "test": "Complete Blood Count",
                            "purpose": "Assess infection severity."
                        }
                    ],
                    "imaging": [
                        {
                            "study": "Chest X-Ray",
                            "purpose": "Rule out pneumonia."
                        }
                    ]
                },
                "treatment_plan": [
                    {
                        "condition": "Upper Respiratory Tract Infection",
                        "recommendation": "Amoxicillin 500 mg",
                        "route": "oral",
                        "duration": "7 days"
                    },
                    {
                        "condition": "Fever",
                        "recommendation": "Paracetamol 650 mg",
                        "route": "oral",
                        "duration": "as needed"
                    }
                ],
                "procedures": "No procedures required",
                "risk_level": "LOW",
                "red_flags": [
                    "Shortness of breath",
                    "Chest pain",
                    "Persistent high fever"
                ]
            }
        }
    }


class ConsultationWorkflowResponse(BaseModel):
    soap: Dict[str, Any]
    assessment: Dict[str, Any]
    clinicalExtraction: Dict[str, Any]

    model_config = {
        "json_schema_extra": {
            "example": {
                "soap": {
                    "subjective": "Fever for 3 days",
                    "objective": "Temperature 101°F"
                },
                "assessment": {
                    "diagnosis": "Viral Fever",
                    "plan": ["Paracetamol", "Hydration"]
                },
                "clinicalExtraction": {
                    "symptoms": ["Fever"],
                    "medications": ["Paracetamol"]
                }
            }
        }
    }


class ClinicalExtractionRequest(BaseModel):

    soap: Dict[str, Any] = Field(
        ...,
        description="SOAP note output"
    )

    assessment: Dict[str, Any] | None = Field(
        default=None,
        description="Assessment agent output"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "soap": {
                    "subjective": {
                        "chief_complaint": "Fever and cough",
                        "history_of_present_illness": "Fever and cough for 3 days"
                    },
                    "objective": {
                        "observations": [
                            "Temperature: 101.2 F"
                        ]
                    },
                    "assessment": {
                        "possible_conditions": [
                            "Upper Respiratory Tract Infection"
                        ]
                    },
                    "plan": {
                        "next_steps": [
                            "CBC",
                            "Chest X-Ray"
                        ]
                    },
                    "summary": "Patient has fever and cough."
                },
                "assessment": {
                    "clinical_overview": "Likely URTI",
                    "differential_diagnosis": [
                        {
                            "condition": "Viral URI",
                            "likelihood": "high",
                            "rationale": "Symptoms are consistent"
                        }
                    ],
                    "diagnostic_plan": {
                        "laboratory_tests": [
                            {
                                "test": "CBC",
                                "purpose": "Evaluate infection"
                            }
                        ]
                    },
                    "treatment_plan": [
                        {
                            "condition": "URI",
                            "recommendation": "Amoxicillin 500 mg",
                            "route": "oral",
                            "duration": "7 days"
                        }
                    ],
                    "risk_level": "LOW"
                }
            }
        }
    }


class ClinicalCondition(BaseModel):
    display: str
    terminologySystem: str


class ClinicalObservation(BaseModel):
    display: str
    terminologySystem: str
    value: str | None = None
    unit: Optional[str] = None


class MedicationRequest(BaseModel):
    display: str
    terminologySystem: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    route: Optional[str] = None


class ServiceRequest(BaseModel):
    display: str
    terminologySystem: str


class ClinicalExtractionResponse(BaseModel):

    conditions: list[ClinicalCondition] = Field(default_factory=list)
    observations: list[ClinicalObservation] = Field(default_factory=list)
    medicationRequests: list[MedicationRequest] = Field(default_factory=list)
    serviceRequests: list[ServiceRequest] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "conditions": [
                    {
                        "display": "Upper Respiratory Tract Infection",
                        "terminologySystem": "SNOMED"
                    }
                ],
                "observations": [
                    {
                        "display": "Temperature",
                        "terminologySystem": "LOINC",
                        "value": "101.2",
                        "unit": "F"
                    }
                ],
                "medicationRequests": [
                    {
                        "display": "Amoxicillin",
                        "terminologySystem": "RXNORM",
                        "dose": "500 mg",
                        "frequency": "Three times daily",
                        "duration": "7 days",
                        "route": "Oral"
                    }
                ],
                "serviceRequests": [
                    {
                        "display": "Complete Blood Count",
                        "terminologySystem": "LOINC"
                    }
                ]
            }
        }
    }


class DoctorAgentResponse(BaseModel):
    questions: List[str]


class GenericAIResponse(BaseModel):
    result: Dict[str, Any]
