
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
