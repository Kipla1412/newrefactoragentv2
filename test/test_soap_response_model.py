"""The SOAP API response must match the JSON the agent prompt produces.

The prompt returns a nested document (``soap_note`` / ``patient`` / ``encounter``
…), while the endpoint previously declared a flat
``{subjective, objective, assessment, plan, summary}`` model — so every real
response failed FastAPI's response validation with a 500.
"""

from api.routers.soap import SOAPResponse


def _contract_payload():
    return {
        "document_type": "SOAP_NOTE",
        "document_version": "1.0",
        "patient": {"patient_id": "10008", "display_name": "Jane Doe"},
        "encounter": {"encounter_id": "enc-1", "documentation_status": "DRAFT"},
        "documentation_metadata": {"source_types": ["consultation_transcript"]},
        "soap_note": {
            "subjective": {
                "chief_complaint": "Fever and cough",
                "history_of_present_illness": "Fever and cough for 3 days",
                "review_of_systems": [{"system": "Respiratory"}],
            },
            "objective": {
                "vital_signs": [{"name": "Temperature", "value": "101.2", "unit": "F"}]
            },
            "assessment": {"primary_diagnosis": {"display": "Viral URI"}},
            "plan": {"investigations": [{"name": "CBC"}]},
        },
        "clinical_actions": [{"action_id": "a1", "action_type": "Investigation"}],
        "review_flags": [{"severity": "low", "category": "documentation"}],
        "completeness": {"subjective_complete": True, "missing_sections": []},
    }


def test_soap_response_accepts_the_agent_contract():
    model = SOAPResponse.model_validate(_contract_payload())

    assert model.document_type == "SOAP_NOTE"
    assert model.soap_note.subjective.chief_complaint == "Fever and cough"
    assert model.soap_note.objective.vital_signs[0]["value"] == "101.2"
    assert model.clinical_actions[0]["action_id"] == "a1"


def test_soap_response_no_longer_requires_a_flat_summary():
    model = SOAPResponse.model_validate(_contract_payload())

    assert not hasattr(model, "summary")


def test_soap_response_tolerates_partial_output():
    model = SOAPResponse.model_validate({"document_type": "SOAP_NOTE"})

    assert model.soap_note.subjective.chief_complaint is None
    assert model.soap_note.plan.medications == []


def test_soap_response_preserves_specialty_extensions():
    payload = _contract_payload()
    payload["specialty"] = {"cardiology": {"ejection_fraction": "55%"}}
    payload["soap_note"]["subjective"]["extra_field"] = "kept"

    dumped = SOAPResponse.model_validate(payload).model_dump()

    assert dumped["specialty"]["cardiology"]["ejection_fraction"] == "55%"
    assert dumped["soap_note"]["subjective"]["extra_field"] == "kept"


def test_soap_response_documents_the_nested_soap_sections():
    schema = SOAPResponse.model_json_schema()

    assert "soap_note" in schema["properties"]
    assert "subjective" in schema["$defs"]["SOAPNote"]["properties"]
    assert "chief_complaint" in schema["$defs"]["SOAPSubjective"]["properties"]