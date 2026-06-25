from patient_agent.domain.rules import refresh_case_status
from patient_agent.domain.state import TriageResult
from patient_agent.storage.repository import get_repository


def save_triage_result(case_id: str, result: dict) -> dict:
    """Save a structured triage result to the patient case."""
    repo = get_repository()
    state = repo.load(case_id)

    state.triage_result = TriageResult.model_validate(result)
    state = refresh_case_status(state)

    repo.save(state)
    return state.model_dump()