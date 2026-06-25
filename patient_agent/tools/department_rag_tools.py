from patient_agent.services.rag.department_store import search_department_candidates
from patient_agent.storage.repository import get_repository


def search_departments_for_case(case_id: str, k: int = 5) -> list[dict]:
    """Search department triage knowledge for the current patient case.

    Use this tool before making department recommendations.
    It returns candidate departments with retrieved reasons.
    """
    repo = get_repository()
    state = repo.load(case_id)

    return search_department_candidates(state, k=k)