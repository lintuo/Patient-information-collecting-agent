from patient_agent.reports.renderer import render_report
from patient_agent.storage.repository import get_repository
from patient_agent.domain.rules import refresh_case_status


def render_patient_report(case_id: str) -> dict:
    """Render a patient report file for the case."""
    repo = get_repository()
    state = repo.load(case_id)

    report_path = render_report(state)
    state.report_path = report_path
    state = refresh_case_status(state)

    repo.save(state)

    return {
        "case_id": case_id,
        "report_path": report_path,
    }