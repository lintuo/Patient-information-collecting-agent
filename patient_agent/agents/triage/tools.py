from patient_agent.tools.state_tools import get_case_state, get_image_findings
from patient_agent.tools.triage_tools import save_triage_result

from patient_agent.tools.department_rag_tools import search_departments_for_case

triage_tools = [
    get_case_state,
    get_image_findings,
    search_departments_for_case,
    save_triage_result,
]