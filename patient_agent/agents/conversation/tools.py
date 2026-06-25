from patient_agent.tools.state_tools import (
    analyze_uploaded_images,
    apply_patient_patch,
    get_case_state,
    get_image_findings,
    get_missing_fields,
)


conversation_tools = [
    get_case_state,
    get_missing_fields,
    get_image_findings,
    apply_patient_patch,
    analyze_uploaded_images,
]