# src/patient_agent/workflow/intake.py

from pathlib import Path
from uuid import uuid4

from patient_agent.domain.state import UploadedFile, ImageJob, PatientCaseState


IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
AUDIO_PREFIX = "audio/"


def save_uploaded_file(
    case_id: str,
    filename: str,
    content_type: str,
    content: bytes,
    upload_root: str = "data/uploads",
) -> UploadedFile:
    file_id = str(uuid4())

    case_dir = Path(upload_root) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    safe_name = filename or f"{file_id}.bin"
    path = case_dir / f"{file_id}-{safe_name}"
    path.write_bytes(content)

    return UploadedFile(
        file_id=file_id,
        path=str(path),
        content_type=content_type,
        original_name=filename,
    )


def attach_file_to_state(
    state: PatientCaseState,
    uploaded: UploadedFile,
) -> PatientCaseState:
    state.uploaded_files.append(uploaded)

    if uploaded.content_type in IMAGE_TYPES:
        state.image_jobs.append(
            ImageJob(
                job_id=str(uuid4()),
                file_id=uploaded.file_id,
                image_path=uploaded.path,
            )
        )

    return state