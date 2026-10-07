from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.prompts.loader import render_session_prompt
from app.schemas import (
    DetailLevel,
    OutputFormat,
    ProjectType,
    SessionEstimationResponse,
    SessionResponse,
)
from app.services.attachments import AttachmentError, append_attachments, extract_text
from app.services.llm_service import generate_chat
from app.services.metadata_extractor import extract_metadata
from app.sessions import create_session, get_session

router = APIRouter(prefix="/sessions", tags=["sessions"])

PROMPT_VERSION = "v1"


@router.post("", response_model=SessionResponse, status_code=201)
def new_session() -> SessionResponse:
    return SessionResponse(session_id=create_session().session_id)


@router.post("/{session_id}/estimate", response_model=SessionEstimationResponse)
def estimate_in_session(
    session_id: str,
    transcript: str = Form(min_length=20),
    attachments: list[UploadFile] = File(default=[]),
    project_type: ProjectType = Form(ProjectType.WEB_SAAS),
    detail_level: DetailLevel = Form(DetailLevel.MEDIUM),
    output_format: OutputFormat = Form(OutputFormat.LINE_ITEMS),
) -> SessionEstimationResponse:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")

    extracted = []
    for upload in attachments:
        data = upload.file.read(settings.max_attachment_bytes + 1)
        if len(data) > settings.max_attachment_bytes:
            raise HTTPException(status_code=413, detail=f"'{upload.filename}' supera el tamano maximo")
        try:
            extracted.append((upload.filename, extract_text(upload.filename or "", data)))
        except AttachmentError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    description = append_attachments(transcript, extracted)

    with session.lock:
        system, user = render_session_prompt(
            description,
            project_type,
            detail_level,
            output_format,
            session.project_metadata,
            version=PROMPT_VERSION,
        )
        messages = session.history.to_messages_list(system, pending_user=user)

        try:
            estimation = generate_chat(messages[0]["content"], messages[1:])
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Error al generar la estimacion: {exc}") from exc

        session.history.add_turn(user, estimation)
        session.project_metadata = extract_metadata(session.project_metadata, description, estimation)

        return SessionEstimationResponse(
            text=estimation,
            prompt_version=PROMPT_VERSION,
            session_id=session_id,
            turns=len(session.history),
            project_metadata=session.project_metadata,
        )
