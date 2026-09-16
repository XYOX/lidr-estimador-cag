from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.llm_service import generate_estimation

router = APIRouter(tags=["estimations"])


class EstimationRequest(BaseModel):
    transcription: str = Field(..., min_length=1, description="Transcripcion de la reunion con el cliente")


class EstimationResponse(BaseModel):
    estimation: str
    model: str
    provider: str


@router.post("/estimate", response_model=EstimationResponse)
def estimate(request: EstimationRequest) -> EstimationResponse:
    try:
        result = generate_estimation(request.transcription)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error al generar la estimacion: {exc}") from exc

    return EstimationResponse(**result)
