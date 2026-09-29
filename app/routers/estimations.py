from fastapi import APIRouter, HTTPException

from app.prompts.loader import render_estimation_prompt
from app.schemas import EstimationRequest, EstimationResponse
from app.services.llm_service import generate_estimation

router = APIRouter(tags=["estimations"])

PROMPT_VERSION = "v1"


@router.post("/estimate", response_model=EstimationResponse)
def estimate(request: EstimationRequest) -> EstimationResponse:
    system, user = render_estimation_prompt(request, version=PROMPT_VERSION)

    try:
        result = generate_estimation(system, user)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error al generar la estimacion: {exc}") from exc

    return EstimationResponse(text=result["estimation"], prompt_version=PROMPT_VERSION)
