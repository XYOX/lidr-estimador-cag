from fastapi import FastAPI

from app.config import settings
from app.routers import estimations

app = FastAPI(
    title=settings.app_name,
    description=(
        "Servicio que genera estimaciones de software a partir de la transcripcion de una reunion, "
        "usando arquitectura CAG (contexto estatico inyectado en el prompt)."
    ),
    version="0.1.0",
)

app.include_router(estimations.router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
