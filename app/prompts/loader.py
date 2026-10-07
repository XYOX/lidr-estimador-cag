from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from app.sessions import ProjectMetadata

PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def _get_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_estimation_prompt(request: EstimationRequest, version: str = "v1") -> tuple[str, str]:
    version_dir = PROMPTS_DIR / "estimation" / version
    if not version_dir.is_dir():
        raise ValueError(f"Version de prompt no encontrada: {version}")

    return _render(request.model_dump(mode="json"), version)


def render_session_prompt(
    description: str,
    project_type: ProjectType,
    detail_level: DetailLevel,
    output_format: OutputFormat,
    project_metadata: ProjectMetadata,
    version: str = "v1",
) -> tuple[str, str]:
    """Como render_estimation_prompt, pero sin el limite de 2000 chars de la descripcion
    (transcript + adjuntos) y con el bloque <project_metadata> de la sesion."""
    context = {
        "description": description,
        "project_type": project_type.value,
        "detail_level": detail_level.value,
        "output_format": output_format.value,
        "project_metadata": project_metadata.model_dump(),
    }
    return _render(context, version)


def _render(context: dict, version: str) -> tuple[str, str]:
    context = {"project_metadata": {}, **context}
    env = _get_environment()
    system = env.get_template(f"estimation/{version}/system.j2").render(**context)
    user = env.get_template(f"estimation/{version}/user.j2").render(**context)
    return system, user
