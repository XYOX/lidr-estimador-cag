from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas import EstimationRequest

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

    env = _get_environment()
    context = request.model_dump(mode="json")

    system = env.get_template(f"estimation/{version}/system.j2").render(**context)
    user = env.get_template(f"estimation/{version}/user.j2").render(**context)

    return system, user
