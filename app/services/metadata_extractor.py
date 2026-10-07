import json
import re

from pydantic import ValidationError

from app.services.llm_service import generate_chat
from app.sessions import ProjectMetadata

EXTRACTOR_SYSTEM = """Eres un extractor de hechos de proyectos de software. Recibes los hechos ya conocidos \
del proyecto y la ultima interaccion (mensaje del usuario y estimacion del asistente). Devuelve \
unicamente un objeto JSON con estas claves:
- project_name: nombre del proyecto (string) o null si no se menciona
- assumed_team_size: tamano del equipo asumido o recomendado (entero) o null
- mentioned_technologies: lista de tecnologias mencionadas (strings), [] si no hay
- agreed_scope: resumen breve del alcance acordado (string) o null
Incluye solo lo que aparece en la interaccion o en los hechos conocidos; no inventes. Sin texto \
adicional ni bloques de codigo."""

MAX_INTERACTION_CHARS = 8000


def extract_metadata(
    current: ProjectMetadata, user_message: str, assistant_message: str
) -> ProjectMetadata:
    """Segunda llamada al LLM que actualiza el project_metadata. Si falla, conserva el actual."""
    prompt = (
        f"Hechos conocidos:\n{current.model_dump_json()}\n\n"
        f"<user_message>\n{user_message[:MAX_INTERACTION_CHARS]}\n</user_message>\n\n"
        f"<assistant_message>\n{assistant_message[:MAX_INTERACTION_CHARS]}\n</assistant_message>"
    )
    try:
        raw = generate_chat(EXTRACTOR_SYSTEM, [{"role": "user", "content": prompt}])
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        update = ProjectMetadata.model_validate(json.loads(match.group(0)))
    except (ValidationError, ValueError, AttributeError, TypeError):
        return current
    except Exception:  # el fallo del extractor no debe tumbar una estimacion ya generada
        return current

    return current.merged_with(update)
