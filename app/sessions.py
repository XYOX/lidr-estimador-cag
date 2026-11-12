"""Estado de sesion en memoria: historial con ventana deslizante + project_metadata.

Todo vive en un diccionario del proceso. Aceptamos esa volatilidad en esta fase: no hay BBDD ni
Redis, un reinicio del servicio (o un segundo worker) pierde las sesiones y el cliente debe crear
una nueva. Persistir la memoria queda fuera del alcance de este modulo.
"""

import threading
import uuid

from pydantic import BaseModel, Field

from app.config import settings


class ProjectMetadata(BaseModel):
    """Hechos del proyecto en curso. Vive aparte del historial y se inyecta en cada system prompt."""

    project_name: str | None = None
    assumed_team_size: int | None = None
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = None

    def is_empty(self) -> bool:
        return not self.model_dump(exclude_defaults=True)

    def merged_with(self, update: "ProjectMetadata") -> "ProjectMetadata":
        """Los valores nuevos no vacios pisan a los previos; las tecnologias se acumulan."""
        technologies = list(self.mentioned_technologies)
        known = {tech.lower() for tech in technologies}
        for tech in update.mentioned_technologies:
            if tech.lower() not in known:
                technologies.append(tech)
                known.add(tech.lower())

        return ProjectMetadata(
            project_name=update.project_name or self.project_name,
            assumed_team_size=update.assumed_team_size or self.assumed_team_size,
            mentioned_technologies=technologies,
            agreed_scope=update.agreed_scope or self.agreed_scope,
        )


class ConversationHistory:
    """Mensajes user/assistant con ventana deslizante de `max_turns` turnos (par user+assistant).

    El system prompt no se guarda aqui: se regenera en cada llamada a partir del project_metadata
    vigente, por lo que esta siempre presente y nunca se descarta.
    """

    def __init__(self, max_turns: int | None = None):
        self.max_turns = max_turns or settings.max_turns
        self._messages: list[dict] = []

    def add_turn(self, user: str, assistant: str) -> None:
        self._messages.extend(
            [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        )
        self._messages = self._messages[-2 * self.max_turns :]

    def to_messages_list(self, system_prompt: str, pending_user: str | None = None) -> list[dict]:
        """Array `messages` listo para el LLM: system + ventana + mensaje en curso.

        Con `pending_user` se reserva un hueco para el, de modo que el total de turnos enviados
        nunca supera `max_turns`.
        """
        keep = self.max_turns - 1 if pending_user is not None else self.max_turns
        window = self._messages[-2 * keep :] if keep > 0 else []
        messages = [{"role": "system", "content": system_prompt}, *window]
        if pending_user is not None:
            messages.append({"role": "user", "content": pending_user})
        return messages

    def __len__(self) -> int:
        return len(self._messages) // 2


class Session:
    def __init__(self, session_id: str, max_turns: int | None = None):
        self.session_id = session_id
        self.history = ConversationHistory(max_turns)
        self.project_metadata = ProjectMetadata()
        self.lock = threading.Lock()  # serializa turnos concurrentes sobre la misma sesion


_sessions: dict[str, Session] = {}


def create_session() -> Session:
    session = Session(str(uuid.uuid4()))
    _sessions[session.session_id] = session
    return session


def get_session(session_id: str) -> Session | None:
    return _sessions.get(session_id)
