import json

import httpx
import pytest

from app.config import settings
from app.main import app

TRANSCRIPT = "Reunion con la clinica: quieren una app para agendar citas dentales con recordatorios."


def make_pdf(text: str) -> bytes:
    """PDF minimo de una pagina con `text`; pypdf reconstruye el xref al leerlo."""
    stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref)
    return out


class FakeLLM:
    """Reemplaza al LLM: registra cada llamada y responde segun el prompt recibido."""

    def __init__(self):
        self.estimation_calls: list[tuple[str, list[dict]]] = []

    def __call__(self, system: str, messages: list[dict]) -> str:
        if system.startswith("Eres un extractor"):
            content = messages[0]["content"]
            update = {}
            if "Proyecto Aurora" in content:
                update["project_name"] = "Aurora"
                update["mentioned_technologies"] = ["React"]
            if "Django" in content:
                update["mentioned_technologies"] = ["Django"]
            return json.dumps(update)

        self.estimation_calls.append((system, messages))
        last = messages[-1]["content"]
        return "Total estimado: 200 horas" if "Redis" in last else "Total estimado: 100 horas"


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr("app.routers.sessions.generate_chat", fake)
    monkeypatch.setattr("app.services.metadata_extractor.generate_chat", fake)
    return fake


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
        yield c


async def new_session(client) -> str:
    response = await client.post("/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


async def estimate(client, session_id, transcript=TRANSCRIPT, files=None):
    return await client.post(
        f"/sessions/{session_id}/estimate", data={"transcript": transcript}, files=files
    )


async def test_metadata_is_updated_across_turns(client, llm):
    session_id = await new_session(client)

    first = await estimate(client, session_id, TRANSCRIPT + " Proyecto Aurora con React.")
    assert first.status_code == 200
    assert first.json()["project_metadata"]["project_name"] == "Aurora"
    # Primera llamada: el bloque <project_metadata> va vacio
    assert "<project_metadata>\n</project_metadata>" in llm.estimation_calls[0][0]

    second = await estimate(client, session_id, TRANSCRIPT + " Ahora el backend sera Django.")
    metadata = second.json()["project_metadata"]
    assert metadata["project_name"] == "Aurora"  # se conserva entre turnos
    assert metadata["mentioned_technologies"] == ["React", "Django"]
    # Segunda llamada: el system prompt ya inyecta lo aprendido y se envia el historial previo
    system, messages = llm.estimation_calls[1]
    assert "Nombre del proyecto: Aurora" in system
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]


async def test_pdf_attachment_influences_estimation(client, llm):
    session_id = await new_session(client)
    without = await estimate(client, session_id)

    pdf = make_pdf("Requisito: cache con Redis")
    with_pdf = await estimate(
        client, session_id, files=[("attachments", ("spec.pdf", pdf, "application/pdf"))]
    )

    assert "100 horas" in without.json()["text"]
    assert "200 horas" in with_pdf.json()["text"]
    last_user = llm.estimation_calls[-1][1][-1]["content"]
    assert "--- attachment: spec.pdf ---" in last_user


async def test_unsupported_attachment_and_unknown_session(client, llm):
    session_id = await new_session(client)
    bad = await estimate(client, session_id, files=[("attachments", ("a.txt", b"hola", "text/plain"))])
    assert bad.status_code == 415

    missing = await estimate(client, "no-existe")
    assert missing.status_code == 404


async def test_history_never_exceeds_max_turns(client, llm, monkeypatch):
    monkeypatch.setattr(settings, "max_turns", 3)
    session_id = await new_session(client)

    for i in range(8):
        response = await estimate(client, session_id, f"{TRANSCRIPT} Turno numero {i}.")
        assert response.status_code == 200

    for _, messages in llm.estimation_calls:
        assert sum(m["role"] == "user" for m in messages) <= 3
    # El historial efectivo conserva los turnos mas recientes
    assert "Turno numero 7" in llm.estimation_calls[-1][1][-1]["content"]
    assert "Turno numero 5" in llm.estimation_calls[-1][1][0]["content"]
    assert response.json()["turns"] == 3
