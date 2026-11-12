import os
import time

import requests
import streamlit as st

from app.schemas import DetailLevel, OutputFormat, ProjectType

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

PROJECT_TYPE_LABELS = {
    ProjectType.MOBILE_APP: "App movil",
    ProjectType.WEB_SAAS: "Web / SaaS",
    ProjectType.INTERNAL_TOOL: "Herramienta interna",
    ProjectType.DATA_PIPELINE: "Pipeline de datos",
}
DETAIL_LEVEL_LABELS = {
    DetailLevel.SUMMARY: "Resumido",
    DetailLevel.MEDIUM: "Medio",
    DetailLevel.DETAILED: "Detallado",
}
OUTPUT_FORMAT_LABELS = {
    OutputFormat.PHASES_TABLE: "Tabla de fases",
    OutputFormat.LINE_ITEMS: "Lista de tareas",
    OutputFormat.NARRATIVE: "Narrativo",
}

st.set_page_config(page_title="Estimador CAG")
st.title("Estimador de Software (CAG)")
st.caption(
    "Pega la transcripcion, adjunta documentos (PDF / Word) y elige los parametros. Se envia como "
    f"multipart a `{API_BASE_URL}/sessions/{{session_id}}/estimate`."
)



def new_conversation() -> None:
    """Crea una sesion en el servicio y resetea el estado local."""
    response = requests.post(f"{API_BASE_URL}/sessions", timeout=10)
    response.raise_for_status()
    st.session_state.session_id = response.json()["session_id"]
    st.session_state.history = []
    st.session_state.last_response = None
    st.session_state.project_metadata = {}


if "session_id" not in st.session_state:
    try:
        new_conversation()
    except requests.exceptions.RequestException as exc:
        st.error(f"No se pudo crear la sesion en el servicio IA: {exc}")
        st.stop()

with st.sidebar:
    st.header("Servicio IA")
    st.caption(f"Sesion: `{st.session_state.session_id}`")
    if st.button("Nueva conversacion"):
        try:
            new_conversation()
        except requests.exceptions.RequestException as exc:
            st.error(f"No se pudo crear la sesion: {exc}")
        else:
            st.rerun()

    st.header("project_metadata")
    st.caption("Memoria del proyecto: vive aparte del historial y se inyecta en el system prompt.")
    st.json(st.session_state.project_metadata or {})

    st.header("Ultima respuesta")
    last = st.session_state.last_response
    if last:
        st.metric("Turnos en memoria", last.get("turns", "-"))
        st.metric("Version de prompt", last.get("prompt_version", "-"))
        elapsed = last.get("elapsed")
        if elapsed is not None:
            st.metric("Tiempo de respuesta", f"{elapsed:.2f} s")
    else:
        st.caption("Aun no se realizo ninguna llamada.")

with st.form("estimation_form"):
    description = st.text_area(
        "Transcripcion",
        placeholder="Pega la transcripcion de la reunion (minimo 20 caracteres)...",
        height=180,
    )
    uploads = st.file_uploader(
        "Adjuntos (PDF o Word)", type=["pdf", "docx"], accept_multiple_files=True
    )
    project_type = st.selectbox(
        "Tipo de proyecto",
        options=list(ProjectType),
        format_func=lambda value: PROJECT_TYPE_LABELS[value],
    )
    detail_level = st.selectbox(
        "Nivel de detalle",
        options=list(DetailLevel),
        format_func=lambda value: DETAIL_LEVEL_LABELS[value],
    )
    output_format = st.selectbox(
        "Formato de salida",
        options=list(OutputFormat),
        format_func=lambda value: OUTPUT_FORMAT_LABELS[value],
    )
    submitted = st.form_submit_button("Generar estimacion")

if submitted:
    if len(description.strip()) < 20:
        st.error("La descripcion debe tener al menos 20 caracteres.")
    else:
        payload = {
            "transcript": description,
            "project_type": project_type.value,
            "detail_level": detail_level.value,
            "output_format": output_format.value,
        }
        files = [("attachments", (u.name, u.getvalue(), u.type)) for u in uploads]
        with st.spinner("Generando estimacion..."):
            start = time.perf_counter()
            try:
                response = requests.post(
                    f"{API_BASE_URL}/sessions/{st.session_state.session_id}/estimate",
                    data=payload,
                    files=files,
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as exc:
                st.error(f"Error al llamar al servicio IA: {exc}")
                data = None

        if data:
            elapsed = time.perf_counter() - start
            st.session_state.history.append(
                {"request": payload, "attachments": [u.name for u in uploads], "response": data}
            )
            st.session_state.project_metadata = data["project_metadata"]
            st.session_state.last_response = {**data, "elapsed": elapsed}
            st.rerun()  # refresca el panel lateral (el sidebar ya se dibujo con el estado previo)

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["request"]["transcript"])
        if turn["attachments"]:
            st.caption("Adjuntos: " + ", ".join(turn["attachments"]))
        st.caption(
            f"{PROJECT_TYPE_LABELS[ProjectType(turn['request']['project_type'])]} · "
            f"{DETAIL_LEVEL_LABELS[DetailLevel(turn['request']['detail_level'])]} · "
            f"{OUTPUT_FORMAT_LABELS[OutputFormat(turn['request']['output_format'])]}"
        )
    with st.chat_message("assistant"):
        st.markdown(turn["response"]["text"])
