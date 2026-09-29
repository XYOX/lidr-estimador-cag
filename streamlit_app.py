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

st.set_page_config(page_title="Estimador CAG", page_icon="🧮")
st.title("🧮 Estimador de Software (CAG)")
st.caption(
    "Describe el proyecto y elige los parametros de la estimacion. El formulario envia un "
    f"`EstimationRequest` por POST a `{API_BASE_URL}/estimate`."
)

if "history" not in st.session_state:
    st.session_state.history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None

with st.sidebar:
    st.header("Servicio IA")
    st.caption(f"Endpoint: `{API_BASE_URL}/estimate`")

    st.header("Ultima respuesta")
    last = st.session_state.last_response
    if last:
        st.metric("Version de prompt", last.get("prompt_version", "-"))
        elapsed = last.get("elapsed")
        if elapsed is not None:
            st.metric("Tiempo de respuesta", f"{elapsed:.2f} s")
    else:
        st.caption("Aun no se realizo ninguna llamada.")

with st.form("estimation_form"):
    description = st.text_area(
        "Descripcion del proyecto",
        placeholder="Describe el proyecto de software a estimar (minimo 20 caracteres)...",
        height=180,
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
            "description": description,
            "project_type": project_type.value,
            "detail_level": detail_level.value,
            "output_format": output_format.value,
        }
        with st.spinner("Generando estimacion..."):
            start = time.perf_counter()
            try:
                response = requests.post(f"{API_BASE_URL}/estimate", json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as exc:
                st.error(f"Error al llamar al servicio IA: {exc}")
                data = None

        if data:
            elapsed = time.perf_counter() - start
            st.session_state.history.append({"request": payload, "response": data})
            st.session_state.last_response = {**data, "elapsed": elapsed}

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["request"]["description"])
        st.caption(
            f"{PROJECT_TYPE_LABELS[ProjectType(turn['request']['project_type'])]} · "
            f"{DETAIL_LEVEL_LABELS[DetailLevel(turn['request']['detail_level'])]} · "
            f"{OUTPUT_FORMAT_LABELS[OutputFormat(turn['request']['output_format'])]}"
        )
    with st.chat_message("assistant"):
        st.markdown(turn["response"]["text"])
