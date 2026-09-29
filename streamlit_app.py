import streamlit as st

from app.config import settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services.llm_service import get_system_prompt, stream_estimation

st.set_page_config(page_title="Estimador CAG", page_icon="🧮")
st.title("🧮 Estimador de Software (CAG)")
st.caption(
    "Pega la transcripcion de una reunion con el cliente y recibe una estimacion de desarrollo "
    "generada por el LLM."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

with st.sidebar:
    st.header("Contexto CAG")

    with st.expander("System prompt activo"):
        st.text_area("system_prompt", value=get_system_prompt(), height=200, disabled=True, label_visibility="collapsed")

    with st.expander("Estimaciones de ejemplo (contexto estatico)"):
        for i, example in enumerate(ESTIMATION_EXAMPLES, start=1):
            st.markdown(f"**Ejemplo {i}**")
            st.caption(example["meeting_summary"])
            st.markdown(example["estimation"])
            st.divider()

    st.header("Metricas de la ultima llamada")
    metrics = st.session_state.last_metrics
    if metrics:
        st.metric("Modelo", metrics.get("model", "-"))
        st.metric("Proveedor", metrics.get("provider", "-"))
        col1, col2 = st.columns(2)
        col1.metric("Tokens entrada", metrics.get("input_tokens", "-"))
        col2.metric("Tokens salida", metrics.get("output_tokens", "-"))
        response_time = metrics.get("response_time")
        if response_time is not None:
            st.metric("Tiempo de respuesta", f"{response_time:.2f} s")
    else:
        st.caption("Aun no se realizo ninguna llamada.")

if not settings.openai_api_key and not settings.anthropic_api_key:
    st.warning(
        "No se encontro ninguna API key configurada. Define OPENAI_API_KEY o ANTHROPIC_API_KEY "
        "en tu archivo .env."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if transcription := st.chat_input("Pega aqui la transcripcion de la reunion..."):
    st.session_state.messages.append({"role": "user", "content": transcription})
    with st.chat_message("user"):
        st.markdown(transcription)

    with st.chat_message("assistant"):
        stream = stream_estimation(transcription)
        try:
            estimation = st.write_stream(stream)
        except Exception as exc:
            st.error(f"Error al generar la estimacion: {exc}")
            estimation = None

    if estimation:
        st.session_state.messages.append({"role": "assistant", "content": estimation})
        st.session_state.last_metrics = stream.metrics
        st.rerun()
