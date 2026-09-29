import time

from app.config import settings
from app.context.examples import ESTIMATION_EXAMPLES


def get_system_prompt() -> str:
    # API publica para consumidores externos (ej. streamlit_app.py); _build_system_prompt
    # queda como detalle interno del modulo.
    return _build_system_prompt()


def _build_system_prompt() -> str:
    examples_text = "\n\n".join(
        f"--- Ejemplo {i} ---\n"
        f"Resumen de la reunion:\n{example['meeting_summary']}\n\n"
        f"Estimacion generada:\n{example['estimation']}"
        for i, example in enumerate(ESTIMATION_EXAMPLES, start=1)
    )

    return f"""Eres un estimador de software experto. Tu trabajo es leer la transcripcion de una \
reunion con un cliente y generar una estimacion de desarrollo de software: desglose de tareas, \
horas por tarea, equipo recomendado y duracion estimada.

Usa como referencia el estilo y nivel de detalle de las siguientes estimaciones previas que ya \
fueron aprobadas por el equipo:

{examples_text}

Genera la nueva estimacion siguiendo el mismo formato (encabezado con el nombre del proyecto, \
desglose de tareas numerado con horas, total de horas, equipo recomendado y duracion estimada). \
Responde unicamente con la estimacion en formato Markdown, sin comentarios adicionales."""


def _call_openai(system_prompt: str, transcription: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcription},
        ],
    )
    return response.choices[0].message.content


def _call_anthropic(system_prompt: str, transcription: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=2048,
        system=system_prompt,
        messages=[
            {"role": "user", "content": transcription},
        ],
    )
    return response.content[0].text


def generate_estimation(transcription: str) -> dict:
    system_prompt = _build_system_prompt()

    if settings.llm_provider == "openai":
        estimation = _call_openai(system_prompt, transcription)
        model = settings.openai_model
    elif settings.llm_provider == "anthropic":
        estimation = _call_anthropic(system_prompt, transcription)
        model = settings.anthropic_model
    else:
        raise ValueError(f"Proveedor de LLM no soportado: {settings.llm_provider}")

    return {
        "estimation": estimation,
        "model": model,
        "provider": settings.llm_provider,
    }


class EstimationStream:
    """Genera una estimacion en streaming y expone metricas de la llamada una vez consumida."""

    def __init__(self, transcription: str):
        self.transcription = transcription
        self.metrics: dict = {}

    def __iter__(self):
        system_prompt = _build_system_prompt()
        start = time.perf_counter()

        if settings.llm_provider == "openai":
            yield from self._stream_openai(system_prompt)
        elif settings.llm_provider == "anthropic":
            yield from self._stream_anthropic(system_prompt)
        else:
            raise ValueError(f"Proveedor de LLM no soportado: {settings.llm_provider}")

        self.metrics["response_time"] = time.perf_counter() - start
        self.metrics["provider"] = settings.llm_provider

    def _stream_openai(self, system_prompt: str):
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        stream = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.transcription},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            if chunk.usage is not None:
                self.metrics["input_tokens"] = chunk.usage.prompt_tokens
                self.metrics["output_tokens"] = chunk.usage.completion_tokens
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

        self.metrics["model"] = settings.openai_model

    def _stream_anthropic(self, system_prompt: str):
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.anthropic_api_key)
        with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": self.transcription}],
        ) as stream:
            yield from stream.text_stream
            final_message = stream.get_final_message()
            self.metrics["input_tokens"] = final_message.usage.input_tokens
            self.metrics["output_tokens"] = final_message.usage.output_tokens

        self.metrics["model"] = settings.anthropic_model


def stream_estimation(transcription: str) -> EstimationStream:
    return EstimationStream(transcription)
