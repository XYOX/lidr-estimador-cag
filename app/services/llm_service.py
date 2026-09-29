import time

from app.config import settings


def _call_openai(system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content


def _call_anthropic(system: str, user: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=2048,
        system=system,
        messages=[
            {"role": "user", "content": user},
        ],
    )
    return response.content[0].text


def generate_estimation(system: str, user: str) -> dict:
    if settings.llm_provider == "openai":
        estimation = _call_openai(system, user)
        model = settings.openai_model
    elif settings.llm_provider == "anthropic":
        estimation = _call_anthropic(system, user)
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

    def __init__(self, system: str, user: str):
        self.system = system
        self.user = user
        self.metrics: dict = {}

    def __iter__(self):
        start = time.perf_counter()

        if settings.llm_provider == "openai":
            yield from self._stream_openai()
        elif settings.llm_provider == "anthropic":
            yield from self._stream_anthropic()
        else:
            raise ValueError(f"Proveedor de LLM no soportado: {settings.llm_provider}")

        self.metrics["response_time"] = time.perf_counter() - start
        self.metrics["provider"] = settings.llm_provider

    def _stream_openai(self):
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        stream = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
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

    def _stream_anthropic(self):
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.anthropic_api_key)
        with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=2048,
            system=self.system,
            messages=[{"role": "user", "content": self.user}],
        ) as stream:
            yield from stream.text_stream
            final_message = stream.get_final_message()
            self.metrics["input_tokens"] = final_message.usage.input_tokens
            self.metrics["output_tokens"] = final_message.usage.output_tokens

        self.metrics["model"] = settings.anthropic_model


def stream_estimation(system: str, user: str) -> EstimationStream:
    return EstimationStream(system, user)
