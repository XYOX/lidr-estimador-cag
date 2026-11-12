from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str = "openai"  # "openai" o "anthropic"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"

    app_name: str = "Estimador CAG"

    # Memoria de sesion: turnos (pares user+assistant) que se conservan en la ventana deslizante
    max_turns: int = 6
    max_attachment_bytes: int = 10 * 1024 * 1024


settings = Settings()
