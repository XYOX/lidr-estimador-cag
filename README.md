# estimador-cag

Servicio FastAPI que recibe la transcripcion de una reunion y devuelve una estimacion de software
generada por un LLM, usando arquitectura CAG (contexto estatico inyectado en el prompt: no hay
base de datos, retrieval ni persistencia).

## Estructura

```
app/
├── main.py              # App FastAPI, health check, registro de routers
├── config.py            # Configuracion via variables de entorno (pydantic-settings)
├── routers/
│   └── estimations.py   # Endpoint POST /api/v1/estimate
├── services/
│   └── llm_service.py   # Construccion del prompt y llamada al LLM (OpenAI / Anthropic)
└── context/
    └── examples.py      # Ejemplos de estimaciones previas (few-shot, contexto estatico)
```

## Setup

```bash
uv sync
cp .env.example .env  # completar con tu API key
```

Variables de entorno (`.env`):

- `LLM_PROVIDER`: `openai` o `anthropic`
- `OPENAI_API_KEY`, `OPENAI_MODEL` (default `gpt-4o-mini`)
- `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` (default `claude-haiku-4-5`)

## Levantar el servidor

```bash
uv run uvicorn app.main:app --reload
```

Swagger UI: http://localhost:8000/docs

## Probar el endpoint

La transcripcion de ejemplo que se usa como parametro para este ejercicio esta en
[`transcripcion_ejemplo.txt`](./transcripcion_ejemplo.txt).

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "En la reunion con el equipo de marketing, el cliente explico que necesita una landing page con formulario de contacto, integracion con su CRM actual (HubSpot), y una seccion de blog con editor WYSIWYG. El plazo ideal seria tenerlo listo en 4 semanas. El diseno ya existe en Figma."
  }'
```

Respuesta esperada:

```json
{
  "estimation": "## Estimacion: ...",
  "model": "claude-haiku-4-5",
  "provider": "anthropic"
}
```
