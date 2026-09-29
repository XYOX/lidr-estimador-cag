# estimador-cag

Servicio FastAPI que recibe los parametros tipados de un proyecto de software y devuelve una
estimacion de desarrollo generada por un LLM. El prompt vive en templates Jinja2 versionados
(`app/prompts/estimation/v1/`), no como f-string en el codigo.

## Estructura

```
app/
├── main.py                       # App FastAPI, health check, registro de routers
├── config.py                     # Configuracion via variables de entorno (pydantic-settings)
├── schemas.py                    # Contrato tipado: EstimationRequest / EstimationResponse
├── routers/
│   └── estimations.py            # Endpoint POST /api/v1/estimate
├── services/
│   └── llm_service.py            # Llamada al LLM (OpenAI / Anthropic), con y sin streaming
└── prompts/
    ├── loader.py                 # render_estimation_prompt(request, version) -> (system, user)
    └── estimation/
        └── v1/
            ├── system.j2         # Rol, instrucciones, bloques condicionales por formato/detalle
            ├── user.j2           # Envuelve la descripcion del proyecto
            └── examples.j2       # Few-shot incluido en system.j2 con {% include %}

streamlit_app.py                  # Cliente: formulario tipado -> POST /api/v1/estimate
tests/prompts/test_estimation_v1.py
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

## Levantar el servicio

```bash
uv run uvicorn app.main:app --reload
```

Swagger UI: http://localhost:8000/docs

## Probar el endpoint

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Necesitamos una app movil para que pacientes de una clinica dental agenden citas y reciban recordatorios por WhatsApp.",
    "project_type": "mobile_app",
    "detail_level": "summary",
    "output_format": "line_items"
  }'
```

`project_type`: `mobile_app` | `web_saas` | `internal_tool` | `data_pipeline`
`detail_level`: `summary` | `medium` | `detailed`
`output_format`: `phases_table` | `line_items` | `narrative`

Respuesta esperada:

```json
{
  "text": "## Estimacion: ...",
  "prompt_version": "v1"
}
```

## Cliente Streamlit (formulario)

```bash
uv run streamlit run streamlit_app.py
```

Requiere el servicio FastAPI corriendo (por defecto en `http://localhost:8000`; configurable con
`API_BASE_URL`). El formulario arma un `EstimationRequest` y hace `POST /api/v1/estimate` al
servicio. La respuesta se muestra como texto libre; el historial de la sesion queda visible en
pantalla a modo de conversacion.

## Tests

```bash
uv run pytest -v
```

`tests/prompts/test_estimation_v1.py` cubre el render de los templates (no llama a ningun LLM,
corre en milisegundos):

- Que la descripcion del usuario aparece literalmente dentro de `<project_description>`.
- Que el `system` prompt refleja el `output_format` elegido (y no menciona los otros).
- Que el `system` prompt agrega la instruccion de listar asunciones por fase solo cuando
  `detail_level=detailed`.
