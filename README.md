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
│   ├── estimations.py            # Endpoint POST /api/v1/estimate (sin memoria)
│   └── sessions.py               # POST /api/v1/sessions y /sessions/{id}/estimate (multi-turno)
├── sessions.py                   # Session, ConversationHistory (ventana deslizante), ProjectMetadata
├── services/
│   ├── llm_service.py            # Llamada al LLM (OpenAI / Anthropic), con y sin streaming
│   ├── attachments.py            # Extraccion local de texto: PDF (pypdf) y Word (python-docx)
│   └── metadata_extractor.py     # Segunda llamada al LLM que actualiza el project_metadata
└── prompts/
    ├── loader.py                 # render_estimation_prompt(request, version) -> (system, user)
    └── estimation/
        └── v1/
            ├── system.j2         # Rol, instrucciones, bloques condicionales por formato/detalle
            ├── user.j2           # Envuelve la descripcion del proyecto
            └── examples.j2       # Few-shot incluido en system.j2 con {% include %}

streamlit_app.py                  # Cliente: formulario tipado -> POST /api/v1/estimate
tests/prompts/test_estimation_v1.py
tests/sessions/test_sessions_api.py
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

## Sesiones multi-turno

```bash
SID=$(curl -s -X POST localhost:8000/api/v1/sessions | python -c "import sys,json;print(json.load(sys.stdin)['session_id'])")

curl -X POST localhost:8000/api/v1/sessions/$SID/estimate \
  -F "transcript=Reunion con la clinica: app de citas dentales, proyecto Aurora, backend en Django." \
  -F "attachments=@requisitos.pdf" -F "attachments=@alcance.docx"
```

La respuesta es la de `EstimationResponse` (`text`, `prompt_version`) mas `session_id`, `turns` y
`project_metadata`. `project_type`, `detail_level` y `output_format` son campos de formulario
opcionales (defaults `web_saas`, `medium`, `line_items`).

**Memoria.** Todo vive en un diccionario del proceso (`app/sessions.py`): un reinicio o un segundo
worker pierde las sesiones. Es una decision consciente para esta fase (sin BBDD ni Redis).

- *Historial*: ultimos `MAX_TURNS` turnos (default 6, variable `MAX_TURNS`). El mensaje en curso
  cuenta dentro del limite, asi que el LLM nunca recibe mas de `MAX_TURNS` turnos. El system prompt
  no se guarda: se regenera en cada llamada, por lo que siempre esta presente.
- *project_metadata*: `project_name`, `assumed_team_size`, `mentioned_technologies`,
  `agreed_scope`. Se inyecta en `<project_metadata>` del system prompt (vacio en el primer turno) y
  no se ve afectado por la ventana: sobrevive aunque los turnos antiguos se descarten.

**Adjuntos: camino B (extraccion local).** pypdf para PDF y python-docx para `.docx`; el texto se
concatena al transcript con `--- attachment: nombre.pdf ---`. Se eligio porque es independiente del
proveedor (el servicio ya soporta OpenAI y Anthropic), da control sobre el texto que entra al
prompt y deja listo el terreno para el chunking de RAG. Limitaciones: PDFs escaneados (sin capa de
texto) devuelven 422, y no se interpretan imagenes ni layouts complejos. Otros formatos: 415;
mas de 10 MB: 413.

**Extraccion del project_metadata: extractor LLM.** Tras cada respuesta se hace una segunda llamada
que devuelve un JSON con los campos y se fusiona con lo previo (valores nuevos pisan, tecnologias se
acumulan). Se eligio sobre la heuristica por robustez: nombres y alcance en lenguaje libre no se
capturan bien con regex. Costo: una llamada extra por turno (mas latencia). Si el extractor falla o
devuelve JSON invalido, se conserva el metadata anterior y la estimacion no se ve afectada.

## Cliente Streamlit (formulario)

```bash
uv run streamlit run streamlit_app.py
```

Requiere el servicio FastAPI corriendo (por defecto en `http://localhost:8000/api/v1`; configurable
con `API_BASE_URL`). Al cargar crea una sesion (`POST /sessions`) y guarda el `session_id` en
`st.session_state`. El formulario envia transcripcion + adjuntos como multipart; el panel lateral
muestra el `project_metadata` vigente y el boton "Nueva conversacion" crea una sesion nueva.

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

`tests/sessions/test_sessions_api.py` usa `httpx.AsyncClient` y un LLM falso (sin red):

- Dos peticiones en la misma sesion actualizan y conservan el `project_metadata`, y el segundo
  system prompt lo inyecta.
- Un PDF adjunto cambia la estimacion y llega al prompt con su separador.
- 8 turnos con `MAX_TURNS=3`: ninguna llamada al LLM lleva mas de 3 turnos.
- Adjunto no soportado (415) y sesion inexistente (404).
