# Arquitectura actual del servicio IA y diagnóstico (pre-trabajo Sesión 09)

Estado del repositorio: cierre de Sesión 08 (`fa48abe`). Todas las cifras de este documento salen de ejecuciones reales contra el servicio local (Postgres 16 + pgvector con los 6 presupuestos del seed: 24 chunks, 67–86 tokens cada uno).

**Notas de honestidad sobre el material de partida**

- Los ficheros `examples/transcripts/*.txt` y `TEMPLATE.md` no estaban en el repositorio ni en ninguna rama. Redacté yo las tres transcripciones siguiendo la descripción del enunciado (`01_clear`, `02_ambiguous`, `03_hard`). **Hay que sustituirlas por las oficiales** y repetir el trace; los números de la sección 2 son de mi versión de `02_ambiguous.txt` (4.349 caracteres, 1.200 tokens).
- Los nombres del enunciado (`ingest/`, `storage/`) no coinciden con el código. Equivalencias: `ingest/` es el endpoint `POST /embeddings/ingest` más el chunker (`app/embedding_pipeline/router.py`, `chunker.py`); `storage/` es `app/db/` (SQLAlchemy + pgvector).
- No existe un endpoint HTTP que devuelva el embedding de un texto. El paso 1 del trace usa `OpenAIEmbedder` directamente con un script (`examples/trace_embed.py`).

---

## 1. Arquitectura actual (cierre de Sesión 08)

```mermaid
flowchart LR
  subgraph FE["Frontend"]
    ST["Streamlit<br/>streamlit_app.py<br/>(formulario a /api/v1/estimate)"]
  end

  subgraph BE["Backend de negocio"]
    BN["Consumidor HTTP (Rails u otro)<br/>fuera de este repo"]
  end

  subgraph AI["Servicio IA (FastAPI, estimator/app)"]
    direction TB
    subgraph LEGACY["S02-S06: estimación conversacional (sin retrieval)"]
      EST["POST /api/v1/estimate<br/>POST /sessions/{id}/estimate<br/>POST /sessions/{id}/estimate-acb"]
      SVC["services/: llm_wrapper, boss, critic, cache<br/>sessions/: memoria, compresión, metadata<br/>prompts/estimation v1-v3"]
      EST --> SVC
    end
    subgraph RAGPART["S07-S08: vectorización y búsqueda"]
      ING["ingest: POST /embeddings/ingest<br/>(budget JSON)"]
      CH["embedding_pipeline/chunker.py<br/>1 componente = 1 chunk"]
      EM["embedding_pipeline/embedder.py<br/>text-embedding-3-small, 1536 dim"]
      SRCH["POST /search<br/>{query: texto, k}"]
      ING --> CH --> EM
      SRCH --> EM
    end
    subgraph STORE["storage (app/db)"]
      PG[("PostgreSQL + pgvector<br/>documents, chunks<br/>índice GIN en metadata")]
    end
    EM -- "vectores 1536" --> PG
    SRCH -- "cosine_distance ORDER BY LIMIT k" --> PG
  end

  ST --> EST
  BN --> EST
  BN -.-> SRCH
  SVC --> LLM(("LLM (LiteLLM)"))

  classDef done fill:#dbeafe,stroke:#1d4ed8,color:#000;
  classDef edge fill:#fee2e2,stroke:#b91c1c,stroke-width:3px,color:#000;
  class ING,CH,EM,PG,EST,SVC done;
  class SRCH edge;
```

**Dónde acaba lo implementado.** El flujo termina en `POST /search` (borde rojo): devuelve chunks crudos y distancias. Ningún elemento de la rama "S02-S06" lee `/search` ni la base vectorial (verificado con `grep` sobre `app/services`, `app/routers`, `app/sessions`, `app/prompts`: cero referencias a `embedding_pipeline` ni a `app.db`). Si llegara una transcripción, el servicio podría estimar sin ejemplos históricos o buscar chunks, pero **no hacer ambas cosas a la vez**.

---

## 2. Trace anotado de `02_ambiguous.txt`

Preparación (servicio contra el Postgres del compose, que publica el puerto 5433):

```bash
cd estimator
DATABASE_URL=postgresql+asyncpg://estimator:estimator@localhost:5433/estimator \
  uv run uvicorn app.main:app --port 8010 &
```

### 2.1 Embedding de la transcripción completa

```bash
cd estimator
PYTHONPATH=. uv run python ../examples/trace_embed.py ../examples/transcripts/02_ambiguous.txt
```

```
input_tokens=1200 input_chars=4349
dimension=1536
first=-0.019882 last=0.027252
l2_norm=0.999947
```

**Comentario.** 1.536 dimensiones y norma ≈ 1 (OpenAI normaliza, así que coseno y producto punto coinciden). El vector es un único promedio semántico de todo lo que se dijo: fisioterapia, estética, Instagram, el proveedor de PHP que falló, contabilidad, telemedicina "como idea". Es el centroide de una charla, no de un requisito, y tiene 1.200 tokens frente a chunks de ~70.

### 2.2 Búsqueda semántica con la transcripción completa, k=5

```bash
jq -n --rawfile q examples/transcripts/02_ambiguous.txt '{query: $q, k: 5}' \
 | curl -s -X POST http://localhost:8010/search -H "Content-Type: application/json" -d @-
```

Respuesta (campos `content` y `metadata` recortados; la salida completa se obtiene con el comando anterior):

```json
{"k": 5, "search_time_ms": 951, "results": [
 {"chunk_id": 16, "document_id": 4, "distance": 0.5079, "metadata": {"chunk_id": "BUD-2023-087::BILL-004",  "client_sector": "healthtech", "estimated_hours": 120.0, "complexity": "low"}},
 {"chunk_id": 13, "document_id": 4, "distance": 0.5302, "metadata": {"chunk_id": "BUD-2023-087::SCHED-001", "client_sector": "healthtech", "estimated_hours": 240.0, "complexity": "medium"}},
 {"chunk_id": 14, "document_id": 4, "distance": 0.5312, "metadata": {"chunk_id": "BUD-2023-087::VIDEO-002", "client_sector": "healthtech", "estimated_hours": 320.0, "complexity": "high"}},
 {"chunk_id": 15, "document_id": 4, "distance": 0.5595, "metadata": {"chunk_id": "BUD-2023-087::EHR-003",   "client_sector": "healthtech", "estimated_hours": 280.0, "complexity": "high"}},
 {"chunk_id": 18, "document_id": 5, "distance": 0.6179, "metadata": {"chunk_id": "BUD-2024-031::CART-002",  "client_sector": "retail",     "estimated_hours": 340.0, "complexity": "high"}}
]}
```

### 2.3 Valoración chunk a chunk

Contexto: la petición real del cliente es (1) reserva online multi-sede con recordatorio y confirmación por WhatsApp, (2) paquetes de sesiones prepagados que se descuentan en cualquier sede. Lo demás (Instagram, contabilidad, fichas con fotos, telemedicina) lo declara explícitamente "para más adelante" o "con otro proveedor".

| # | Chunk | Presupuesto / sector | ¿Relevante? |
|---|---|---|---|
| 1 | `BILL-004` Billing (120 h) | BUD-2023-087, healthtech (clínicas privadas, MX) | Parcial. Es facturación por consulta con Stripe; el cliente pide paquetes prepagados con saldo de sesiones, que es otro problema (control de saldo, multi-sede). Aparece primero probablemente por los términos "pagos/boletas" repetidos en la transcripción. |
| 2 | `SCHED-001` Appointment scheduling (240 h) | BUD-2023-087, healthtech | Sí, es el mejor resultado: calendario, disponibilidad y recordatorios. Le faltan multi-sede, selección de terapeuta y WhatsApp, así que 240 h es un piso. |
| 3 | `VIDEO-002` Teleconsultation (320 h) | BUD-2023-087, healthtech | No. Coincide con la mención de telemedicina que el cliente descartó ("lo dejo ahí como idea"). Es ruido que inflaría la estimación 320 h. |
| 4 | `EHR-003` Medical records (280 h) | BUD-2023-087, healthtech | Dudoso. Coincide con las fichas digitalizadas que el cliente pospone ("para más adelante"). Útil como fase 2, no para el alcance inicial. |
| 5 | `CART-002` Checkout (340 h) | BUD-2024-031, retail | Débil. Solo comparte "pagos". Es el primer síntoma de que k=5 ya se agotó la parte útil del corpus. |

**Balance.** El resultado es mejor de lo esperado en el sector (healthtech es el vecino correcto) pero malo como insumo de estimación: 1 chunk claramente útil de 5, 2 que contradicen lo que el cliente quiere, y ninguno cubre los paquetes de sesiones.

### 2.4 Comprobaciones adicionales (usadas en la sección 3)

```bash
# Distribución completa de distancias (k=24 = todo el corpus)
jq -n --rawfile q examples/transcripts/02_ambiguous.txt '{query: $q, k: 24}' \
 | curl -s -X POST localhost:8010/search -H "Content-Type: application/json" -d @- \
 | jq -r '.results[] | "\(.distance)\t\(.metadata.chunk_id)"'
```

- Rango sobre los 24 chunks: **0,508 a 0,723**. Todo el corpus cabe en 0,215 de distancia, y el top-5 en 0,11.
- Solo el párrafo donde el cliente concreta lo que quiere (reservar, WhatsApp, paquetes), como query: `SCHED-001` 0,498 (primero), `BILL-004` 0,520, `VIDEO-002` 0,545, `EHR-003` 0,563, `NOTIF-004` (seguros) 0,696. Mejora el orden, pero sigue trayendo teleconsulta y registros médicos.
- Query solo de "paquetes prepagados de sesiones que se descuentan en cualquier sede": mejor resultado `BILL-004` a **0,686**, y `PAY-002` (transferencias bancarias fintech) en tercer lugar a 0,755. No hay precedente en el corpus y el sistema no lo dice.
- `01_clear.txt`: top-1 `NOTIF-004` 0,423 (seguros, el sector correcto). `03_hard.txt`: top-3 todos de logística (0,520–0,545).
- Consulta fuera de dominio del propio `output_examples.txt` ("mobile application for restaurant reservations"): distancias 0,574–0,641, es decir, el mismo rango que una consulta aceptable.

---

## 3. Diagnóstico: cinco fallos

### Fallo 1: La transcripción entera como query produce un vector difuso y distancias comprimidas
- **Problema observado:** con 1.200 tokens de conversación frente a chunks de ~70, las 24 distancias caen en 0,508–0,723 y el top-5 en 0,11 (sección 2.4). `BILL-004` (facturación) queda primero y `SCHED-001` segundo, aunque la prioridad declarada del cliente es la reserva con recordatorios. Con solo el párrafo del requisito, `SCHED-001` pasa a primero (0,498).
- **Causa probable:** `POST /search` embebe `request.query` tal cual; no existe etapa que separe lo que el cliente pide de lo que comenta. El ruido (Instagram, el proveedor de PHP, contabilidad) mueve el centroide.
- **Propuesta de solución:** una etapa de comprensión de la transcripción que produzca una consulta limpia por requisito antes de buscar, en lugar de embeber la conversación.

### Fallo 2: Un solo vector no puede representar varios requisitos ni distinguir alcance de "ideas"
- **Problema observado:** el top-5 incluye `VIDEO-002` (0,531) y `EHR-003` (0,560), que el cliente pospuso explícitamente, y ningún chunk cubre los paquetes de sesiones. Al buscar ese requisito aislado el mejor candidato está a 0,686 y es irrelevante. En `03_hard` el cliente se contradice (Android/iOS/PWA, rutas sí/no, SAP sí/no) y un vector único los promedia sin resolverlos.
- **Causa probable:** no hay extracción de requisitos con estado (en alcance, fuera de alcance, indeciso, prioridad). `ProjectMetadata` (S05) solo guarda `agreed_scope` libre, nombre, tecnologías y equipo, y no está conectado al retrieval.
- **Propuesta de solución:** extraer una lista estructurada de requisitos con estado y prioridad, y hacer una búsqueda por requisito, marcando como "ambiguo" lo que no esté cerrado para no estimarlo en silencio.

### Fallo 3: Se recuperan componentes sueltos y 4 de los 5 vienen del mismo presupuesto
- **Problema observado:** 4 de 5 resultados son de `BUD-2023-087`; son hermanos que comparten cabecera `[Project][Sector]`. Entran por afinidad de proyecto, no por afinidad de requisito (`VIDEO-002`, `EHR-003`). Cada chunk trae horas propias, pero nada del total (`total_estimated_hours`: 960) ni de qué componentes de ese presupuesto se parecen al pedido y cuáles no.
- **Causa probable:** el chunker incluye cabecera del proyecto en el texto embebido y `/search` hace top-k plano por chunk, sin diversidad, agrupación por presupuesto ni reranking.
- **Propuesta de solución:** una etapa de reranking/agrupación que puntúe por presupuesto y devuelva por cada requisito el mejor componente, con evidencia de su presupuesto de origen.

### Fallo 4: No existe señal de "sin precedente"; la distancia no es interpretable
- **Problema observado:** una consulta fuera de dominio (restaurantes) devuelve 0,574–0,641 y la transcripción ambigua 0,508–0,618; los paquetes prepagados devuelven 0,686 con un chunk de otro dominio. Nada distingue "buen match" de "lo menos malo". `CART-002` (retail) aparece como quinto resultado solo porque k=5 obliga a rellenar.
- **Causa probable:** `/search` siempre devuelve exactamente k filas, sin umbral, sin calibración de distancias contra el corpus y sin evaluación de retrieval (no hay golden set de relevancia).
- **Propuesta de solución:** un umbral calibrado con un conjunto de consultas etiquetadas, y una respuesta explícita de cobertura (cubierto / cubierto parcialmente / sin precedente) por requisito.

### Fallo 5: El retrieval no está conectado a la generación y no filtra por metadata
- **Problema observado:** `/sessions/{id}/estimate` estima sin ejemplos históricos y `/search` devuelve chunks sin que nadie los use; no hay endpoint transcripción → estimación. Además, las búsquedas ignoran sector, año o complejidad, pese a existir el índice GIN sobre `metadata`.
- **Causa probable:** faltan las etapas Augmentation (armar contexto con evidencia y requisitos) y Generation fundamentada (estimar por componente citando presupuestos origen); el filtrado por metadata nunca se expuso en `SearchRequest` (solo `query` y `k`).
- **Propuesta de solución:** una etapa de armado de contexto y una de generación que consuman el retrieval por requisito y produzcan horas por componente con citas y supuestos explícitos, más filtros opcionales de metadata en la búsqueda.

### Otros (no priorizados)
- Corpus mínimo: 6 presupuestos, 24 chunks, ninguno de fisioterapia/estética. Con el corpus actual cualquier estimación de ese sector extrapola desde healthtech.
- Sin endpoint que devuelva embeddings; el trace requirió usar el módulo desde un script.
- Solo búsqueda densa. Términos exactos ("WhatsApp", "Stripe") podrían beneficiarse de búsqueda híbrida (BM25).
- Límite de entrada del modelo de embeddings (8.191 tokens): una reunión de más de ~1 h superaría el límite y `embed_one` fallaría sin troceado ni resumen previo.
- Idioma: transcripción en español contra corpus en inglés. En mis pruebas la query en español no empeoró el ranking (`SCHED-001` 0,498 en ES frente a 0,531 en EN), pero con una muestra tan pequeña no se puede descartar.
- Latencia de `/search`: 359–951 ms en las ejecuciones, dominada por la llamada de embedding sincrónica.

---

## 4. Propuesta de evolución arquitectónica

```mermaid
flowchart LR
  subgraph FE["Frontend"]
    ST["Streamlit / UI de reunión"]
  end
  subgraph BE["Backend de negocio"]
    BN["Consumidor HTTP (Rails u otro)"]
  end

  subgraph AI["Servicio IA"]
    direction TB
    TIN["NUEVO POST /estimate/from-transcript<br/>transcripción -> estimación"]
    subgraph NEWQ["NUEVO: Query"]
      RX["requirement_extractor<br/>requisitos con estado y prioridad"]
      QB["query_builder<br/>1 query limpia por requisito"]
    end
    subgraph NEWR["NUEVO: Retrieval"]
      RET["retriever<br/>filtros de metadata + top-k por requisito"]
      RR["reranker + coverage_check<br/>agrupa por presupuesto, umbral, sin precedente"]
    end
    subgraph NEWA["NUEVO: Augmentation y Generation"]
      CB["context_builder<br/>requisitos + evidencia + supuestos"]
      GEN["grounded_estimator<br/>horas por componente con citas"]
      VAL["output validator<br/>reusa critic/guardrails existentes"]
    end
    subgraph OLD["YA EXISTE (S06-S08)"]
      EM["embedder"]
      PG[("PostgreSQL + pgvector")]
      LLM["llm_wrapper + critic + prompts"]
      SRCH["POST /search"]
    end
    EVAL["NUEVO: retrieval_eval<br/>golden set + umbrales"]

    TIN --> RX --> QB --> RET --> RR --> CB --> GEN --> VAL
    RET --> EM
    RET --> PG
    GEN --> LLM
    EVAL -.calibra.-> RR
  end

  ST --> BN --> TIN
  ST --> TIN

  classDef new fill:#dcfce7,stroke:#15803d,stroke-width:3px,color:#000;
  classDef old fill:#e5e7eb,stroke:#6b7280,color:#000;
  class TIN,RX,QB,RET,RR,CB,GEN,VAL,EVAL new;
  class EM,PG,LLM,SRCH old;
```

Verde con borde grueso = nuevo respecto a la sección 1; gris = ya existe.

**Responsabilidades, flujo y prioridad.** `requirement_extractor` convierte la transcripción en una lista estructurada de requisitos (descripción, estado en-alcance / fuera / indeciso, prioridad) y `query_builder` redacta una consulta limpia por requisito. `retriever` busca cada consulta con filtros de metadata, y `reranker + coverage_check` agrupa por presupuesto, descarta por umbral y declara "sin precedente" donde corresponde. `context_builder` arma el contexto (requisitos + evidencia con horas y presupuesto origen + supuestos), `grounded_estimator` genera horas por componente citando la evidencia, y el validador reutiliza el critic y los guardrails de S03–S06. Los datos que circulan son: transcripción → requisitos JSON → consultas → chunks con distancia → evidencia por requisito con cobertura → prompt con contexto → estimación estructurada con citas. La pieza más crítica es `requirement_extractor`: los fallos 1 y 2 nacen de embeber la conversación cruda, y sin requisitos limpios y con estado ni el retriever, ni el umbral, ni la generación pueden mejorar. Si solo pudiera construir una, atacaría esa y la validaría contra los tres transcripts antes de tocar nada más.
