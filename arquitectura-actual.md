# Arquitectura actual del servicio IA y diagnóstico (pre-trabajo Sesión 09)

Estado del repositorio: cierre de Sesión 08 (`fa48abe`). Todas las cifras de este documento salen de ejecuciones reales contra el servicio local (Postgres 16 + pgvector con los 6 presupuestos del seed: 24 chunks, 67–86 tokens cada uno).

**Notas sobre el material de partida**

- Las transcripciones oficiales y `TEMPLATE.md` no estaban en este repositorio; las copié desde `LIDR-academy-temp/ai-engineering` (rama `main`, `examples/transcripts/` y `examples/TEMPLATE.md`). No usé el `arquitectura-actual.md` de ese repositorio: el diagnóstico sale solo de las ejecuciones propias.
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

Transcripción: reunión con Rubén Castaño (Casa Castaño, tienda gourmet física, España). 2.854 caracteres, 890 tokens. Pide, mezclados con divagaciones: tienda online, programa de puntos ("tampoco quiero liarlo mucho al principio"), panel de control ("lo veo clarísimo"), pago con tarjeta ("fundamental"), correo de confirmación de compra, y "decidme un número".

Preparación (servicio contra el Postgres del compose, que publica el puerto 5433):

```bash
cd estimator
DATABASE_URL=postgresql+asyncpg://estimator:estimator@localhost:5433/estimator \
  uv run uvicorn app.main:app --port 8010 &
```

### Paso 1: Embeber la transcripción completa

```bash
cd estimator
PYTHONPATH=. uv run python ../examples/trace_embed.py ../examples/transcripts/02_ambiguous.txt
```

```
input_tokens=890 input_chars=2854
dimension=1536
first=0.006248 last=0.019028
l2_norm=1.000350
```

**Comentario.** 1.536 dimensiones y norma ≈ 1 (OpenAI normaliza, así que coseno y producto punto coinciden). Es un único vector que promedia tienda online, fidelización, panel, pagos, el primo de Francia y el cuaderno de la tienda; es el centroide de una conversación, no de un requisito, y se compara contra chunks de ~70 tokens.

### Paso 2: Búsqueda semántica (top-5)

`POST /search` recibe texto (no vector) y lo embebe internamente.

```bash
jq -n --rawfile q examples/transcripts/02_ambiguous.txt '{query: $q, k: 5}' \
 | curl -s -X POST http://localhost:8010/search -H "Content-Type: application/json" -d @-
```

Respuesta (resumida; `content` completo en la respuesta real, aquí solo componente y descripción):

```json
{"k": 5, "search_time_ms": 723, "results": [
 {"chunk_id": 18, "document_id": 5, "distance": 0.6241, "metadata": {"chunk_id": "BUD-2024-031::CART-002", "client_sector": "retail",     "estimated_hours": 340.0, "complexity": "high"},   "component": "Checkout",         "description": "Cart, address book, multiple payment methods and order confirmation emails."},
 {"chunk_id": 17, "document_id": 5, "distance": 0.6266, "metadata": {"chunk_id": "BUD-2024-031::CAT-001",  "client_sector": "retail",     "estimated_hours": 300.0, "complexity": "medium"}, "component": "Product catalog",  "description": "Searchable catalog with faceted filters, variants and image optimisation."},
 {"chunk_id": 16, "document_id": 4, "distance": 0.6296, "metadata": {"chunk_id": "BUD-2023-087::BILL-004",  "client_sector": "healthtech", "estimated_hours": 120.0, "complexity": "low"},    "component": "Billing",          "description": "Invoicing per consultation with tax receipts and payment links."},
 {"chunk_id": 14, "document_id": 4, "distance": 0.6304, "metadata": {"chunk_id": "BUD-2023-087::VIDEO-002", "client_sector": "healthtech", "estimated_hours": 320.0, "complexity": "high"},   "component": "Teleconsultation", "description": "Browser video calls between doctor and patient with waiting room and session recording consent."},
 {"chunk_id": 20, "document_id": 5, "distance": 0.6477, "metadata": {"chunk_id": "BUD-2024-031::INV-004",  "client_sector": "retail",     "estimated_hours": 440.0, "complexity": "high"},   "component": "Inventory sync",   "description": "Nightly and event-driven stock synchronisation with the warehouse ERP."}
]}
```

### Paso 3: Lectura de los chunks devueltos

| # | Chunk | Presupuesto / sector | ¿Relevante? |
|---|---|---|---|
| 1 | `CART-002` Checkout (340 h) | BUD-2024-031, retail (e-commerce con fidelización) | Sí. Carrito, varios métodos de pago y correos de confirmación de pedido: cubre a la vez dos de sus peticiones (pago con tarjeta y correo al comprar). |
| 2 | `CAT-001` Product catalog (300 h) | BUD-2024-031, retail | Sí. Es la base de "que la gente entre, vea los productos y compre". |
| 3 | `BILL-004` Billing (120 h) | BUD-2023-087, healthtech | No. Facturación por consulta médica; Rubén no habla de facturas. Ruido semántico por "pago". |
| 4 | `VIDEO-002` Teleconsultation (320 h) | BUD-2023-087, healthtech | No, claramente irrelevante: nada en la transcripción sugiere vídeo. Sumaría 320 h inventadas, y queda por encima de la fidelización. |
| 5 | `INV-004` Inventory sync (440 h) | BUD-2024-031, retail | Parcial. Él quiere *ver* el stock en el panel, no sincronizarlo con un ERP de almacén; con 440 h sobreestimaría. |

**Balance.** El sector se acierta (retail es el vecino correcto y su presupuesto aparece 3 veces), pero solo 2 de 5 resultados son claramente útiles. Lo peor es lo que falta: `LOYAL-003` (programa de puntos) sale en la posición 7 (0,657), detrás de la teleconsulta, y lo más parecido a su panel de control, `REPORT-003`, en la 11 (0,705).

### Comprobaciones adicionales (usadas en la sección 3)

Ranking completo (k=24 = todo el corpus) y consultas por requisito, con los mismos comandos cambiando `query`/`k`:

```bash
jq -n --rawfile q examples/transcripts/02_ambiguous.txt '{query: $q, k: 24}' \
 | curl -s -X POST localhost:8010/search -H "Content-Type: application/json" -d @- \
 | jq -r '.results[] | "\(.distance)\t\(.metadata.chunk_id)"'
```

- Con la transcripción entera, las 24 distancias caen en **0,624 a 0,774**; el top-5 completo en 0,624–0,648 (**0,024 de rango**). `VIDEO-002` (irrelevante, 0,6304) y `CART-002` (correcto, 0,6241) están a 0,006 de distancia.
- Consultas escritas a mano, una por requisito: *"Programa de puntos o club de fidelización..."* → `LOYAL-003` primero (0,609); *"Panel de control con pedidos del día, productos más vendidos, stock y gráficas"* → `REPORT-003` (logística) primero (0,647); *"Pago con tarjeta fácil y seguro en el checkout..."* → `CART-002` (0,648); *"Tienda online: catálogo de productos y compra..."* → `CAT-001` (0,660); *"Correo de confirmación al cliente cuando compra"* → `CART-002` (0,697), con `NOTIF-004` (seguros) segundo. Cada requisito aislado recupera el chunk correcto en primer lugar, aunque en español contra un corpus en inglés.
- Referencia con las otras transcripciones: `01_clear` (fintech) top-3 `AUTH-001` 0,400, `PAY-002` 0,418, `UI-004` 0,449; `03_hard` (clínica Vida+) top-3 healthtech, con `VIDEO-002` primero (0,475) y `SCHED-001` segundo (0,509).

---

## 3. Diagnóstico: cinco fallos

### Fallo 1: La transcripción entera como query comprime las distancias y desordena el ranking
- **Problema observado:** con 890 tokens de conversación frente a chunks de ~70, el top-5 cae en 0,024 de rango y `VIDEO-002` (teleconsulta, sin relación con el cliente) queda 4.º, por delante de `LOYAL-003`, que está 7.º. Las mismas peticiones, buscadas una a una, devuelven el chunk correcto en primer lugar (paso 2, comprobaciones).
- **Causa probable:** `POST /search` embebe `request.query` tal cual; no hay etapa que transforme la conversación en consultas de búsqueda. El ruido (cuaderno, sobrina, primo en Francia) arrastra el centroide hacia términos genéricos como "pago" o "cliente".
- **Propuesta de solución:** una etapa previa a la búsqueda que convierta la transcripción en consultas limpias, en lugar de embeber la conversación.

### Fallo 2: Un solo vector no distingue requisitos, ni prioridad, ni "ideas"
- **Problema observado:** la transcripción contiene cinco peticiones con pesos distintos (panel "clarísimo", pago "fundamental", puntos "sin liarlo al principio", Francia "no me hagas caso") y el resultado no refleja ninguna jerarquía. Dos de las cinco (puntos y panel) quedan fuera del top-5 y entra `INV-004` (440 h) por una petición que Rubén no hizo. En `03_hard` el cliente cambia de alcance a mitad de reunión y un vector único lo promedia.
- **Causa probable:** no hay extracción de requisitos con estado. `ProjectMetadata` (S05) solo guarda nombre, equipo, tecnologías y un `agreed_scope` libre, y no está conectado al retrieval.
- **Propuesta de solución:** extraer una lista estructurada de requisitos con estado (en alcance, opcional, descartado, indeciso) y prioridad, y buscar por requisito; lo indeciso se marca, no se estima en silencio.

### Fallo 3: La búsqueda no usa sector ni otros metadatos, y entran chunks de dominios ajenos
- **Problema observado:** para una tienda de productos gourmet entran en el top-5 dos chunks de healthtech (`BILL-004` 0,630 y `VIDEO-002` 0,630), el 40 % de los resultados.
- **Causa probable:** `SearchRequest` solo tiene `query` y `k`; el índice GIN sobre `metadata` (sector, año, complejidad, tecnología) existe pero nunca se consulta, y nadie extrae el sector de la transcripción.
- **Propuesta de solución:** inferir sector y señales (país, tecnología, tamaño) en la etapa de comprensión y exponerlos como filtros o sesgo del retrieval, con una segunda pasada sin filtro cuando el sector no tiene cobertura.

### Fallo 4: La distancia no es interpretable y no hay señal de cobertura
- **Problema observado:** el mejor resultado correcto (0,624, `CART-002`) y el irrelevante (0,630, `VIDEO-002`) son indistinguibles; la misma búsqueda sobre `01_clear` da 0,400. El panel de control, pedido como lo más claro, solo tiene como vecino un reporte de logística (`REPORT-003`, 140 h, "low") y el sistema no avisa de que no hay precedente en retail. Lo mismo con el correo de confirmación (solo existe dentro de `CART-002` y como `NOTIF-004` de seguros).
- **Causa probable:** `/search` devuelve siempre exactamente k filas, sin umbral ni calibración, y no existe un conjunto de consultas etiquetadas que permita saber qué distancia significa "relevante".
- **Propuesta de solución:** un conjunto etiquetado de consultas para calibrar un umbral, y una respuesta de cobertura por requisito (cubierto, parcial, sin precedente), que alimente los supuestos de la estimación.

### Fallo 5: El retrieval no está conectado a la generación y no hay salida numérica con supuestos
- **Problema observado:** Rubén termina con "decidme un número". Hoy no existe un camino transcripción → estimación: `/sessions/{id}/estimate` no lee la base vectorial (grep sin coincidencias en `app/services`, `routers`, `sessions`, `prompts`) y `/search` devuelve chunks que nadie consume.
- **Causa probable:** faltan las etapas Augmentation (armar el contexto con requisitos y evidencia) y Generation fundamentada (horas por componente que citen presupuestos origen).
- **Propuesta de solución:** una etapa que arme el contexto y otra que genere la estimación por componente, con rango, supuestos y citas, reutilizando el critic y los guardrails que ya existen.

### Otros (no priorizados)
- Chunks sueltos sin agregación: 3 de los 5 resultados son hermanos de `BUD-2024-031` y ninguno informa del total del presupuesto (1.340 h) ni de qué componentes cubre el pedido.
- Corpus mínimo: 6 presupuestos, 24 chunks. No hay panel/reportes de retail ni notificaciones por correo como componente propio.
- Sin endpoint que devuelva embeddings; el trace requirió un script que usa el módulo directamente.
- Solo búsqueda densa; términos exactos ("tarjeta", "stock") podrían beneficiarse de búsqueda híbrida.
- Límite de entrada del modelo de embeddings (8.191 tokens): una reunión más larga requeriría troceado o resumen previo; hoy `embed_one` falla.
- Horas sin contexto: los presupuestos históricos son de ES, MX, PE y CO; no hay ajuste por país, equipo o año.
- Idioma: la query en español contra corpus en inglés funcionó bien en las pruebas por requisito, así que no lo considero un fallo con esta muestra.
- Latencia: 359–951 ms por búsqueda, dominada por la llamada síncrona de embedding.

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
