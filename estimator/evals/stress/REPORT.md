# Baseline CAG: informe de estres

Medicion del CAG de la sesion 5 (`session_05_live`) sin optimizar nada: `MAX_CONVERSATION_TURNS=6`,
anclas heuristicas, summarizer acumulativo, `max_attachment_chars=60000`.

- **Datos:** `evals/stress/results.csv`, 195 turnos (180 de conversacion + 15 de adjuntos), 3 repeticiones,
  `gpt-4o-mini` en actor, extractor y summarizer. Coste total de la campana: 0.36 USD.
- **Reproducir:** `uv run python -m evals.stress.run --http http://127.0.0.1:<puerto> --repeats 3`
  y `uv run python -m evals.stress.report`. Las tablas de abajo salen de ese script, no estan escritas a mano.
- **Metricas** (`evals/stress/metrics.py`, no en `evals/metrics.py`: puntuan el `turn_observed` y el snapshot de
  sesion, no un `EstimationResult`): `LatencyBudgetMetric(10 s)`, `CostBudgetMetric(0.005 USD)` y `MemoryDriftMetric`.
  Los dos presupuestos son supuestos mios; no hay un SLA de cliente.

## Como leer las columnas

- `latency_ms` es el tiempo total del turno (actor + extractor + summarizer + reintentos de Instructor).
  `tokens_in` y `cost_usd` suman todas esas llamadas.
- **recall durable** = el hecho esta en summary, anchors o metadata (definicion del enunciado).
  **recall visible** = ademas cuenta la ventana deslizante, es decir, lo que el LLM realmente tiene delante.
  Con N <= 6 el durable es bajo solo porque los hechos recientes aun viven en la ventana.
- **failed attempts** = reenvios HTTP tras un 502 (el validador de esquema agoto sus reintentos). Cada reenvio se
  cuenta, no se oculta.

### Summary

| scenario | attach KB | turns | P50 ms | P95 ms | cost USD / session | exact hit | semantic hit | recall durable | recall visible | attach recall | failed attempts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| attachments | 0 | 3 | 3532 | 3659 | 0.0005 | 0% | 0% | - | - | - | 0 |
| attachments | 5 | 3 | 9712 | 14183 | 0.0015 | 0% | 0% | - | - | 1.00 | 3 |
| attachments | 20 | 3 | 10854 | 35236 | 0.0059 | 0% | 0% | - | - | 1.00 | 3 |
| attachments | 50 | 3 | 13300 | 17123 | 0.0050 | 0% | 0% | - | - | 1.00 | 0 |
| attachments | 100 | 3 | 13191 | 14397 | 0.0076 | 0% | 0% | - | - | 1.00 | 1 |
| contradiction | 0 | 60 | 11071 | 19378 | 0.0347 | 0% | 0% | 0.82 | 1.00 | - | 13 |
| growing | 0 | 60 | 6425 | 17189 | 0.0228 | 0% | 0% | 0.78 | 1.00 | - | 5 |
| pivot | 0 | 60 | 12131 | 22830 | 0.0410 | 0% | 0% | 0.92 | 0.98 | - | 13 |

### Curve 1 - latency vs tokens_in

| tokens_in (turn total) | n | P50 ms | P95 ms | mean cost USD | llm calls |
|---|---|---|---|---|---|
| 2,000-4,000 | 24 | 3615 | 5809 | 0.00062 | 2.0 |
| 4,000-6,000 | 45 | 6314 | 8987 | 0.00093 | 2.6 |
| 6,000-8,000 | 65 | 11336 | 18634 | 0.00154 | 3.0 |
| 8,000-12,000 | 32 | 12521 | 18268 | 0.00202 | 3.4 |
| 12,000-16,000 | 7 | 15794 | 21023 | 0.00273 | 4.0 |
| >= 16,000 | 22 | 17189 | 32996 | 0.00531 | 4.6 |

### Curve 2 - cumulative cost (USD) vs turn

| scenario | turn 1 | turn 3 | turn 6 | turn 10 | turn 15 | turn 20 | last / first turn cost |
|---|---|---|---|---|---|---|---|
| contradiction | 0.0010 | 0.0042 | 0.0076 | 0.0173 | 0.0261 | 0.0347 | 1.67x |
| growing | 0.0008 | 0.0020 | 0.0054 | 0.0099 | 0.0160 | 0.0228 | 1.79x |
| pivot | 0.0016 | 0.0040 | 0.0095 | 0.0208 | 0.0310 | 0.0410 | 1.07x |

### Curve 3 - fact recall vs N

| scenario | N (turns) | recall durable | recall visible | project name durable | project name visible | stale term in memory |
|---|---|---|---|---|---|---|
| contradiction | 1 | - | - | - | - | 0/3 |
| contradiction | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 |
| contradiction | 6 | 0.56 | 1.00 | 1.00 | 1.00 | 0/3 |
| contradiction | 10 | 0.75 | 1.00 | 1.00 | 1.00 | 3/3 |
| contradiction | 20 | 0.96 | 1.00 | 1.00 | 1.00 | 3/3 |
| growing | 1 | - | - | - | - | 0/3 |
| growing | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 0/3 |
| growing | 6 | 0.67 | 1.00 | 1.00 | 1.00 | 0/3 |
| growing | 10 | 0.78 | 1.00 | 1.00 | 1.00 | 0/3 |
| growing | 20 | 0.77 | 0.96 | 1.00 | 1.00 | 0/3 |
| pivot | 1 | - | - | - | - | 0/3 |
| pivot | 3 | 0.83 | 1.00 | 1.00 | 1.00 | 0/3 |
| pivot | 6 | 0.93 | 1.00 | 1.00 | 1.00 | 1/3 |
| pivot | 10 | 0.85 | 1.00 | 1.00 | 1.00 | 3/3 |
| pivot | 20 | 0.93 | 0.95 | 1.00 | 1.00 | 2/3 |

Completed turns: 195. Aborted sessions (all retries failed): 0. Failed attempts that a retry recovered: 38.

## Lectura: donde empieza a romperse mi CAG y por que

**La latencia y la fiabilidad se rompen mucho antes que la memoria o el coste.** En los turnos sin reintentos, la
latencia P50 pasa de 3.6 s en el turno 1 a 4.2 s en los turnos 4-6, salta a 8.1 s en los turnos 7-10 y llega a
12.2 s en los turnos 16-20 (P95 de 19.5 s). El salto aparece en el turno 7, que es el primero en el que se
expulsa un par de la ventana y el summarizer anade una tercera llamada secuencial al LLM. El tamano del prompt
explica menos: `tokens_in` crece de forma sublineal (2.955 en el turno 1, 7.191 en los turnos 16-20, x2.4) porque
la ventana de 6 turnos acota el contexto, y el coste por turno sube x3.0 (0.00052 a 0.00159 USD). Los `tokens_out`
tambien se duplican (427 a 854), asi que parte de la latencia es el modelo escribiendo estimaciones mas largas. Con
el presupuesto de 10 s, solo el 46 % de los 195 turnos lo cumple (P50 global 10.3 s, P95 20.8 s) y el P50 cruza
los 10 s hacia el turno 11. Con adjuntos pasa lo mismo: P50 de 3.5 s sin adjunto, 9.7 s con 5 KB, 10.9 s con
20 KB y 13.3 s con 50 KB; el cruce de los 10 s esta entre 5 y 20 KB. El coste no es el problema: el peor turno de la
campana cuesta menos de 0.01 USD. La degradacion mas peligrosa es la dura: el 34 % de los turnos de conversacion
(61 de 180) necesito al menos una llamada extra al LLM porque `gpt-4o-mini` no cuadra la suma de las fases con
`total_cost_eur`. En una primera pasada sin reenvios, 13 de 69 peticiones (19 %) terminaron en 502. Con reenvios,
ninguna sesion se aborto, pero 26 turnos necesitaron al menos un reenvio y esos turnos tardan mas (P50 de 12.7 s
frente a 9.0 s sin reintentos). Ese coste no se refleja en `cost_usd`, porque un turno que falla no emite
`turn_observed`.

**La memoria aguanta 20 turnos; lo que falla es que no distingue lo vigente de lo obsoleto, y los adjuntos grandes
se truncan en silencio.** El nombre del proyecto sobrevive al turno 20 en las 9 conversaciones (summary y metadata;
recall durable 1.00 en `growing`, `pivot` y `contradiction`). Una vez fuera de la ventana, el 97.3 % de los hechos
sigue en memoria durable (786 comprobaciones), sin caida visible con la antiguedad. De ese 2.7 % de perdidas,
18 de 21 son un artefacto de mi match exacto: el modelo escribe `route optimization` y el hecho declarado es
`route optimisation`, de modo que el recall visible de `pivot` (0.95 en N=20) esta subestimado. Las 3 perdidas
reales ocurren en una sola repeticion de `growing`: `forecast` (turno 12) y `webhooks` (turno 14) desaparecen del
summary hacia el turno 19-20. Aun asi, el summary de esa repeticion llega a 3.562 caracteres en el turno 20 con un
tope de 4.000 y crece unos 210 por turno desde el turno 10, asi que a este ritmo lo alcanzaria en unos 2 turnos y la compresion
empezaria a perder hechos por saturacion (extrapolacion, no medida). En `contradiction`, el presupuesto viejo
(30000 EUR) queda en anchors en 3/3 repeticiones y en el summary en 1/3; el nuevo (80000 EUR) tambien se ancla en
3/3. Del turno 9 al 13 la memoria durable solo contiene el 30000 (el 80000 sigue en la ventana), y despues contiene
ambos sin ninguna regla de precedencia. En `pivot`, `mentioned_technologies` no se limpia: en la repeticion 2 queda
`['Node.js', 'React', 'Flutter']` del turno 5 al 20, porque el merge es una union. En las otras dos el extractor
nunca registro React. Con adjuntos, el tope de 60.000 caracteres es el limite duro: en el PDF de 100 KB el texto
llega cortado a 60.000, se pierden 2 de los 4 marcadores (Kafka al 65 % y Snowflake al 95 % del documento) y la
respuesta es un 200 sin ningun aviso. El modelo uso el 100 % de lo que recibio, asi que el recall del adjunto
completo cae al 50 % por una decision de arquitectura y no por un error del LLM. Ese es el caso limite que
justifica saltar a RAG: documentos que superan el tope, donde el CAG pierde contenido por construccion, ademas de
pagar 14 veces los tokens de un turno sin adjunto (2.936 a 41.546 con 100 KB). La contradiccion y la obsolescencia
son un problema de politica de memoria y RAG no las resuelve por si solo.

## Limitaciones de la medicion

- `estimate_conversational` no pasa por la cache exacta ni por la semantica, asi que `cache_hit_kind` es siempre
  `none` y los hit rates salen en 0 % por construccion.
- No hay escenario de rafagas: el endpoint es `async def` con llamadas bloqueantes y serializa las peticiones.
- Conversaciones de hasta 20 turnos. No se puede afirmar nada sobre turnos mayores salvo la extrapolacion indicada.
- El ratio "ultimo / primer turno" de la curva 2 esta contaminado por reintentos en el turno 1 (`pivot` sale 1.07x).
  Sin reintentos el turno 20 cuesta x3.0 el turno 1.
- Los 5 de 15 adjuntos con reintentos inflan tokens, latencia y coste de esas filas (un caso llego a 7 llamadas
  y 35 s). Con 3 repeticiones, los P95 de cada fila de adjuntos son indicativos.
- `stale term in memory` busca la subcadena (`React`, `30000`) en memoria durable. Un summary que dice "se abandono
  React" cuenta como presente, asi que es una cota superior de la obsolescencia real.
- Modelo unico (`gpt-4o-mini`). Los modelos `gpt-5.6-luna` y `gpt-6-luna` no funcionan con este wrapper
  (`max_tokens` y function tools con `reasoning_effort`) y no tienen precio en `MODEL_COSTS`.
