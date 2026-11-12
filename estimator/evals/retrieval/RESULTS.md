# Retrieval evaluation: hybrid search and reranking (Session 10)

Golden set: 5 queries in `golden_set.json`, relevance annotated by hand at chunk
(budget component) level against the 24 components of `data/budgets_sample.json`.
Latency is end to end per query (OpenAI query embedding + SQL recall + rerank),
in-process, median of 10 runs per query after a warm-up, averaged over the 5 queries.

| Config | Search | Rerank | P@5 | Recall@5 | nDCG@5 | Latency (ms) | of which embed | recall | rerank |
|---|---|---|---|---|---|---|---|---|---|
| A | Vector | No | 0.56 | 1.00 | 1.00 | 350 | 353 | 5 | 0 |
| B | Hybrid (RRF, k=60) | No | 0.56 | 1.00 | 1.00 | 349 | 345 | 9 | 0 |
| C | Vector | Yes | 0.56 | 1.00 | 1.00 | 456 | 361 | 5 | 96 |
| D | Hybrid (RRF, k=60) | Yes | 0.56 | 1.00 | 1.00 | 438 | 350 | 8 | 95 |

Max attainable P@5 on this golden set is 0.56 (queries have 2-4 relevant chunks, so P@5 cannot
reach 1.0). All four configurations hit that ceiling. The embed column is a stage average and
is network-bound, so it varies by a few ms between runs; per-stage figures do not sum exactly
to the total. Raw per-query output: `results.json`.

Reproduce:

```bash
export DATABASE_URL=postgresql+asyncpg://estimator:estimator@localhost:5433/estimator
uv run alembic upgrade head
uv run python -m evals.retrieval.run --runs 10 --out evals/retrieval/results.json
```

## Conclusions

**This measurement does not show a relevance gain from hybrid search or reranking.** All four
configurations tie on P@5, recall@5 and nDCG@5, so the only measured difference is cost: reranking
adds about 90-105 ms (+25-30%) and hybrid search adds about 4 ms. The tie is a property of the
test bed, not evidence that the techniques are useless: the corpus has 24 chunks, `RECALL_K=50`
already covers all of it, and every golden query was answered perfectly by plain vector search,
so there was no headroom to improve. The "payments app returned for an e-commerce query" failure
did not reproduce: the payments chunks (`PAY-002`, `BILL-004`) rank below the three e-commerce
components in every configuration.

For the project I would ship **B (hybrid, no rerank)** as the default and keep the reranker
behind `RERANK_ENABLED`. Hybrid costs almost nothing and covers what embeddings are weakest at,
exact terms such as `JWT`, `SAML` or a framework name, which will matter more as the corpus
grows. Reranking is not justified on this evidence: ~100 ms for zero measured gain, on a request
whose latency is already dominated by the ~350 ms embedding call. Its cost also scales with the
candidate count (about 4 ms per pair here, so ~200 ms for 50 candidates), whereas its benefit
should appear only when the corpus is large enough for look-alike chunks to crowd the top 5.
Before enabling it, re-run this harness on a corpus of hundreds of budgets with adversarial
distractors; if D beats B there by a meaningful nDCG margin, a few hundred milliseconds are
acceptable, since an estimation request is dominated by the LLM generation call anyway.

## Limitations

- 24 chunks and 5 queries: differences smaller than one query's worth (0.2 on P@5) are noise.
- Relevance was annotated by the same person who wrote the queries, with the corpus in view.
- Queries are in English because the dataset is (the exercise brief says Spanish); the text
  search configuration is therefore `english`, not `spanish`.
- Latency includes a network call to OpenAI; absolute numbers depend on connection and region.

## Provenance

- `app/retrieval/reranker.py` is the cross-encoder wrapper from the course support repo
  (`session_10` branch), unmodified.
- Migration, lexical search, RRF fusion, orchestration, endpoint switches, golden set and
  harness were written for this PR.
- While building it, a test showed that hybrid search without rerank recalled only `top_k`
  candidates per branch, which starves RRF; hybrid now always recalls `RECALL_K` per branch.
