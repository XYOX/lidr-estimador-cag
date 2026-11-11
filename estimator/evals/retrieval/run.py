"""Compare the four retrieval configurations against the hand-annotated golden set.

    A = vector            B = hybrid (vector + full-text, RRF)
    C = vector + rerank   D = hybrid + rerank

Reproducible run (Postgres reachable through DATABASE_URL, OPENAI_API_KEY set):

    uv run python -m evals.retrieval.run --runs 5 --out evals/retrieval/results.json

Latency is measured end to end per query (query embedding, SQL recall, rerank), in-process,
after one untimed warm-up pass that loads the cross-encoder. Each query is repeated
``--runs`` times; the per-query latency is the median, the config latency the mean of those.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

from openai import OpenAI

from app.config import get_settings
from app.db.session import get_engine, get_sessionmaker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.retrieval.pipeline import get_reranker, retrieve

GOLDEN_PATH = Path(__file__).with_name("golden_set.json")
TOP_K = 5

CONFIGS = {
    "A": {"label": "Vector", "mode": "vector", "rerank": False},
    "B": {"label": "Hybrid (RRF)", "mode": "hybrid", "rerank": False},
    "C": {"label": "Vector + rerank", "mode": "vector", "rerank": True},
    "D": {"label": "Hybrid + rerank", "mode": "hybrid", "rerank": True},
}


def precision_at_k(retrieved: list[str], relevant: set[str], k: int = TOP_K) -> float:
    return sum(1 for chunk_id in retrieved[:k] if chunk_id in relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = TOP_K) -> float:
    return sum(1 for chunk_id in retrieved[:k] if chunk_id in relevant) / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int = TOP_K) -> float:
    """Binary-relevance nDCG: like P@k but rewards putting relevant chunks higher in the top-k."""
    dcg = sum(1 / math.log2(i + 2) for i, c in enumerate(retrieved[:k]) if c in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


async def run_config(name: str, queries: list[dict], embedder: OpenAIEmbedder, runs: int) -> dict:
    cfg = CONFIGS[name]
    per_query = []
    async with get_sessionmaker()() as session:
        for item in queries:
            relevant = set(item["relevant"])
            latencies, retrieved, stages = [], [], []
            for _ in range(runs):
                started = time.perf_counter()
                result = await retrieve(
                    session, embedder, item["query"], mode=cfg["mode"], rerank=cfg["rerank"],
                    top_k=TOP_K,
                )
                latencies.append((time.perf_counter() - started) * 1000)
                retrieved = [c.metadata["chunk_id"] for c in result.candidates]
                stages.append(result.timings_ms)
            per_query.append(
                {
                    "id": item["id"],
                    "retrieved": retrieved,
                    "precision_at_5": precision_at_k(retrieved, relevant),
                    "recall_at_5": recall_at_k(retrieved, relevant),
                    "ndcg_at_5": ndcg_at_k(retrieved, relevant),
                    "stage_ms": {
                        stage: statistics.mean(t.get(stage, 0) for t in stages)
                        for stage in ("embed", "recall", "rerank")
                    },
                    "latency_ms": statistics.median(latencies),
                }
            )
    return {
        "config": name,
        **cfg,
        "precision_at_5": statistics.mean(q["precision_at_5"] for q in per_query),
        "recall_at_5": statistics.mean(q["recall_at_5"] for q in per_query),
        "ndcg_at_5": statistics.mean(q["ndcg_at_5"] for q in per_query),
        "stage_ms": {
            stage: statistics.mean(q["stage_ms"][stage] for q in per_query)
            for stage in ("embed", "recall", "rerank")
        },
        "latency_ms": statistics.mean(q["latency_ms"] for q in per_query),
        "queries": per_query,
    }


def to_markdown(results: list[dict], max_p5: float) -> str:
    lines = [
        "| Config | Search | Rerank | P@5 | Recall@5 | nDCG@5 | Latency (ms) | embed | recall | rerank |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['config']} | {'Hybrid' if r['mode'] == 'hybrid' else 'Vector'} "
            f"| {'Yes' if r['rerank'] else 'No'} | {r['precision_at_5']:.2f} "
            f"| {r['recall_at_5']:.2f} | {r['ndcg_at_5']:.2f} | {r['latency_ms']:.0f} "
            f"| {r['stage_ms']['embed']:.0f} | {r['stage_ms']['recall']:.0f} "
            f"| {r['stage_ms']['rerank']:.0f} |"
        )
    lines.append(f"\nMax attainable P@5 on this golden set: {max_p5:.2f}")
    return "\n".join(lines)


async def main(runs: int, out: Path | None, configs: list[str]) -> None:
    golden = json.loads(GOLDEN_PATH.read_text())["queries"]
    embedder = OpenAIEmbedder(OpenAI(api_key=get_settings().OPENAI_API_KEY))
    if any(CONFIGS[c]["rerank"] for c in configs):
        get_reranker().load()  # keep model loading out of the timed runs
    results = []
    for name in configs:
        await run_config(name, golden[:1], embedder, 1)  # untimed warm-up (pool, caches)
        results.append(await run_config(name, golden, embedder, runs))
    await get_engine().dispose()

    max_p5 = statistics.mean(min(len(q["relevant"]), TOP_K) / TOP_K for q in golden)
    print(to_markdown(results, max_p5))
    if out:
        out.write_text(json.dumps({"runs": runs, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=5, help="repetitions per query for latency")
    parser.add_argument("--out", type=Path, help="write full per-query results as JSON")
    parser.add_argument("--configs", default="ABCD", help="subset of configurations, e.g. AC")
    args = parser.parse_args()
    asyncio.run(main(args.runs, args.out, list(args.configs)))
