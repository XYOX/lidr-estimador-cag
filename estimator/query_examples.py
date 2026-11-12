"""Five representative semantic queries against POST /search.

    docker compose run --rm estimator python query_examples.py
"""

from __future__ import annotations

import os
import sys

import httpx

BASE_URL = os.environ.get("ESTIMATOR_API_BASE_URL", "http://estimator:8000")
K = 5
PREVIEW_CHARS = 120

QUERIES = (
    ("Direct known component (sanity check)",
     "REST API development with JWT authentication for financial sector"),
    ("Semantic reformulation (different vocabulary)",
     "secure backend service with token-based access control for banking applications"),
    ("Out-of-domain (should not be in the corpus)",
     "mobile application for restaurant reservations"),
    ("Ambiguous (short and generic)",
     "integration with external system"),
    ("Very specific (precise technical vocabulary)",
     "migration from monolith to microservices architecture using Kubernetes"),
)


def preview(content: str) -> str:
    # Chunks start with a "[Project] [Sector]" header shared by every component of a budget;
    # skip it so the preview shows what distinguishes the chunk.
    body = content.split("\n\n", 1)[-1]
    flat = " ".join(body.split())
    return flat[:PREVIEW_CHARS] + ("..." if len(flat) > PREVIEW_CHARS else "")


def main() -> int:
    with httpx.Client(base_url=BASE_URL, timeout=60) as client:
        for number, (label, query) in enumerate(QUERIES, start=1):
            response = client.post("/search", json={"query": query, "k": K})
            response.raise_for_status()
            body = response.json()
            print(f"\n=== Query {number}: {label}")
            print(f'"{query}"  (k={body["k"]}, {body["search_time_ms"]} ms)')
            for rank, hit in enumerate(body["results"], start=1):
                print(
                    f'  {rank}. chunk_id={hit["chunk_id"]:<5} distance={hit["distance"]:.4f} '
                    f'type={hit["chunk_type"]}'
                )
                print(f"     {preview(hit['content'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
