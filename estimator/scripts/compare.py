"""Cosine similarity between texts, as a sanity check of the embeddings.

    python scripts/compare.py --text-a "OAuth 2.0 authentication backend for fintech" \
                              --text-b "JWT-based authorization service for banking app"
    python scripts/compare.py --sanity     # three built-in pairs: close, related, unrelated
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `app` importable

from openai import OpenAI  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402

AUTH_FINTECH = "OAuth 2.0 authentication backend for fintech"

SANITY_PAIRS = (
    ("close", AUTH_FINTECH, "JWT-based authorization service for banking app"),
    ("related", "React dashboard with sales charts", "Admin panel with analytics widgets built in Vue"),
    ("unrelated", AUTH_FINTECH, "Recipe for chocolate cake with strawberries"),
)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm


def _print_pair(embedder: OpenAIEmbedder, text_a: str, text_b: str) -> float:
    similarity = cosine_similarity(embedder.embed_one(text_a), embedder.embed_one(text_b))
    print(f"Text A: {text_a}")
    print(f"Text B: {text_b}")
    print(f"Cosine similarity: {similarity:.4f}")
    return similarity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--text-a")
    parser.add_argument("--text-b")
    parser.add_argument("--sanity", action="store_true", help="run the three built-in pairs")
    args = parser.parse_args()

    if bool(args.text_a) != bool(args.text_b):
        parser.error("--text-a and --text-b must be given together")
    if not args.sanity and not args.text_a:
        parser.error("pass --text-a and --text-b, or --sanity")

    api_key = get_settings().OPENAI_API_KEY
    if not api_key:
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 2
    embedder = OpenAIEmbedder(OpenAI(api_key=api_key))

    if args.text_a:
        _print_pair(embedder, args.text_a, args.text_b)
        return 0

    scores = {}
    for label, text_a, text_b in SANITY_PAIRS:
        print(f"--- {label}")
        scores[label] = _print_pair(embedder, text_a, text_b)
    ordered = scores["close"] > scores["related"] > scores["unrelated"]
    print(f"\nclose > related > unrelated: {'PASS' if ordered else 'FAIL'}")
    return 0 if ordered else 1


if __name__ == "__main__":
    sys.exit(main())
