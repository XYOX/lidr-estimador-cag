"""Session 09 pre-work: embed a raw transcript with the service's own embedder.

    cd estimator && PYTHONPATH=. uv run python ../examples/trace_embed.py ../examples/transcripts/02_ambiguous.txt
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from openai import OpenAI

from app.config import get_settings
from app.embedding_pipeline.chunker import count_tokens
from app.embedding_pipeline.embedder import OpenAIEmbedder


def main() -> int:
    text = Path(sys.argv[1]).read_text(encoding="utf-8")
    embedder = OpenAIEmbedder(OpenAI(api_key=get_settings().OPENAI_API_KEY))
    vector = embedder.embed_one(text)
    print(f"input_tokens={count_tokens(text)} input_chars={len(text)}")
    print(f"dimension={len(vector)}")
    print(f"first={vector[0]:.6f} last={vector[-1]:.6f}")
    print(f"l2_norm={math.sqrt(sum(v * v for v in vector)):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
