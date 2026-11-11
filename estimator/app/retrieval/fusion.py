"""Reciprocal Rank Fusion: merge several rankings into one using only ranks, never scores.

Cosine distances and ts_rank values live on unrelated scales, so they cannot be added or
averaged. RRF sidesteps that: each list contributes ``1 / (k + rank)`` per item and the
contributions are summed. Items found by several lists rise to the top.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def rrf_fuse(rankings: Sequence[Sequence[Hashable]], *, k: int = 60) -> list[tuple[Hashable, float]]:
    """Fuse ordered id lists (best first) into ``[(id, rrf_score)]``, best first.

    Ties keep the order of first appearance, so the result is deterministic.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
