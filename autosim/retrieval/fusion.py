"""Reciprocal Rank Fusion of multiple Passage rankings (by doc_id / seg_id)."""
from autosim.retrieval.backend import Passage


def rrf(rankings: list[list[Passage]], *, k: int = 60, top_n: int | None = None) -> list[Passage]:
    scores: dict[str, float] = {}
    text: dict[str, str] = {}
    for ranking in rankings:
        for rank, p in enumerate(ranking):
            scores[p.doc_id] = scores.get(p.doc_id, 0.0) + 1.0 / (k + rank + 1)
            text.setdefault(p.doc_id, p.text)
    fused = [Passage(did=abs(hash(d)) % (10 ** 9), text=text[d], score=s, doc_id=d)
             for d, s in sorted(scores.items(), key=lambda kv: -kv[1])]
    return fused[:top_n] if top_n else fused
