from collections import Counter
from autosim.retrieval.backend import Passage


class FakeRetrievalBackend:
    def __init__(self, docs: dict[int, str]):
        self.docs = docs

    def search(self, query: str, *, k: int = 10) -> list[Passage]:
        q = Counter(query.lower().split())
        scored = []
        for did, text in self.docs.items():
            overlap = sum((Counter(text.lower().split()) & q).values())
            if overlap:
                scored.append(Passage(did=did, text=text, score=float(overlap)))
        scored.sort(key=lambda p: (-p.score, p.did))
        return scored[:k]
