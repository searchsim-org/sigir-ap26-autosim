from typing import Protocol
from pydantic import BaseModel


class Passage(BaseModel):
    did: int
    text: str
    score: float = 0.0
    doc_id: str = ""        # external id (e.g. ClueWeb22-ID) for live retrieval


class RetrievalBackend(Protocol):
    def search(self, query: str, *, k: int = 10) -> list[Passage]: ...
