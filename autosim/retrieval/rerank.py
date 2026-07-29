"""Cross-encoder reranker via the gateway's Cohere-style /v1/rerank route
(qwen3-reranker-4b). Reorders a candidate Passage list by relevance.
"""
import httpx

from autosim.retrieval.backend import Passage


class Reranker:
    def __init__(self, *, base_url: str, api_key: str, model: str = "qwen3-reranker-4b",
                 timeout: float = 120.0):
        self.url = base_url.rstrip("/") + "/v1/rerank"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.model = model
        self.timeout = timeout

    def rerank(self, query: str, passages: list[Passage], *, top_n: int = 20) -> list[Passage]:
        if not passages:
            return []
        r = httpx.post(self.url, headers=self.headers, timeout=self.timeout,
                       json={"model": self.model, "query": query,
                             "documents": [p.text[:1000] for p in passages], "top_n": top_n})
        r.raise_for_status()
        out = []
        for res in r.json()["results"]:
            p = passages[res["index"]]
            out.append(Passage(did=p.did, text=p.text, score=float(res["relevance_score"]), doc_id=p.doc_id))
        return out
