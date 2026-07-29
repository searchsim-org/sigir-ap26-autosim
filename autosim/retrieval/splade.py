"""Official iKAT SPLADE first-stage retriever (public service) + passage text via
OpenSearch _mget. Returns Passages whose doc_id is the qrels-matching seg_id.
No fakes/fallbacks: a failed call raises.
"""
import httpx

from autosim.retrieval.backend import Passage


class SpladeBackend:
    def __init__(self, *, splade_url: str, user: str, password: str, os_backend, timeout: float = 60.0):
        self.url = splade_url.rstrip("/")
        self.auth = (user, password)
        self.os_backend = os_backend           # OpenSearchBM25Backend, for mget_texts
        self.timeout = timeout

    def search(self, query: str, *, k: int = 10) -> list[Passage]:
        r = httpx.get(f"{self.url}/search", params={"q": query, "k": k},
                      auth=self.auth, timeout=self.timeout)
        r.raise_for_status()
        results = r.json().get("results", [])
        texts = self.os_backend.mget_texts([x["seg_id"] for x in results])
        return [Passage(did=abs(hash(x["seg_id"])) % (10 ** 9), text=texts.get(x["seg_id"], ""),
                        score=float(x["score"]), doc_id=x["seg_id"]) for x in results]
