from autosim.retrieval.backend import Passage


class OpenSearchBackend:
    """Hybrid BM25 + jina-v3 kNN over ClueWeb22-B (index `clueweb22-b-jina`).
    Requires an opensearch-py client and an embedder for the query vector."""

    def __init__(self, client, index: str, embed, *, knn_k: int = 100):
        self.client, self.index, self.embed, self.knn_k = client, index, embed, knn_k

    def search(self, query: str, *, k: int = 10) -> list[Passage]:
        body = {
            "size": k,
            "query": {"hybrid": {"queries": [
                {"match": {"text": query}},
                {"knn": {"embedding": {"vector": self.embed(query), "k": self.knn_k}}},
            ]}},
            "collapse": {"field": "did"},
        }
        resp = self.client.search(index=self.index, body=body)
        return [Passage(did=int(h["_source"]["did"]), text=h["_source"].get("text", ""),
                        score=float(h["_score"])) for h in resp["hits"]["hits"]]
