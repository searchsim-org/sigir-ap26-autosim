"""Real BM25 retrieval over the ClueWeb22-B iKAT index (reader-scoped).

Discovered live (2026-06-28): index `clueweb22-b-ikat` exposes fields
`seg_id, clueweb_id, passage_id, contents, url`; the `reader` role can run
`indices:data/read/search` but not monitor/admin, and the index has no
knn_vector field, so retrieval here is lexical (BM25 on `contents`).
No fakes/fallbacks: a failed search raises.
"""
from opensearchpy import OpenSearch

from autosim.retrieval.backend import Passage


class OpenSearchBM25Backend:
    def __init__(self, *, host: str, user: str, password: str, index: str,
                 text_field: str = "contents", id_field: str = "clueweb_id",
                 port: int = 443, timeout: float = 60.0):
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}], http_auth=(user, password),
            use_ssl=True, verify_certs=True, ssl_show_warn=False, timeout=timeout)
        self.index, self.text_field, self.id_field = index, text_field, id_field

    def search(self, query: str, *, k: int = 10) -> list[Passage]:
        resp = self.client.search(index=self.index, body={
            "size": k,
            "_source": ["seg_id", self.id_field, "passage_id", "url", self.text_field],
            "query": {"match": {self.text_field: query}},
        })
        passages: list[Passage] = []
        for h in resp["hits"]["hits"]:
            src = h["_source"]
            seg = str(src.get("seg_id") or h.get("_id", ""))   # seg_id matches the qrels passage id
            passages.append(Passage(did=abs(hash(seg)) % (10 ** 9), text=src.get(self.text_field, ""),
                                    score=float(h["_score"]), doc_id=seg))
        return passages

    def mget_texts(self, seg_ids: list[str]) -> dict[str, str]:
        """Fetch passage text by seg_id (== _id) for SPLADE/fusion results."""
        if not seg_ids:
            return {}
        resp = self.client.mget(index=self.index, body={"ids": seg_ids}, _source=[self.text_field])
        return {d["_id"]: d["_source"].get(self.text_field, "") for d in resp["docs"] if d.get("found")}
