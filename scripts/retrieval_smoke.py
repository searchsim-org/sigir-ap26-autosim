"""Live smoke for the real retrieval pool: BM25 + SPLADE + RRF + reranker,
over a real iKAT resolved utterance. Confirms qrels-matching seg_ids + real text.
"""
from autosim.runtime import load_env, build_retrieval, build_splade, build_reranker
from autosim.retrieval.fusion import rrf


def main() -> None:
    env = load_env()
    bm25 = build_retrieval(env)
    splade = build_splade(env, bm25)
    reranker = build_reranker(env)

    q = ("Can you help me find a diet for myself considering that I'm vegetarian, "
         "allergic to soybeans, lactose intolerant, and can't exercise too much?")

    b = bm25.search(q, k=5)
    s = splade.search(q, k=5)
    f = rrf([bm25.search(q, k=30), splade.search(q, k=30)], top_n=5)
    rk = reranker.rerank(q, splade.search(q, k=20), top_n=5)

    print("BM25 top5         :", [(p.doc_id, round(p.score, 2)) for p in b])
    print("SPLADE top5       :", [(p.doc_id, round(p.score, 2)) for p in s])
    print("RRF(BM25,SPLADE)  :", [(p.doc_id, round(p.score, 4)) for p in f])
    print("SPLADE+rerank top5:", [(p.doc_id, round(p.score, 3)) for p in rk])
    print("\nSPLADE[0] text:", (s[0].text[:120] if s else None))
    ok = all(":" in p.doc_id for p in (b + s + f + rk))
    print("all doc_ids are qrels-format seg_ids:", ok)


if __name__ == "__main__":
    main()
