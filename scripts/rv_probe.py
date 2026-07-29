"""Probe: is ClueWeb22-B retrieval reachable, and what do judged turns look like (tid/keys)?
Fast-fail check before building the realism->validity test."""
from pathlib import Path

from autosim.runtime import load_env, build_retrieval
from autosim.data.ikat_real import load_ikat_topics, load_qrels

DATA = Path("data/ikat/2023")
env = load_env()
qrels = load_qrels(next(DATA.glob("*qrels*")))
topics = load_ikat_topics(DATA / "2023_test_topics.json")
judged = [t for t in qrels if any(v > 0 for v in qrels[t].values()) and t in topics]
print(f"judged turns: {len(judged)}; sample tid: {judged[0]}")
print(f"topic[tid] keys: {list(topics[judged[0]].keys())}")
print(f"sample resolved: {topics[judged[0]].get('resolved','')[:80]!r}")
print(f"sample utterance: {topics[judged[0]].get('utterance','')[:80]!r}")
bm25 = build_retrieval(env)
res = bm25.search(topics[judged[0]].get("resolved") or topics[judged[0]].get("utterance"), k=5)
print(f"RETRIEVAL OK: {len(res)} hits; top doc_id={res[0].doc_id if res else None}")
