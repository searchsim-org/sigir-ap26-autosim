"""P2.9: connect the behavioral surface to the IR pipeline. The self-improved
(evolved, terse) simulator writes its OWN query each turn; we feed that query to the
real BM25 and SPLADE retrievers and score nDCG@5 against the NIST qrels. Comparing
gold vs default(verbose) vs evolved(terse) queries quantifies the *retrieval cost of
behavioral realism* -- does behaving like a real (terse) user hurt the user's ability
to retrieve relevant passages?

This is the honest end-to-end link: behavior -> query -> retrieval -> real qrels.
(We do not recompute the system-ranking tau under generated queries: changing the
query changes which passages are retrieved, breaking qrels alignment for a fair
cross-system nDCG. Retrieval effectiveness of the user's own query is the sound link.)

Run: .venv/bin/python scripts/end2end_realism.py [n]
Out: data/ikat/2023/end2end_realism.txt
"""
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autosim.runtime import load_env, build_generator, build_retrieval, build_splade
from autosim.data.ikat_real import load_qrels
from autosim.eval.trec_metrics import ndcg_at_k
from scripts.self_improve_behavior import gen_utt
from scripts.style_table_stats import DEFAULT, EVOLVED

DATA = Path("data/ikat/2023")
WORKERS = 16


def load_judged_records(qrels):
    topics = json.loads((DATA / "2023_test_topics.json").read_text())
    out = []
    for tp in topics:
        num = tp["number"]
        ptkb = list(tp.get("ptkb", {}).values())
        hist = []
        for t in tp.get("turns", []):
            tid = f"{num}_{t['turn_id']}"
            rec = {"tid": tid, "ptkb": ptkb,
                   "history": "\n".join(hist[-4:]) or "(start of conversation)",
                   "resolved": t.get("resolved_utterance", t.get("utterance", ""))}
            if tid in qrels and any(v > 0 for v in qrels[tid].values()):
                out.append(rec)
            hist.append(f"User: {t.get('utterance', '')}")
            if t.get("response"):
                hist.append(f"Assistant: {t['response'][:200]}")
    return out


def main(n=60):
    env = load_env()
    gen = build_generator(env)
    bm25 = build_retrieval(env)
    splade = build_splade(env, bm25)
    qrels = load_qrels(DATA / "2023-qrels.all-turns.txt")
    recs = load_judged_records(qrels)
    random.Random(99).shuffle(recs)
    recs = recs[:n]

    def work(rec):
        tid = rec["tid"]
        queries = {"gold": rec["resolved"],
                   "default": gen_utt(gen, DEFAULT, rec),
                   "evolved": gen_utt(gen, EVOLVED, rec)}
        row = {"qlen": {}, "bm25": {}, "splade": {}}
        for qn, q in queries.items():
            row["qlen"][qn] = len(q.split())
            try:
                bm = bm25.search(q, k=20)
                row["bm25"][qn] = ndcg_at_k([p.doc_id for p in bm], qrels[tid], 5)
            except Exception:
                row["bm25"][qn] = None
            try:
                sp = splade.search(q, k=20)
                row["splade"][qn] = ndcg_at_k([p.doc_id for p in sp], qrels[tid], 5)
            except Exception:
                row["splade"][qn] = None
        return row

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(work, recs))

    def agg(field, qn):
        vals = [r[field][qn] for r in rows if r[field].get(qn) is not None]
        return statistics.mean(vals) if vals else 0.0

    lines = [f"Retrieval cost of behavioral realism (n={n} judged turns)",
             f"  {'query':9s} {'avg_words':>9s} {'BM25 nDCG@5':>12s} {'SPLADE nDCG@5':>14s}"]
    for qn in ["gold", "default", "evolved"]:
        ql = statistics.mean([r["qlen"][qn] for r in rows])
        lines.append(f"  {qn:9s} {ql:9.1f} {agg('bm25', qn):12.4f} {agg('splade', qn):14.4f}")
    out = "\n".join(lines)
    print(out, flush=True)
    (DATA / "end2end_realism.txt").write_text(out + "\n")


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:2]]
    main(*a) if a else main()
