"""CROSS-CORPUS validity test: does AutoSim's sim-to-real ranking transfer off
ClueWeb22-B? We run the same protocol on TREC CAsT 2020 (conversational search over
MS MARCO + TREC CAR -- a different corpus, and no PTKB personas, so AutoSim acts as
the persona-free relevance assessor it already validated as the locus of validity).

Pool: BM25 (+RM3) over the prebuilt MS MARCO passage index, for raw / automatic /
manual query rewrites -- a real effectiveness spread. Reference = condensed nDCG@5 over
NIST CAsT qrels (MS MARCO subset; CAR-relevant passages are unreachable by this index
and we report that caveat). AutoSim = mean satisfaction over each system's top-5.

Caveat reported in-paper: MS MARCO covers ~half of CAsT-2020 relevance; this tests the
MS-MARCO-retrievable slice, which is still a corpus disjoint from ClueWeb22-B.

Run: .venv/bin/python scripts/cast_crosscorpus.py
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ir_datasets
from scipy.stats import kendalltau
from pyserini.search.lucene import LuceneSearcher

from autosim.runtime import load_env, build_generator
from autosim.eval.assessor import AutoSimAssessor
from autosim.eval.trec_metrics import system_ndcg, system_ndcg_condensed

OUT = Path("data/cast"); OUT.mkdir(parents=True, exist_ok=True)
WORKERS = 16
QTYPES = {"raw": "raw_utterance", "auto": "automatic_rewritten_utterance",
          "manual": "manual_rewritten_utterance"}


def doc_text(searcher, docid):
    d = searcher.doc(docid)
    if d is None:
        return ""
    try:
        return json.loads(d.raw())["contents"]
    except Exception:
        return d.contents() or ""


def build_pool(searcher, queries, judged):
    """-> systems: {name: {tid: [(MARCO_id, text, score)]}}; uses top-10, keeps text for top-5."""
    systems = {}
    for rm3 in (False, True):          # non-RM3 first (default), then enable RM3
        if rm3:
            searcher.set_rm3()
        for qkey, qfield in QTYPES.items():
            name = f"bm25{'+rm3' if rm3 else ''}({qkey})"
            run = {}
            for tid in judged:
                q = getattr(queries[tid], qfield, "") or queries[tid].raw_utterance
                hits = searcher.search(q, k=10)
                ranked = []
                for h in hits:
                    pid = f"MARCO_{h.docid}"
                    txt = doc_text(searcher, h.docid) if len(ranked) < 5 else ""
                    ranked.append((pid, txt, h.score))
                run[tid] = ranked
            systems[name] = run
            print(f"  built {name}", flush=True)
    return systems


def main():
    ds = ir_datasets.load("trec-cast/v1/2020")
    queries = {q.query_id: q for q in ds.queries_iter()}
    qrels = {}
    for x in ds.qrels_iter():
        qrels.setdefault(x.query_id, {})[x.doc_id] = x.relevance
    judged = [t for t in qrels if any(v > 0 for v in qrels[t].values()) and t in queries]
    print(f"CAsT 2020: {len(judged)} judged turns", flush=True)

    print("loading prebuilt MS MARCO index (downloads ~2.5GB on first use)...", flush=True)
    searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
    systems = build_pool(searcher, queries, judged)
    names = list(systems)

    ndcg = {n: system_ndcg(systems[n], qrels, 5)[0] for n in names}
    ndcg_c = {n: system_ndcg_condensed(systems[n], qrels, 5)[0] for n in names}

    # AutoSim (persona-free: empty PTKB) scores each system's top-5 per turn
    env = load_env(); asr = AutoSimAssessor(build_generator(env))
    tasks = [(n, tid) for n in names for tid in judged]

    def score(t):
        n, tid = t
        texts = [tx for _, tx, _ in systems[n][tid][:5] if tx]
        try:
            return t, asr.score_turn([], queries[tid].manual_rewritten_utterance, texts, k=5)
        except Exception:
            return t, None
    acc = {n: [0.0, 0] for n in names}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for (n, tid), v in ex.map(score, tasks):
            if v is not None:
                acc[n][0] += v; acc[n][1] += 1
    autosim = {n: (acc[n][0] / acc[n][1] if acc[n][1] else 0.0) for n in names}

    order = sorted(names, key=lambda n: -ndcg_c[n])
    lines = [f"CROSS-CORPUS: TREC CAsT 2020 over MS MARCO (BM25 pool), {len(judged)} turns",
             f"{'system':16s} {'nDCG@5':>8s} {'cond':>8s} {'AutoSim':>8s}"]
    for n in order:
        lines.append(f"{n:16s} {ndcg[n]:8.4f} {ndcg_c[n]:8.4f} {autosim[n]:8.4f}")

    def boot(x, y, B=10000, seed=0):
        rng = random.Random(seed); m = len(x); ts = []
        for _ in range(B):
            idx = [rng.randrange(m) for _ in range(m)]
            t = kendalltau([x[i] for i in idx], [y[i] for i in idx])[0]
            if t == t:
                ts.append(t)
        ts.sort(); return ts[int(.025 * len(ts))], ts[int(.975 * len(ts))]
    for label, ref in [("condensed nDCG@5", ndcg_c), ("raw nDCG@5", ndcg)]:
        x = [ref[n] for n in names]; y = [autosim[n] for n in names]
        tau, p = kendalltau(x, y); lo, hi = boot(x, y)
        lines.append(f"Kendall tau (AutoSim, {label:16s}) = {tau:+.3f} (p={p:.3f})  "
                     f"system-bootstrap 95% CI [{lo:+.3f},{hi:+.3f}]  (N={len(names)} systems)")
    out = "\n".join(lines)
    print("\n" + out, flush=True)
    (OUT / "cast2020_crosscorpus.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()
