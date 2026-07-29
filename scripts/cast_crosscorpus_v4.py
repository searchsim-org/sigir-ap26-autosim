"""Cross-corpus v4: broaden the CAsT-2020 validity pool BEYOND lexical-only retrieval by
adding a NEURAL rerank tier, so the evaluator is tested when the pool contains neural systems.
Fifteen systems:
  {raw, prev1, allprev, auto, manual} x {BM25, BM25+RM3}          (10 lexical, as in v3)
  {raw, prev1, allprev, auto, manual} x BM25 -> qwen3-reranker-4b (5 neural cross-encoder)
The reranker reorders BM25 top-50 (gateway /v1/rerank). No-persona AutoSim assessor (CAsT has
no PTKBs). If AutoSim's satisfaction still tracks NIST nDCG on the 15-system pool, the validity
is not an artifact of a lexical-only pool.

Run: JAVA_HOME=$(/usr/libexec/java_home) .venv/bin/python scripts/cast_crosscorpus_v4.py
"""
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ir_datasets
from scipy.stats import kendalltau
from pyserini.search.lucene import LuceneSearcher

from autosim.runtime import load_env, build_generator, build_reranker
from autosim.retrieval.backend import Passage
from autosim.eval.assessor import AutoSimAssessor
from autosim.eval.trec_metrics import system_ndcg, system_ndcg_condensed

OUT = Path("data/cast")
WORKERS = 16
TOPK = 50   # BM25 depth fed to the reranker


def doc_text(bm25, docid):
    d = bm25.doc(docid)
    if d is None:
        return ""
    try:
        return json.loads(d.raw())["contents"]
    except Exception:
        return d.contents() or ""


def build_queries(ds):
    byid = {q.query_id: q for q in ds.queries_iter()}
    topics = defaultdict(list)
    for q in byid.values():
        topics[q.topic_number].append(q)
    out = {}
    for tn, qs in topics.items():
        qs.sort(key=lambda q: int(q.turn_number))
        prev_raws = []
        for q in qs:
            raw = q.raw_utterance
            out[q.query_id] = {
                "raw": raw,
                "prev1": ((prev_raws[-1] + " ") if prev_raws else "") + raw,
                "allprev": (" ".join(prev_raws) + " " + raw).strip(),
                "auto": q.automatic_rewritten_utterance or raw,
                "manual": q.manual_rewritten_utterance or raw,
            }
            prev_raws.append(raw)
    return byid, out


def main():
    ds = ir_datasets.load("trec-cast/v1/2020")
    byid, qstr = build_queries(ds)
    qrels = {}
    for x in ds.qrels_iter():
        qrels.setdefault(x.query_id, {})[x.doc_id] = x.relevance
    judged = [t for t in qrels if any(v > 0 for v in qrels[t].values()) and t in qstr]
    print(f"CAsT 2020: {len(judged)} judged turns", flush=True)

    bm25 = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
    reranker = build_reranker(load_env())
    AXIS = ["raw", "prev1", "allprev", "auto", "manual"]

    # --- Phase 1 (serial pyserini): BM25 top-50 WITH texts, no-RM3; build base lexical systems ---
    top50 = {}   # (ax, turn) -> [(docid, text, score) x <=50]
    systems = {}
    for ax in AXIS:
        base = {}
        for t in judged:
            hits = bm25.search(qstr[t][ax], k=TOPK)
            lst = [(h.docid, doc_text(bm25, h.docid), h.score) for h in hits]
            top50[(ax, t)] = lst
            base[t] = [(f"MARCO_{d}", (tx if i < 5 else ""), sc) for i, (d, tx, sc) in enumerate(lst[:10])]
        systems[f"bm25({ax})"] = base
        print(f"  built bm25({ax}) + cached top-{TOPK} texts", flush=True)

    # --- Phase 2 (parallel gateway): rerank BM25 top-50 -> neural systems ---
    def do_rerank(key):
        ax, t = key
        passages = [Passage(did=i, text=tx, doc_id=d) for i, (d, tx, sc) in enumerate(top50[key]) if tx]
        try:
            rr = reranker.rerank(qstr[t][ax], passages, top_n=10)
        except Exception:
            rr = []
        return key, [(f"MARCO_{p.doc_id}", p.text, p.score) for p in rr]

    keys = [(ax, t) for ax in AXIS for t in judged]
    rr_out = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for key, val in ex.map(do_rerank, keys):
            rr_out[key] = val
    for ax in AXIS:
        systems[f"bm25+rerank({ax})"] = {t: rr_out[(ax, t)] for t in judged}
        print(f"  built bm25+rerank({ax})", flush=True)

    # --- Phase 3 (serial pyserini): RM3 lexical systems ---
    bm25.set_rm3()
    for ax in AXIS:
        base = {}
        for t in judged:
            hits = bm25.search(qstr[t][ax], k=10)
            base[t] = [(f"MARCO_{h.docid}", (doc_text(bm25, h.docid) if i < 5 else ""), h.score)
                       for i, h in enumerate(hits)]
        systems[f"bm25+rm3({ax})"] = base
        print(f"  built bm25+rm3({ax})", flush=True)

    names = list(systems)
    ndcg = {n: system_ndcg(systems[n], qrels, 5)[0] for n in names}
    ndcg_c = {n: system_ndcg_condensed(systems[n], qrels, 5)[0] for n in names}

    # --- assessor (parallel) ---
    env = load_env(); asr = AutoSimAssessor(build_generator(env))
    tasks = [(n, t) for n in names for t in judged]

    def score(it):
        n, t = it
        texts = [tx for _, tx, _ in systems[n][t][:5] if tx]
        try:
            return it, asr.score_turn([], qstr[t]["manual"], texts, k=5)
        except Exception:
            return it, None
    acc = {n: [0.0, 0] for n in names}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for (n, t), v in ex.map(score, tasks):
            if v is not None:
                acc[n][0] += v; acc[n][1] += 1
    autosim = {n: (acc[n][0] / acc[n][1] if acc[n][1] else 0.0) for n in names}

    def boot(x, y, B=10000, seed=0):
        rng = random.Random(seed); m = len(x); ts = []
        for _ in range(B):
            idx = [rng.randrange(m) for _ in range(m)]
            t = kendalltau([x[i] for i in idx], [y[i] for i in idx])[0]
            if t == t:
                ts.append(t)
        ts.sort(); return ts[int(.025 * len(ts))], ts[int(.975 * len(ts))]

    order = sorted(names, key=lambda n: -ndcg_c[n])
    neural = [n for n in names if "rerank" in n]
    lines = [f"CROSS-CORPUS v4: CAsT 2020 / MS MARCO, 15-system pool (10 lexical + 5 neural rerank), {len(judged)} turns",
             f"{'system':22s} {'nDCG@5':>8s} {'cond':>8s} {'AutoSim':>8s}"]
    for n in order:
        tag = " [neural]" if "rerank" in n else ""
        lines.append(f"{n:22s} {ndcg[n]:8.4f} {ndcg_c[n]:8.4f} {autosim[n]:8.4f}{tag}")
    for label, ref in [("condensed nDCG@5", ndcg_c), ("raw nDCG@5", ndcg)]:
        x = [ref[n] for n in names]; y = [autosim[n] for n in names]
        tau, p = kendalltau(x, y); lo, hi = boot(x, y)
        lines.append(f"Kendall tau (AutoSim, {label:16s}, N={len(names)}) = {tau:+.3f} (p={p:.3f})  "
                     f"system-bootstrap 95% CI [{lo:+.3f},{hi:+.3f}]")
    # tau restricted to the neural subpool (does the assessor rank neural systems too?)
    xn = [ndcg_c[n] for n in neural]; yn = [autosim[n] for n in neural]
    tn, pn = kendalltau(xn, yn)
    lines.append(f"Kendall tau on the 5 neural rerank systems only (condensed) = {tn:+.3f} (p={pn:.3f})")
    out = "\n".join(lines)
    print("\n" + out, flush=True)
    (OUT / "cast2020_crosscorpus_v4.txt").write_text(out + "\n")
    (OUT / "cast2020_crosscorpus_v4.json").write_text(json.dumps(
        {"names": names, "ndcg": ndcg, "ndcg_condensed": ndcg_c, "autosim": autosim,
         "n_turns": len(judged)}, indent=1))


if __name__ == "__main__":
    main()
