"""Cross-corpus v3 (reliable, BM25-only over the cached MS MARCO index): widen the
CAsT-2020 pool along the PRINCIPLED conversational query-resolution quality axis, which
is exactly what a valid evaluator must rank correctly. Ten systems:
  {raw (no context), prev1 (1-turn context), allprev (full context),
   auto (system rewrite), manual (oracle)} x {BM25, BM25+RM3}.
This spans a wide effectiveness range and gives the validity tau enough systems/spread.
No-persona AutoSim assessor (CAsT has no PTKBs -> tests the judging surface).

Run: JAVA_HOME=$(/usr/libexec/java_home) .venv/bin/python scripts/cast_crosscorpus_v3.py
"""
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ir_datasets
from scipy.stats import kendalltau
from pyserini.search.lucene import LuceneSearcher

from autosim.runtime import load_env, build_generator
from autosim.eval.assessor import AutoSimAssessor
from autosim.eval.trec_metrics import system_ndcg, system_ndcg_condensed

OUT = Path("data/cast")
WORKERS = 16


def doc_text(bm25, docid):
    d = bm25.doc(docid)
    if d is None:
        return ""
    try:
        return json.loads(d.raw())["contents"]
    except Exception:
        return d.contents() or ""


def build_queries(ds):
    """Per turn-id -> {raw, prev1, allprev, auto, manual} query strings (context spectrum)."""
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
    AXIS = ["raw", "prev1", "allprev", "auto", "manual"]

    def retrieve(q):
        out = []
        for h in bm25.search(q, k=10):
            txt = doc_text(bm25, h.docid) if len(out) < 5 else ""
            out.append((f"MARCO_{h.docid}", txt, h.score))
        return out

    systems = {}
    for rm3 in (False, True):
        if rm3:
            bm25.set_rm3()
        for ax in AXIS:
            name = f"bm25{'+rm3' if rm3 else ''}({ax})"
            systems[name] = {t: retrieve(qstr[t][ax]) for t in judged}
            print(f"  built {name}", flush=True)

    names = list(systems)
    ndcg = {n: system_ndcg(systems[n], qrels, 5)[0] for n in names}
    ndcg_c = {n: system_ndcg_condensed(systems[n], qrels, 5)[0] for n in names}

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
    lines = [f"CROSS-CORPUS v3: CAsT 2020 / MS MARCO, 10-system query-resolution pool, {len(judged)} turns",
             f"{'system':18s} {'nDCG@5':>8s} {'cond':>8s} {'AutoSim':>8s}"]
    for n in order:
        lines.append(f"{n:18s} {ndcg[n]:8.4f} {ndcg_c[n]:8.4f} {autosim[n]:8.4f}")
    for label, ref in [("condensed nDCG@5", ndcg_c), ("raw nDCG@5", ndcg)]:
        x = [ref[n] for n in names]; y = [autosim[n] for n in names]
        tau, p = kendalltau(x, y); lo, hi = boot(x, y)
        lines.append(f"Kendall tau (AutoSim, {label:16s}) = {tau:+.3f} (p={p:.3f})  "
                     f"system-bootstrap 95% CI [{lo:+.3f},{hi:+.3f}]  (N={len(names)})")
    out = "\n".join(lines)
    print("\n" + out, flush=True)
    (OUT / "cast2020_crosscorpus_v3.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()
