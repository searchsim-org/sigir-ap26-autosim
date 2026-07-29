"""Cross-corpus v2: widen the CAsT-2020/MS-MARCO pool with SPLADE (a different
retrieval method) so the validity tau has enough systems to be powered. 9 systems:
BM25 (+RM3) x {raw,auto,manual} and SPLADE x {raw,auto,manual}. SPLADE retrieves
better/different passages, extending the effectiveness range. Text for SPLADE hits is
fetched from the BM25 index (same MS MARCO docids). No-persona AutoSim assessor.

Run with JAVA_HOME set:
  JAVA_HOME=$(/usr/libexec/java_home) .venv/bin/python scripts/cast_crosscorpus_v2.py
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ir_datasets
from scipy.stats import kendalltau
from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher

from autosim.runtime import load_env, build_generator
from autosim.eval.assessor import AutoSimAssessor
from autosim.eval.trec_metrics import system_ndcg, system_ndcg_condensed

OUT = Path("data/cast")
WORKERS = 16
QTYPES = {"raw": "raw_utterance", "auto": "automatic_rewritten_utterance",
          "manual": "manual_rewritten_utterance"}


def doc_text(bm25, docid):
    d = bm25.doc(docid)
    if d is None:
        return ""
    try:
        return json.loads(d.raw())["contents"]
    except Exception:
        return d.contents() or ""


def retrieve(searcher, bm25, q, marco_prefix=True):
    hits = searcher.search(q, k=10)
    out = []
    for h in hits:
        pid = f"MARCO_{h.docid}"
        txt = doc_text(bm25, h.docid) if len(out) < 5 else ""
        out.append((pid, txt, h.score))
    return out


def main():
    ds = ir_datasets.load("trec-cast/v1/2020")
    queries = {q.query_id: q for q in ds.queries_iter()}
    qrels = {}
    for x in ds.qrels_iter():
        qrels.setdefault(x.query_id, {})[x.doc_id] = x.relevance
    judged = [t for t in qrels if any(v > 0 for v in qrels[t].values()) and t in queries]
    print(f"CAsT 2020: {len(judged)} judged turns", flush=True)

    bm25 = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
    splade = LuceneImpactSearcher.from_prebuilt_index(
        "msmarco-v1-passage.splade-pp-ed", "naver/splade-cocondenser-ensembledistil")

    systems = {}
    # BM25 (+RM3)
    for rm3 in (False, True):
        if rm3:
            bm25.set_rm3()
        for qk, qf in QTYPES.items():
            name = f"bm25{'+rm3' if rm3 else ''}({qk})"
            systems[name] = {t: retrieve(bm25, bm25, getattr(queries[t], qf, "") or
                                         queries[t].raw_utterance) for t in judged}
            print(f"  built {name}", flush=True)
    bm25_plain = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")  # rm3-free, for text
    # SPLADE (text via bm25_plain)
    for qk, qf in QTYPES.items():
        name = f"splade({qk})"
        systems[name] = {t: retrieve(splade, bm25_plain, getattr(queries[t], qf, "") or
                                     queries[t].raw_utterance) for t in judged}
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
            return it, asr.score_turn([], queries[t].manual_rewritten_utterance, texts, k=5)
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
    lines = [f"CROSS-CORPUS v2: CAsT 2020 over MS MARCO, BM25+SPLADE pool, {len(judged)} turns",
             f"{'system':16s} {'nDCG@5':>8s} {'cond':>8s} {'AutoSim':>8s}"]
    for n in order:
        lines.append(f"{n:16s} {ndcg[n]:8.4f} {ndcg_c[n]:8.4f} {autosim[n]:8.4f}")
    for label, ref in [("condensed nDCG@5", ndcg_c), ("raw nDCG@5", ndcg)]:
        x = [ref[n] for n in names]; y = [autosim[n] for n in names]
        tau, p = kendalltau(x, y); lo, hi = boot(x, y)
        lines.append(f"Kendall tau (AutoSim, {label:16s}) = {tau:+.3f} (p={p:.3f})  "
                     f"system-bootstrap 95% CI [{lo:+.3f},{hi:+.3f}]  (N={len(names)})")
    out = "\n".join(lines)
    print("\n" + out, flush=True)
    (OUT / "cast2020_crosscorpus_v2.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()
