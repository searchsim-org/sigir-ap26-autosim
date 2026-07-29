"""Definitive real sim-to-real validity on the FULL TREC iKAT 2023 judged set.

Resilient: retry-with-backoff on every network call, per-turn JSONL cache (resumable
on restart), and honest skip+log if a turn persistently fails (no fakes/fallbacks).
Primary metric: condensed nDCG@5 (raw reported too).

Run:    .venv/bin/python scripts/sim2real_full.py        # full run / resume from cache
Aggreg: .venv/bin/python scripts/sim2real_full.py agg    # recompute result from cache only
"""
import json
import os
import sys
import time
from pathlib import Path

from scipy.stats import kendalltau

from autosim.runtime import load_env, build_generator, build_retrieval, build_splade, build_reranker
from autosim.data.ikat_real import load_ikat_topics, load_qrels
from autosim.eval.trec_metrics import system_ndcg, system_ndcg_condensed
from autosim.eval.assessor import AutoSimAssessor
from autosim.metrics.distributions import bootstrap_ci
from autosim.metrics.ranking import induced_ranking
from autosim.retrieval.fusion import rrf

YEAR = os.environ.get("AUTOSIM_YEAR", "2023")          # set AUTOSIM_YEAR=2024 to replicate
DATA = Path(f"data/ikat/{YEAR}")
CACHE = DATA / ("pool_cache_v2.jsonl" if YEAR == "2023" else f"pool_cache_{YEAR}.jsonl")
RESULT = DATA / ("sim2real_result_v2.txt" if YEAR == "2023" else f"sim2real_result_{YEAR}.txt")
QRELS = next(DATA.glob("*qrels*"))                     # .txt (2023/24) or .trec (2025)
TOPICS = DATA / f"{YEAR}_test_topics.json"
SYSTEMS = ["bm25(raw)", "bm25(resolved)", "bm25(ptkb)",
           "splade(raw)", "splade(resolved)", "splade(ptkb)",
           "rrf(resolved)", "rrf(ptkb)",
           "bm25+rerank", "splade+rerank", "rrf+rerank", "splade(ptkb)+rerank"]


def _retry(fn, tries: int = 4, base: float = 2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:               # transient network/LLM error -> back off and retry the REAL call
            last = e
            time.sleep(base * (i + 1))
    raise last


def _tuples(ps, n=10):
    return [(p.doc_id, p.text, p.score) for p in ps[:n]]


def build_systems(bm25, splade, reranker, qr, qu, ptkb):
    qp = (qr + " " + " ".join(ptkb[:8])).strip()        # PTKB-expanded query (persona context)
    bm25_raw = _retry(lambda: bm25.search(qu, k=50))
    bm25_res = _retry(lambda: bm25.search(qr, k=50))
    bm25_pt = _retry(lambda: bm25.search(qp, k=50))
    spl_raw = _retry(lambda: splade.search(qu, k=50))
    spl_res = _retry(lambda: splade.search(qr, k=50))
    spl_pt = _retry(lambda: splade.search(qp, k=50))
    rrf_res = rrf([bm25_res, spl_res], top_n=50)
    rrf_pt = rrf([bm25_pt, spl_pt], top_n=50)
    return {
        "bm25(raw)": _tuples(bm25_raw),
        "bm25(resolved)": _tuples(bm25_res),
        "bm25(ptkb)": _tuples(bm25_pt),
        "splade(raw)": _tuples(spl_raw),
        "splade(resolved)": _tuples(spl_res),
        "splade(ptkb)": _tuples(spl_pt),
        "rrf(resolved)": _tuples(rrf_res),
        "rrf(ptkb)": _tuples(rrf_pt),
        "bm25+rerank": _tuples(_retry(lambda: reranker.rerank(qr, bm25_res, top_n=10))),
        "splade+rerank": _tuples(_retry(lambda: reranker.rerank(qr, spl_res, top_n=10))),
        "rrf+rerank": _tuples(_retry(lambda: reranker.rerank(qr, rrf_res, top_n=10))),
        "splade(ptkb)+rerank": _tuples(_retry(lambda: reranker.rerank(qp, spl_pt, top_n=10))),
    }


def load_cache():
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["tid"]] = rec
    return done


def aggregate(done, qrels, skipped):
    runs = {n: {} for n in SYSTEMS}
    autos = {n: [] for n in SYSTEMS}
    for tid, rec in done.items():
        for n in SYSTEMS:
            if n in rec["rankings"]:
                runs[n][tid] = [(sid, "", 0.0) for sid in rec["rankings"][n]]
                autos[n].append(rec["autosim"][n])
    ndcg = {n: system_ndcg(runs[n], qrels, 5)[0] for n in SYSTEMS}
    ndcg_c = {n: system_ndcg_condensed(runs[n], qrels, 5)[0] for n in SYSTEMS}
    autosim = {n: (sum(autos[n]) / len(autos[n]) if autos[n] else 0.0) for n in SYSTEMS}

    lines = [f"{'system':22s} {'nDCG@5':>8s} {'cond_nDCG':>9s} {'AutoSim':>8s}"]
    for n in induced_ranking(ndcg_c):
        lines.append(f"{n:22s} {ndcg[n]:8.4f} {ndcg_c[n]:9.4f} {autosim[n]:8.4f}")
    for label, ref in [("condensed nDCG@5", ndcg_c), ("raw nDCG@5", ndcg)]:
        tau, p = kendalltau([ref[n] for n in SYSTEMS], [autosim[n] for n in SYSTEMS])
        conc = [1.0 if (ref[a] > ref[b]) == (autosim[a] > autosim[b]) else 0.0
                for i, a in enumerate(SYSTEMS) for b in SYSTEMS[i + 1:]]
        lo, hi = bootstrap_ci(conc, n=2000, seed=0)
        lines.append(f"Kendall-tau(AutoSim, {label:18s}) = {tau:+.3f} (p={p:.3f})  "
                     f"concordance 95% CI [{lo:.2f}, {hi:.2f}]")
    lines.append(f"({len(SYSTEMS)} systems, {len(done)} turns; {len(skipped)} skipped)")
    out = "\n".join(lines)
    print("\n" + out, flush=True)
    RESULT.write_text(out + "\n")
    if skipped:
        print("skipped:", skipped[:10], flush=True)


def main():
    qrels = load_qrels(QRELS)
    if len(sys.argv) > 1 and sys.argv[1] == "agg":
        aggregate(load_cache(), qrels, [])
        return

    env = load_env()
    bm25 = build_retrieval(env)
    splade = build_splade(env, bm25)
    reranker = build_reranker(env)
    assessor = AutoSimAssessor(build_generator(env))
    topics = load_ikat_topics(TOPICS)

    judged = [t for t in qrels if any(v > 0 for v in qrels[t].values()) and t in topics]
    done = load_cache()
    todo = [t for t in judged if t not in done]
    print(f"full run: {len(judged)} judged turns | {len(done)} cached | {len(todo)} to do", flush=True)

    skipped = []
    with CACHE.open("a") as cache:
        for tid in todo:
            try:
                per = build_systems(bm25, splade, reranker, topics[tid]["resolved"],
                                    topics[tid]["utterance"], topics[tid]["ptkb"])
                autosim = {}
                for name, ranked in per.items():
                    texts = [t for _, t, _ in ranked[:5]]
                    autosim[name] = _retry(lambda tx=texts: assessor.score_turn(
                        topics[tid]["ptkb"], topics[tid]["resolved"], tx, k=5))
                rec = {"tid": tid,
                       "rankings": {n: [sid for sid, _, _ in r] for n, r in per.items()},
                       "autosim": autosim}
                cache.write(json.dumps(rec) + "\n")
                cache.flush()
                done[tid] = rec
                print(f"  [{len(done)}/{len(judged)}] {tid} ok", flush=True)
            except Exception as e:
                skipped.append((tid, repr(e)[:140]))
                print(f"  SKIP {tid}: {e!r}"[:180], flush=True)

    aggregate(done, qrels, skipped)


if __name__ == "__main__":
    main()
