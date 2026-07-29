"""Headline real sim-to-real validity on TREC iKAT 2023, over a CONTROLLED pool of
real retrieval systems built on the official corpus (BM25 / SPLADE / RRF / +rerank /
query variants).

reference ranking = systems by mean nDCG@5 over the REAL NIST qrels
AutoSim ranking   = systems by AutoSim persona-conditioned mean satisfaction
headline          = Kendall-tau(AutoSim, NIST) + pairwise-concordance bootstrap CI

Run: .venv/bin/python scripts/sim2real_pool.py [n_turns]
No fakes/fallbacks.
"""
import sys
from pathlib import Path

from scipy.stats import kendalltau

from autosim.runtime import load_env, build_generator, build_retrieval, build_splade, build_reranker
from autosim.data.ikat_real import load_ikat_topics, load_qrels
from autosim.eval.trec_metrics import system_ndcg, system_ndcg_condensed
from autosim.eval.assessor import AutoSimAssessor
from autosim.metrics.distributions import bootstrap_ci
from autosim.metrics.ranking import induced_ranking
from autosim.retrieval.fusion import rrf

DATA = Path("data/ikat/2023")


def _tuples(passages, n=10):
    return [(p.doc_id, p.text, p.score) for p in passages[:n]]


def build_systems(bm25, splade, reranker, q_resolved, q_utterance):
    """Run all retrievers once per turn; derive the system pool from shared candidates."""
    bm25_r = bm25.search(q_resolved, k=50)
    splade_r = splade.search(q_resolved, k=50)
    bm25_u = bm25.search(q_utterance, k=50)
    rrf_r = rrf([bm25_r, splade_r], top_n=50)
    return {
        "bm25(raw-utterance)": _tuples(bm25_u),
        "bm25(resolved)": _tuples(bm25_r),
        "splade(resolved)": _tuples(splade_r),
        "rrf(bm25+splade)": _tuples(rrf_r),
        "bm25+rerank": _tuples(reranker.rerank(q_resolved, bm25_r, top_n=10)),
        "splade+rerank": _tuples(reranker.rerank(q_resolved, splade_r, top_n=10)),
        "rrf+rerank": _tuples(reranker.rerank(q_resolved, rrf_r, top_n=10)),
    }


def main(n_turns: int = 8) -> None:
    env = load_env()
    bm25 = build_retrieval(env)
    splade = build_splade(env, bm25)
    reranker = build_reranker(env)
    assessor = AutoSimAssessor(build_generator(env))

    topics = load_ikat_topics(DATA / "2023_test_topics.json")
    qrels = load_qrels(DATA / "2023-qrels.all-turns.txt")
    judged = [t for t in qrels if any(v > 0 for v in qrels[t].values()) and t in topics][:n_turns]
    print(f"pool over {len(judged)} judged turns x 7 real systems")

    systems: dict[str, dict] = {}
    for i, tid in enumerate(judged):
        per = build_systems(bm25, splade, reranker, topics[tid]["resolved"], topics[tid]["utterance"])
        for name, ranked in per.items():
            systems.setdefault(name, {})[tid] = ranked
        print(f"  [{i + 1}/{len(judged)}] {tid}")

    ndcg = {name: system_ndcg(run, qrels, 5)[0] for name, run in systems.items()}
    ndcg_c = {name: system_ndcg_condensed(run, qrels, 5)[0] for name, run in systems.items()}
    autosim = {name: assessor.score_system(run, topics, judged, k=5) for name, run in systems.items()}

    print(f"\n{'system':22s} {'nDCG@5':>8s} {'cond_nDCG':>9s} {'AutoSim':>8s}")
    for name in induced_ranking(ndcg_c):
        print(f"{name:22s} {ndcg[name]:8.4f} {ndcg_c[name]:9.4f} {autosim[name]:8.4f}")

    names = list(systems)
    for label, ref in [("raw nDCG@5", ndcg), ("condensed nDCG@5", ndcg_c)]:
        tau, p = kendalltau([ref[n] for n in names], [autosim[n] for n in names])
        conc = [1.0 if (ref[a] > ref[b]) == (autosim[a] > autosim[b]) else 0.0
                for i, a in enumerate(names) for b in names[i + 1:]]
        lo, hi = bootstrap_ci(conc, n=1000, seed=0)
        print(f"Kendall-tau(AutoSim, {label:18s}) = {tau:+.3f} (p={p:.3f})  "
              f"concordance 95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"({len(names)} systems, {len(judged)} turns)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
