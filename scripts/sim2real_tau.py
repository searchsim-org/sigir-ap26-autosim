"""Real sim-to-real validity on TREC iKAT 2023.

Reference ranking  = systems by mean nDCG@5 over the REAL NIST qrels.
AutoSim ranking    = systems by AutoSim's persona-conditioned mean satisfaction.
Headline           = Kendall-tau between the two rankings (+ bootstrap CI).

System pool = the official baselines in data/ikat/2023/ plus any participant runs
dropped into data/ikat/2023/runs/. No fakes.

Run: .venv/bin/python scripts/sim2real_tau.py [n_turns]
"""
import sys
from pathlib import Path

from scipy.stats import kendalltau

from autosim.runtime import load_env, build_generator
from autosim.data.ikat_real import load_ikat_topics, load_qrels, load_ikat_run, load_runs_dir
from autosim.eval.trec_metrics import system_ndcg
from autosim.eval.assessor import AutoSimAssessor
from autosim.metrics.distributions import bootstrap_ci
from autosim.metrics.ranking import induced_ranking

DATA = Path("data/ikat/2023")


def main(n_turns: int = 12) -> None:
    topics = load_ikat_topics(DATA / "2023_test_topics.json")
    qrels = load_qrels(DATA / "2023-qrels.all-turns.txt")

    runs: dict[str, dict] = {}
    for f in sorted(DATA.glob("*.official.run.json")):
        name, r = load_ikat_run(f)
        runs[name] = r
    runs.update(load_runs_dir(DATA / "runs"))   # participant runs, when present
    print(f"systems: {len(runs)} | judged turns in qrels: {len(qrels)}")

    # --- reference ranking: NIST nDCG@5 ---
    ndcg = {name: system_ndcg(r, qrels, k=5)[0] for name, r in runs.items()}

    # --- AutoSim ranking: persona-conditioned satisfaction over a shared turn subset ---
    judged = [t for t in qrels if any(v > 0 for v in qrels[t].values()) and t in topics]
    common = [t for t in judged if all(t in r for r in runs.values())][:n_turns]
    print(f"AutoSim over {len(common)} shared judged turns, top-5 passages/turn")
    assessor = AutoSimAssessor(build_generator(load_env()))
    autosim = {name: assessor.score_system(r, topics, common, k=5) for name, r in runs.items()}

    # --- report ---
    print(f"\n{'system':55s} {'nDCG@5':>8s} {'AutoSim':>8s}")
    for name in induced_ranking(ndcg):
        print(f"{name[:55]:55s} {ndcg[name]:8.4f} {autosim[name]:8.4f}")

    names = list(runs)
    if len(names) >= 3:
        tau, p = kendalltau([ndcg[n] for n in names], [autosim[n] for n in names])
        per = [1.0 if (ndcg[a] > ndcg[b]) == (autosim[a] > autosim[b]) else 0.0
               for i, a in enumerate(names) for b in names[i + 1:]]
        lo, hi = bootstrap_ci(per, n=1000, seed=0)
        print(f"\nKendall-tau(AutoSim, NIST) = {tau:.3f} (p={p:.3f})  "
              f"pairwise-concordance 95% CI [{lo:.2f}, {hi:.2f}]")
    else:
        print(f"\n[need >=3 systems for tau; have {len(names)} — drop participant runs into {DATA/'runs'}]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
