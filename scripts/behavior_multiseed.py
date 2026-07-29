"""P1.7: run the behavioral self-improvement loop over MANY seeds and report
per-arm mean +/- SD of the held-out CSG, so 'gated = stable, no-calib = erratic'
becomes a quantitative variance claim rather than one trajectory.

Each seed varies the cal/hold split and the LLM sampling. Parallelizes generation.
Run: .venv/bin/python scripts/behavior_multiseed.py [n_cal] [n_hold] [K] [N] [n_seeds]
Out: data/ikat/2023/behavior_multiseed.txt (+ .json)
"""
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autosim.runtime import load_env, build_generator, build_judge
from scripts.self_improve_behavior import (load_turns, gen_utt, csg, style_gap,
                                           reflect_n, DEFAULT_STYLE)

DATA = Path("data/ikat/2023")
WORKERS = 24
SEEDS = [13, 29, 47, 71, 101, 211, 307, 401, 503, 601, 701, 809, 907]


def gen_all_par(gen, style, recs):
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        return list(ex.map(lambda r: gen_utt(gen, style, r), recs))


def run_seed(gen, judge, seed, n_cal, n_hold, K, N):
    turns = load_turns()
    random.Random(seed).shuffle(turns)
    cal, hold = turns[:n_cal], turns[n_cal:n_cal + n_hold]
    real_cal = [r["real"] for r in cal]
    real_hold = [r["real"] for r in hold]

    def hold_csg(style):
        return csg(gen_all_par(gen, style, hold), real_hold)

    csg0 = hold_csg(DEFAULT_STYLE)
    # FULL (gated)
    P, full = DEFAULT_STYLE, [csg0]
    for k in range(K):
        sim_cal = gen_all_par(gen, P, cal)
        cur = csg(sim_cal, real_cal)
        cands = reflect_n(judge, P, style_gap(sim_cal, real_cal), N)
        scored = [(c, csg(gen_all_par(gen, c, cal), real_cal)) for c in cands]
        bc, best = max(scored, key=lambda x: x[1])
        if best >= cur:
            P = bc
        full.append(hold_csg(P))
    # NO-CALIB (blind)
    P2, nocal = DEFAULT_STYLE, [csg0]
    for k in range(K):
        P2 = reflect_n(judge, P2, "(no calibration feedback provided)", N)[0]
        nocal.append(hold_csg(P2))
    return {"full": [float(x) for x in full], "nocal": [float(x) for x in nocal],
            "static": float(csg0), "final_style": P}


def main(n_cal=40, n_hold=40, K=2, N=4, n_seeds=5):
    env = load_env()
    gen, judge = build_generator(env), build_judge(env)
    seeds = SEEDS[:n_seeds]
    results = []
    for s in seeds:
        r = run_seed(gen, judge, s, n_cal, n_hold, K, N)
        results.append(r)
        print(f"seed {s}: full {[round(x, 3) for x in r['full']]}  "
              f"nocal {[round(x, 3) for x in r['nocal']]}  static {r['static']:.3f}", flush=True)

    def stat(vals):
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    full_final = [r["full"][-1] for r in results]
    nocal_final = [r["nocal"][-1] for r in results]
    full_best = [max(r["full"]) for r in results]
    nocal_best = [max(r["nocal"]) for r in results]
    static = [r["static"] for r in results]
    fm, fs = stat(full_final); nm, ns = stat(nocal_final); sm, ss = stat(static)
    from scipy.stats import ttest_rel
    p_fs = float(ttest_rel(full_final, static).pvalue) if len(full_final) > 1 else float("nan")
    p_ns = float(ttest_rel(nocal_final, static).pvalue) if len(nocal_final) > 1 else float("nan")
    lines = [
        f"Behavioral self-improvement over {len(seeds)} seeds (cal={n_cal}, hold={n_hold}, K={K}, N={N})",
        f"  held-out CSG at final cycle:",
        f"    full (gated)   mean={fm:.3f} SD={fs:.3f}  range=[{min(full_final):.3f},{max(full_final):.3f}]",
        f"    no-calib       mean={nm:.3f} SD={ns:.3f}  range=[{min(nocal_final):.3f},{max(nocal_final):.3f}]",
        f"    static         mean={sm:.3f} SD={ss:.3f}",
        f"  best-over-cycles CSG: full mean={stat(full_best)[0]:.3f} SD={stat(full_best)[1]:.3f}  "
        f"no-calib mean={stat(nocal_best)[0]:.3f} SD={stat(nocal_best)[1]:.3f}",
        f"  reliability: full SD={fs:.3f} vs no-calib SD={ns:.3f}  "
        f"(gate {'reduces' if fs < ns else 'does NOT reduce'} final-cycle variance)",
        f"  paired t-test vs static (n={len(static)} seeds): full p={p_fs:.4f}, no-calib p={p_ns:.4f}",
    ]
    out = "\n".join(lines)
    print("\n" + out, flush=True)
    (DATA / "behavior_multiseed.txt").write_text(out + "\n")
    (DATA / "behavior_multiseed.json").write_text(json.dumps(
        {"seeds": seeds, "results": results, "p_full_vs_static": p_fs, "p_nocal_vs_static": p_ns}, indent=2))


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:6]]
    main(*a) if a else main()
