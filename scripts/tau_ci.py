"""Honest inference for the sim-to-real Kendall-tau (fixes the mislabeled CI).

The published draft reported a *concordance* bootstrap interval next to tau, which
is NOT a CI on tau. This recomputes, from the cache only (no API):
  - tau-b (ties-corrected) for condensed and raw nDCG@5,
  - a Monte-Carlo permutation p-value (the test the draft promised),
  - a proper SYSTEM-LEVEL bootstrap 95% CI on tau (resample the 12 systems),
  - leave-one-out (jackknife) tau range,
  - leave-one-factor-out sub-pool taus (does tau survive without the easy factors?).

Run: .venv/bin/python scripts/tau_ci.py
"""
import json
import os
import random
from pathlib import Path

from scipy.stats import kendalltau

from autosim.data.ikat_real import load_qrels
from autosim.eval.trec_metrics import system_ndcg, system_ndcg_condensed

YEAR = os.environ.get("AUTOSIM_YEAR", "2023")
DATA = Path(f"data/ikat/{YEAR}")
CACHE = DATA / ("pool_cache_v2.jsonl" if YEAR == "2023" else f"pool_cache_{YEAR}.jsonl")
OUT = DATA / ("tau_ci.txt" if YEAR == "2023" else f"tau_ci_{YEAR}.txt")
QRELS = next(DATA.glob("*qrels*"))
SYSTEMS = ["bm25(raw)", "bm25(resolved)", "bm25(ptkb)",
           "splade(raw)", "splade(resolved)", "splade(ptkb)",
           "rrf(resolved)", "rrf(ptkb)",
           "bm25+rerank", "splade+rerank", "rrf+rerank", "splade(ptkb)+rerank"]


def load_scores():
    done = {}
    for line in CACHE.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            done[rec["tid"]] = rec
    qrels = load_qrels(QRELS)
    runs = {n: {} for n in SYSTEMS}
    autos = {n: [] for n in SYSTEMS}
    for tid, rec in done.items():
        for n in SYSTEMS:
            if n in rec["rankings"]:
                runs[n][tid] = [(sid, "", 0.0) for sid in rec["rankings"][n]]
                autos[n].append(rec["autosim"][n])
    ndcg = {n: system_ndcg(runs[n], qrels, 5)[0] for n in SYSTEMS}
    ndcg_c = {n: system_ndcg_condensed(runs[n], qrels, 5)[0] for n in SYSTEMS}
    autosim = {n: sum(autos[n]) / len(autos[n]) for n in SYSTEMS}
    return ndcg, ndcg_c, autosim


def perm_p(x, y, R=200000, seed=0):
    obs = abs(kendalltau(x, y)[0])
    rng = random.Random(seed)
    yy = list(y)
    hits = 0
    for _ in range(R):
        rng.shuffle(yy)
        if abs(kendalltau(x, yy)[0]) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (R + 1)


def boot_ci(x, y, B=10000, seed=0):
    rng = random.Random(seed)
    n = len(x)
    taus = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        xs = [x[i] for i in idx]
        ys = [y[i] for i in idx]
        t = kendalltau(xs, ys)[0]
        if t == t:                      # drop nan (degenerate resample)
            taus.append(t)
    taus.sort()
    lo = taus[int(0.025 * len(taus))]
    hi = taus[int(0.975 * len(taus))]
    return lo, hi


def jackknife(names, ref, au):
    out = []
    for drop in names:
        keep = [n for n in names if n != drop]
        t = kendalltau([ref[n] for n in keep], [au[n] for n in keep])[0]
        out.append(t)
    return min(out), sum(out) / len(out), max(out)


def main():
    ndcg, ndcg_c, autosim = load_scores()
    lines = []
    for label, ref in [("condensed nDCG@5", ndcg_c), ("raw nDCG@5", ndcg)]:
        x = [ref[n] for n in SYSTEMS]
        y = [autosim[n] for n in SYSTEMS]
        tau, p_asym = kendalltau(x, y)
        p_perm = perm_p(x, y)
        lo, hi = boot_ci(x, y)
        jmin, jmean, jmax = jackknife(SYSTEMS, ref, autosim)
        lines += [
            f"=== {label} (N={len(SYSTEMS)} systems) ===",
            f"  Kendall tau-b              = {tau:+.3f}",
            f"  asymptotic p               = {p_asym:.4f}",
            f"  permutation p (R=2e5)      = {p_perm:.4f}",
            f"  system-level bootstrap 95% CI on tau = [{lo:+.3f}, {hi:+.3f}]",
            f"  leave-one-out tau          = [{jmin:+.3f}, {jmax:+.3f}] (mean {jmean:+.3f})",
        ]
    # leave-one-factor-out sub-pools (do the easy factors drive tau?)
    subpools = {
        "no rerank (8 sys)": [n for n in SYSTEMS if "rerank" not in n],
        "no ptkb   (8 sys)": [n for n in SYSTEMS if "ptkb" not in n],
        "no raw    (10 sys)": [n for n in SYSTEMS if "raw" not in n],
        "resolved+rerank only": ["bm25(resolved)", "splade(resolved)", "rrf(resolved)",
                                  "bm25+rerank", "splade+rerank", "rrf+rerank"],
    }
    lines.append("=== leave-one-factor-out (condensed nDCG@5) ===")
    for name, sub in subpools.items():
        t, p = kendalltau([ndcg_c[n] for n in sub], [autosim[n] for n in sub])
        lines.append(f"  {name:22s} N={len(sub):2d}  tau={t:+.3f} (p={p:.3f})")
    out = "\n".join(lines)
    print(out, flush=True)
    OUT.write_text(out + "\n")


if __name__ == "__main__":
    main()
