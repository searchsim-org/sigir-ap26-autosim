"""Realism->validity effectiveness-bias test (reviewer-hardened).
For each held-out judged iKAT turn, BM25-retrieve over ClueWeb22-B for {real, resolved, default_sim,
terse_prompt_sim, trunc_default, weight_sim} and score nDCG@5 vs NIST qrels. Report, per year and
pooled: absolute-effectiveness bias (mean sim nDCG - mean real nDCG; 0 = unbiased), a paired Wilcoxon
test, the mean-ratio to real with a bootstrap CI (bounds the "fourfold" claim), and a query-length vs
nDCG analysis (isolates the terseness confound). The key control is terse_prompt_sim: if a hand-written
terse PROMPT also removes the bias, the effect is length, not weight-level substrate; if only weight_sim
removes it, realism (not mere terseness) is what buys validity.
Run: .venv/bin/python scripts/rv_validity.py [year ...]   (default: 2023 2024)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

from autosim.runtime import load_env, build_retrieval
from autosim.data.ikat_real import load_qrels

YEARS = sys.argv[1:] or ["2023", "2024"]
SIM_ARMS = ["default_sim", "terse_prompt_sim", "trunc_default", "weight_sim"]
ALL_ARMS = ["real", "resolved"] + SIM_ARMS


def ndcg(ranked_ids, rel, k=5):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked_ids[:k]))
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(sorted(rel.values(), reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def bias_stats(sim, real, rng, boot=5000):
    d = sim - real
    n = len(d)
    bb = np.array([d[rng.integers(0, n, n)].mean() for _ in range(boot)])
    lo, hi = np.percentile(bb, [2.5, 97.5])
    rb = np.array([sim[i].mean() / max(real[i].mean(), 1e-9) for i in (rng.integers(0, n, n) for _ in range(boot))])
    rlo, rhi = np.percentile(rb, [2.5, 97.5])
    try:
        p = float(wilcoxon(sim, real, zero_method="zsplit").pvalue) if np.any(sim != real) else 1.0
    except ValueError:
        p = float("nan")
    return {"mean_ndcg": round(float(sim.mean()), 3), "bias": round(float(d.mean()), 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)], "paired_wilcoxon_p": round(p, 4),
            "n_sim>real": int((sim > real).sum()), "n_real>sim": int((real > sim).sum()),
            "ratio_to_real": round(float(sim.mean() / max(real.mean(), 1e-9)), 2),
            "ratio_ci95": [round(float(rlo), 2), round(float(rhi), 2)]}


def main():
    env = load_env()
    bm25 = build_retrieval(env)
    pooled = []
    for y in YEARS:
        D = Path(f"data/ikat/{y}")
        qf = D / "ikat_rv_queries.jsonl"
        if not qf.exists():
            print(f"  (no queries for {y}, skip)"); continue
        qrels = load_qrels(next(D.glob("*qrels*")))
        for l in qf.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r["tid"] in qrels and any(v > 0 for v in qrels[r["tid"]].values()):
                r["_rel"] = qrels[r["tid"]]; r["_year"] = y
                r["trunc_default"] = " ".join(r.get("default_sim", "").split()[:max(1, len(r["real"].split()))])
                pooled.append(r)
    print(f"pooled held-out judged turns ({'+'.join(YEARS)}): {len(pooled)}", flush=True)
    have_terse = sum(1 for r in pooled if r.get("terse_prompt_sim"))
    print(f"  turns with terse_prompt_sim present: {have_terse}/{len(pooled)}", flush=True)

    def nd(q, rel):
        try:
            return ndcg([h.doc_id for h in bm25.search(q or " ", k=5)], rel)
        except Exception:
            return 0.0

    per = {k: np.array([nd(r.get(k, ""), r["_rel"]) for r in pooled]) for k in ALL_ARMS}
    wlen = {k: float(np.mean([len(str(r.get(k, "")).split()) for r in pooled])) for k in ALL_ARMS}
    rng = np.random.default_rng(0)

    print("mean nDCG@5:", {k: round(float(v.mean()), 3) for k, v in per.items()}, flush=True)
    print("mean query words:", {k: round(v, 1) for k, v in wlen.items()}, flush=True)

    out = {"years": YEARS, "n": len(pooled), "mean_ndcg": {k: round(float(v.mean()), 3) for k, v in per.items()},
           "mean_query_words": {k: round(v, 1) for k, v in wlen.items()}, "bias_pooled": {}, "bias_by_year": {}}

    # per-turn fidelity (correlation vs real-query nDCG)
    out["fidelity"] = {}
    for k in SIM_ARMS:
        pr = float(pearsonr(per[k], per["real"])[0]); sr = float(spearmanr(per[k], per["real"])[0])
        out["fidelity"][k] = {"pearson": round(pr, 3), "spearman": round(sr, 3)}
        print(f"  fidelity {k:16s} pearson={pr:+.3f}  spearman={sr:+.3f}", flush=True)

    # absolute-effectiveness bias, pooled + per year
    print("\n== absolute-effectiveness bias (mean sim nDCG - mean real nDCG; 0 = unbiased) ==", flush=True)
    print(f"  real mean nDCG@5 = {per['real'].mean():.3f}", flush=True)
    for k in SIM_ARMS:
        out["bias_pooled"][k] = bias_stats(per[k], per["real"], rng)
        s = out["bias_pooled"][k]
        print(f"  {k:16s} bias={s['bias']:+.3f} CI{s['ci95']} ratio={s['ratio_to_real']}x CI{s['ratio_ci95']} "
              f"wilcoxon_p={s['paired_wilcoxon_p']} (sim>real {s['n_sim>real']}, real>sim {s['n_real>sim']})", flush=True)
    for y in YEARS:
        mask = np.array([r["_year"] == y for r in pooled])
        if mask.sum() == 0:
            continue
        out["bias_by_year"][y] = {"n": int(mask.sum()), "real_mean": round(float(per["real"][mask].mean()), 3)}
        for k in SIM_ARMS:
            if per[k][mask].sum() == 0 and k == "terse_prompt_sim":
                continue
            out["bias_by_year"][y][k] = bias_stats(per[k][mask], per["real"][mask], rng)
        print(f"  [{y}] n={int(mask.sum())} " +
              " ".join(f"{k}={out['bias_by_year'][y][k]['bias']:+.3f}" for k in SIM_ARMS if k in out["bias_by_year"][y]),
              flush=True)

    # length-confound: how much of nDCG is explained by pure query length across all sim queries?
    lens, ndcgs = [], []
    for k in SIM_ARMS + ["real"]:
        lens += [len(str(r.get(k, "")).split()) for r in pooled]
        ndcgs += list(per[k])
    lens, ndcgs = np.array(lens, float), np.array(ndcgs, float)
    lp = float(pearsonr(lens, ndcgs)[0])
    out["length_vs_ndcg_pearson"] = round(lp, 3)
    print(f"\n== length confound: pearson(query_words, nDCG@5) across all arms = {lp:+.3f} (r^2={lp*lp:.3f}) ==", flush=True)

    out["per_turn"] = {k: [round(float(x), 4) for x in v] for k, v in per.items()}
    Path("data/ikat").mkdir(exist_ok=True)
    (Path("data/ikat") / "rv_validity_pooled.json").write_text(json.dumps(out, indent=1))
    print("saved -> data/ikat/rv_validity_pooled.json", flush=True)


if __name__ == "__main__":
    main()
