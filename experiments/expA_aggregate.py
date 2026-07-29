"""Aggregate the disjoint-seed replicates of expA_adversarial2 into mean +/- std.
Reports, per arm, the round-0 -> final change in AUC and diversity, and the headline
naive-vs-coverage divergence in distinct-2 (the collapse the coverage term prevents).

Usage: python expA_aggregate.py s0 s1 s2   (suffixes of expA_adv2*.json to pool)
"""
import glob
import json
import os
import sys

import numpy as np

RUNS = os.path.expanduser("~/v2sim/runs")
sufs = sys.argv[1:] or ["_s0", "_s1", "_s2"]
sufs = [s if s.startswith("_") else "_" + s for s in sufs]
runs = []
for s in sufs:
    p = f"{RUNS}/expA_adv2{s}.json"
    if os.path.exists(p):
        runs.append(json.load(open(p)))
print(f"pooled {len(runs)} seeds: {[r['seed'] for r in runs]}\n")


def arr(arm, field, where):  # where in {'first','last'}
    out = []
    for r in runs:
        t = r[arm]["traj"]; out.append(t[0][field] if where == "first" else t[-1][field])
    return np.array(out, float)


def ms(a):
    return f"{a.mean():.3f} +/- {a.std():.3f}"


for arm in ["naive", "coverage"]:
    print(f"== {arm} ==")
    for f in ["auc", "distinct2", "len_entropy", "embed_dispersion"]:
        print(f"   {f:16s} {ms(arr(arm, f, 'first'))}  ->  {ms(arr(arm, f, 'last'))}")
    print()

# headline: change in distinct2 from round0 to final, per arm, paired across seeds
d_naive = arr("naive", "distinct2", "last") - arr("naive", "distinct2", "first")
d_cov = arr("coverage", "distinct2", "last") - arr("coverage", "distinct2", "first")
print("HEADLINE  Delta-distinct2 (final - round0), paired per seed:")
print(f"   naive   : {ms(d_naive)}   (negative = collapse)")
print(f"   coverage: {ms(d_cov)}")
print(f"   gap (coverage - naive): {ms(d_cov - d_naive)}   [>0 = coverage preserves more diversity]")
# real diversity target
rp2 = np.array([r["real_prof"]["distinct2"] for r in runs], float)
print(f"\n   real users' distinct2 target: {ms(rp2)}")
