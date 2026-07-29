"""LEVER-1: make the effectiveness bias consequential. Two decision-relevant views, from the
per-turn nDCG@5 already computed in rv_validity_pooled.json:
(a) Failure-masking: on turns where the REAL user's query retrieves nothing relevant (nDCG@5=0),
    how often does each simulator's query still "succeed" (>0)? A default sim that succeeds there
    HIDES exactly the failures an evaluation should surface.
(b) Ship-it decision: at a quality bar, does evaluating with each simulator PASS a system that real
    users find broken? The simulator's realism decides whether the ship/no-ship call is right.
Run: .venv/bin/python scripts/consequential_bias.py
"""
import json
from pathlib import Path

import numpy as np

d = json.loads(Path("data/ikat/rv_validity_pooled.json").read_text())
pt = d["per_turn"]
arms = {k: np.array(pt[k], float) for k in ["real", "default_sim", "terse_prompt_sim", "weight_sim"]}
real = arms["real"]
n = len(real)
print(f"n held-out judged turns = {n}\n")

# (a) failure-masking
fails = real == 0.0
print(f"(a) FAILURE-MASKING")
print(f"  real user retrieves NOTHING relevant (nDCG@5=0) on {fails.sum()}/{n} = {fails.mean():.0%} of turns")
out = {"n": n, "real_fail_rate": round(float(fails.mean()), 3), "masking": {}, "shipit": {}}
for k in ["default_sim", "terse_prompt_sim", "weight_sim"]:
    a = arms[k]
    succ = (a[fails] > 0.0)
    out["masking"][k] = {"succeeds_on_real_fail_%": round(float(succ.mean()), 3),
                         "mean_ndcg_on_real_fail": round(float(a[fails].mean()), 3)}
    print(f"  {k:16s} 'succeeds' (nDCG>0) on {succ.mean():5.0%} of those real-fail turns; mean nDCG there {a[fails].mean():.3f}")

# (b) ship-it decision at a quality bar
print(f"\n(b) SHIP-IT DECISION (evaluate the BM25 system with each simulator; is the call right?)")
for bar in [0.05, 0.10]:
    verdicts = {k: ("PASS" if arms[k].mean() >= bar else "FAIL") for k in arms}
    out["shipit"][f"bar_{bar}"] = {k: {"mean": round(float(arms[k].mean()), 3), "verdict": verdicts[k]} for k in arms}
    print(f"  bar nDCG@5>={bar}:  " + "  ".join(f"{k.split('_')[0]}={arms[k].mean():.3f}({verdicts[k]})" for k in arms))
    wrong = [k for k in ["default_sim", "terse_prompt_sim", "weight_sim"] if verdicts[k] != verdicts["real"]]
    print(f"     -> disagree with the REAL-user verdict ({verdicts['real']}): {', '.join(wrong) if wrong else 'none'}")

# false-success rate summary (headline)
fp_default = (arms["default_sim"][fails] > 0).mean()
fp_weight = (arms["weight_sim"][fails] > 0).mean()
print(f"\nHEADLINE: on turns real users get nothing, the DEFAULT sim reports a relevant hit "
      f"{fp_default:.0%} of the time (masking the failure); the WEIGHT-level sim only {fp_weight:.0%}.")
Path("data/ikat").mkdir(exist_ok=True)
Path("data/ikat/consequential_bias.json").write_text(json.dumps(out, indent=1))
print("saved -> data/ikat/consequential_bias.json")
