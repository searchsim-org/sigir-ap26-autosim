"""Score collected human-study responses. Each input JSON is one annotator's download from the app.
Reports per-condition accuracy vs chance (0.5), with an inattentive-annotator exclusion on the control.
Run: python scripts/score_human_study.py responses/*.json
"""
import json
import math
import sys
from statistics import NormalDist

files = sys.argv[1:]
if not files:
    print("usage: score_human_study.py <response1.json> [response2.json ...]"); sys.exit(1)

by_cond = {"weight": [], "default": []}          # per-annotator accuracy lists
kept = 0
for f in files:
    r = json.loads(open(f).read())["responses"]
    acc = {c: [x["correct"] for x in r if x["condition"] == c] for c in by_cond}
    ctrl = sum(acc["default"]) / len(acc["default"]) if acc["default"] else 0
    if ctrl < 0.6:                               # inattentive: fails the easy control
        print(f"  (excluded {f}: control acc {ctrl:.2f} < 0.60)"); continue
    kept += 1
    for c in by_cond:
        if acc[c]:
            by_cond[c].append(sum(acc[c]) / len(acc[c]))


def test(vals):
    if not vals:
        return None
    m = sum(vals) / len(vals)
    n = len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    # one-sample t vs 0.5 -> normal approx p
    z = (m - 0.5) / (sd / math.sqrt(n)) if sd > 0 and n > 1 else 0.0
    p = 2 * (1 - NormalDist().cdf(abs(z)))
    return {"n_annotators": n, "mean_acc": round(m, 3), "sd": round(sd, 3), "p_vs_chance": round(p, 3)}


print(f"\nkept {kept}/{len(files)} annotators (control acc >= 0.60)")
for c in ["weight", "default"]:
    t = test(by_cond[c])
    print(f"  {c:8s} (real vs {c}-sim): {t}")
print("\nExpected if the weight-level simulator is indistinguishable: weight mean_acc ~ 0.5 (p>0.05), "
      "default mean_acc >> 0.5 (p<0.001).")
