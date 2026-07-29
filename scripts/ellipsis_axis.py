"""TODO-1: a fourth behavioral axis, query self-containment (D2 reformulation). Real conversational
users write elliptical queries that lean on the dialogue context; a verbose simulator restates the full,
self-contained need. We measure self-containment as word-overlap (Jaccard) between each arm's next-turn
query and its RESOLVED (context-completed) form: high overlap = self-contained, low = elliptical. If the
weight-level simulator matches real users' ellipsis where the default does not, the substrate effect
extends to reformulation behavior. (D5 satisfaction / D6 stopping need turn-level labels iKAT's held-out
set lacks, so they stay future work.)
Run: .venv/bin/python scripts/ellipsis_axis.py
"""
import json
import re
from pathlib import Path

import numpy as np

WORD = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(WORD.findall((s or "").lower()))


def jac(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def main():
    turns = []
    for y in ["2023", "2024"]:
        f = Path(f"data/ikat/{y}/ikat_rv_queries.jsonl")
        if f.exists():
            turns += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    turns = [t for t in turns if t.get("resolved")]
    print(f"turns with a resolved form: {len(turns)}", flush=True)

    arms = ["real", "default_sim", "terse_prompt_sim", "weight_sim"]
    out = {"n": len(turns), "self_containment_overlap_with_resolved": {}, "mean_words": {}}
    rng = np.random.default_rng(0)
    for a in arms:
        v = np.array([jac(t.get(a, ""), t["resolved"]) for t in turns])
        n = len(v)
        bb = np.array([v[rng.integers(0, n, n)].mean() for _ in range(5000)])
        lo, hi = np.percentile(bb, [2.5, 97.5])
        out["self_containment_overlap_with_resolved"][a] = {"mean": round(float(v.mean()), 3),
                                                            "ci95": [round(float(lo), 3), round(float(hi), 3)]}
        out["mean_words"][a] = round(float(np.mean([len((t.get(a) or "").split()) for t in turns])), 1)
    # gap of each simulator to real (0 = matches real ellipsis)
    real_m = out["self_containment_overlap_with_resolved"]["real"]["mean"]
    for a in ["default_sim", "terse_prompt_sim", "weight_sim"]:
        out[f"gap_to_real_{a}"] = round(out["self_containment_overlap_with_resolved"][a]["mean"] - real_m, 3)
    print("self-containment (Jaccard overlap with the resolved full-need form; higher = more self-contained):", flush=True)
    for a in arms:
        s = out["self_containment_overlap_with_resolved"][a]
        print(f"  {a:12s} {s['mean']:.3f}  95%CI{s['ci95']}  (mean words {out['mean_words'][a]})", flush=True)
    print(f"gap to real: default {out['gap_to_real_default_sim']:+.3f}, weight {out['gap_to_real_weight_sim']:+.3f} "
          f"(closer to 0 = matches real ellipsis)", flush=True)
    Path("data/ikat/ellipsis_axis.json").write_text(json.dumps(out, indent=1))
    print("saved -> data/ikat/ellipsis_axis.json", flush=True)


if __name__ == "__main__":
    main()
