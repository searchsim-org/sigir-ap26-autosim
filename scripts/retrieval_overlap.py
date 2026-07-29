"""LEVER-3: a second behavioral axis beyond utterance style (D1) -> retrieval FOOTPRINT (D4-like):
does the simulator surface the same documents a real user's query surfaces? For each held-out turn,
BM25-retrieve the top-5 doc ids for {real, default_sim, terse_prompt_sim, weight_sim} and measure
overlap@5 with the real user's set. If the weight-level sim overlaps real far more than the
prompt-level arms, realism extends past phrasing to WHAT the user retrieves -- a second dimension.
Run: .venv/bin/python scripts/retrieval_overlap.py [year ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

from autosim.runtime import load_env, build_retrieval

YEARS = sys.argv[1:] or ["2023", "2024"]
ARMS = ["real", "default_sim", "terse_prompt_sim", "weight_sim"]


def main():
    env = load_env()
    bm25 = build_retrieval(env)

    def topk(q, k=5):
        try:
            return [h.doc_id for h in bm25.search(q or " ", k=k)]
        except Exception:
            return []

    turns = []
    for y in YEARS:
        f = Path(f"data/ikat/{y}/ikat_rv_queries.jsonl")
        if f.exists():
            turns += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    print(f"held-out turns: {len(turns)}", flush=True)

    ov = {a: [] for a in ARMS if a != "real"}
    for t in turns:
        real_ids = set(topk(t.get("real", "")))
        if not real_ids:
            continue
        for a in ov:
            sim_ids = set(topk(t.get(a, "")))
            ov[a].append(len(real_ids & sim_ids) / 5.0)          # overlap@5 with real

    rng = np.random.default_rng(0)
    out = {"years": YEARS, "n": len(ov["default_sim"]), "overlap_at5_vs_real": {}}
    print("mean overlap@5 with the REAL user's retrieved set (higher = same retrieval footprint):", flush=True)
    for a in ov:
        v = np.array(ov[a]); n = len(v)
        bb = np.array([v[rng.integers(0, n, n)].mean() for _ in range(5000)])
        lo, hi = np.percentile(bb, [2.5, 97.5])
        out["overlap_at5_vs_real"][a] = {"mean": round(float(v.mean()), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)]}
        print(f"  {a:16s} {v.mean():.3f}  95%CI[{lo:.3f},{hi:.3f}]", flush=True)
    Path("data/ikat/retrieval_overlap.json").write_text(json.dumps(out, indent=1))
    print("saved -> data/ikat/retrieval_overlap.json", flush=True)


if __name__ == "__main__":
    main()
