"""Claim B (the paper's core): REAL self-improvement via GEPA-style search with a
DENSE, low-variance selection signal.

Each cycle: generate N candidate assessor criteria (reflective), score each on the
calibration split by the per-(turn,system) correlation between AutoSim satisfaction
and per-turn nDCG@5 (hundreds of points -> stable selection), SELECT THE BEST, gate
it (no regression). VALIDATE on the system-level held-out Kendall-tau (the quantity
we ultimately care about) it never optimised against.

Ablation: full (dense calibration selection) vs static vs no-calibration (blind).
`start=weak` exposes a real gap; generator (qwen) assesses, judge (gemma) proposes.

Run: .venv/bin/python scripts/self_improve.py [n_cal] [n_hold] [K] [start] [N]
"""
import json
import random
import sys
from pathlib import Path

from scipy.stats import kendalltau, spearmanr

from autosim.runtime import load_env, build_generator, build_judge, build_retrieval
from autosim.data.ikat_real import load_ikat_topics, load_qrels
from autosim.eval.trec_metrics import ndcg_at_k
from autosim.eval.assessor import AutoSimAssessor, DEFAULT_CRITERIA

DATA = Path("data/ikat/2023")
VCACHE = DATA / "pool_cache_v2.jsonl"
TEXTCACHE = DATA / "assessor_inputs.json"
SYSTEMS = ["bm25(raw)", "bm25(resolved)", "bm25(ptkb)", "splade(raw)", "splade(resolved)",
           "splade(ptkb)", "rrf(resolved)", "rrf(ptkb)", "bm25+rerank", "splade+rerank",
           "rrf+rerank", "splade(ptkb)+rerank"]
WEAK_CRITERIA = "Rate how satisfied you are with these passages."


def load_v2():
    recs = {}
    for line in VCACHE.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["tid"]] = r
    return recs


def build_text_cache(recs, bm25):
    if TEXTCACHE.exists():
        return json.loads(TEXTCACHE.read_text())
    cache = {}
    for tid, rec in recs.items():
        needed = set()
        for ids in rec["rankings"].values():
            needed.update(ids[:5])
        texts = bm25.mget_texts(list(needed))
        cache[tid] = {s: [texts.get(sid, "") for sid in rec["rankings"][s][:5]] for s in SYSTEMS}
    TEXTCACHE.write_text(json.dumps(cache))
    return cache


def per_turn_ndcg(recs, qrels):
    """{tid: {system: condensed nDCG@5 for that system on that turn}} over judged turns."""
    out = {}
    for tid, rec in recs.items():
        if tid in qrels and any(v > 0 for v in qrels[tid].values()):
            out[tid] = {s: ndcg_at_k([sid for sid in rec["rankings"][s] if sid in qrels[tid]],
                                     qrels[tid], 5) for s in SYSTEMS}
    return out


def assess_sat(assessor, textcache, topics, tids):
    """{tid: {system: satisfaction}}."""
    return {t: {s: assessor.score_turn(topics[t]["ptkb"], topics[t]["resolved"], textcache[t][s], k=5)
                for s in SYSTEMS} for t in tids}


def system_means(sat, tids):
    return {s: sum(sat[t][s] for t in tids) / len(tids) for s in SYSTEMS}


def dense_corr(sat, ndcg_ts, tids):
    """Spearman over all (turn, system) pairs -> low-variance SELECTION signal."""
    xs, ys = [], []
    for t in tids:
        for s in SYSTEMS:
            xs.append(sat[t][s])
            ys.append(ndcg_ts[t][s])
    r, _ = spearmanr(xs, ys)
    return float(r) if r == r else 0.0


def sys_tau(sat, tids, ndcg_ts):
    """System-level Kendall-tau (the VALIDATION metric) over a turn subset."""
    sm = system_means(sat, tids)
    nd = {s: sum(ndcg_ts[t][s] for t in tids) / len(tids) for s in SYSTEMS}
    t, _ = kendalltau([sm[s] for s in SYSTEMS], [nd[s] for s in SYSTEMS])
    return float(t) if t == t else 0.0


def disagreements(sm, nd, m=6):
    out = []
    for i, a in enumerate(SYSTEMS):
        for b in SYSTEMS[i + 1:]:
            if (sm[a] > sm[b]) != (nd[a] > nd[b]) and sm[a] != sm[b]:
                hi = a if nd[a] > nd[b] else b
                lo = b if hi == a else a
                out.append(f"- experts rank '{hi}' ABOVE '{lo}', but the simulator ranked '{lo}' higher.")
    return "\n".join(out[:m]) or "(no disagreements)"


def reflect_n(judge, criteria, disag, n=5):
    base = (
        "You tune the CRITERIA a simulated user applies when rating search-result passages, so its "
        "judgments better match expert relevance.\n"
        f"Current criteria:\n{criteria}\n\n"
        f"On calibration data the simulator MISRANKED these system pairs:\n{disag}\n\n"
        "Write improved criteria (2-4 sentences): be specific about what makes a passage genuinely "
        "satisfying (faithful, on-topic, directly answers the specific need) versus superficially "
        "matching (keyword overlap, tangential, verbose). Output ONLY the criteria."
    )
    cands = []
    for _ in range(n):
        new = judge.complete(base, temperature=0.8).text.strip()
        if 20 < len(new) < 800 and new not in cands:
            cands.append(new)
    return cands or [criteria]


def main(n_cal=40, n_hold=40, K=2, start="weak", N=5):
    env = load_env()
    gen = build_generator(env)
    judge = build_judge(env)
    bm25 = build_retrieval(env)
    topics = load_ikat_topics(DATA / "2023_test_topics.json")
    qrels = load_qrels(DATA / "2023-qrels.all-turns.txt")
    recs = load_v2()
    textcache = build_text_cache(recs, bm25)
    ndcg_ts = per_turn_ndcg(recs, qrels)

    tids = [t for t in recs if t in topics and t in ndcg_ts]
    random.Random(13).shuffle(tids)
    cal, hold = tids[:n_cal], tids[n_cal:n_cal + n_hold]
    start_crit = WEAK_CRITERIA if start == "weak" else DEFAULT_CRITERIA
    print(f"start={start} cal={len(cal)} hold={len(hold)} K={K} N={N} | dense selection", flush=True)

    def hold_tau(crit):
        return sys_tau(assess_sat(AutoSimAssessor(gen, crit), textcache, topics, hold), hold, ndcg_ts)

    nd_cal = {s: sum(ndcg_ts[t][s] for t in cal) / len(cal) for s in SYSTEMS}
    tau0 = hold_tau(start_crit)

    # FULL: N candidates/cycle, select best by DENSE calibration correlation, gate
    P, full = start_crit, [tau0]
    for k in range(K):
        sat_cur = assess_sat(AutoSimAssessor(gen, P), textcache, topics, cal)
        cur = sys_tau(sat_cur, cal, ndcg_ts)       # select on system-level cal tau (the right objective)
        cands = reflect_n(judge, P, disagreements(system_means(sat_cur, cal), nd_cal), n=N)
        scored = [(c, sys_tau(assess_sat(AutoSimAssessor(gen, c), textcache, topics, cal), cal, ndcg_ts))
                  for c in cands]
        best_c, best = max(scored, key=lambda x: x[1])
        if best >= cur:
            P = best_c
        full.append(hold_tau(P))
        print(f"[full k={k + 1}] cur_caltau={cur:+.3f} best_caltau={best:+.3f} "
              f"{'ACCEPT' if best >= cur else 'reject'} hold_tau={full[-1]:+.3f}", flush=True)

    # NO-CALIBRATION: N candidates, BLIND selection
    P2, nocal = start_crit, [tau0]
    for k in range(K):
        P2 = reflect_n(judge, P2, "(no calibration feedback provided)", n=N)[0]
        nocal.append(hold_tau(P2))
        print(f"[nocal k={k + 1}] hold_tau={nocal[-1]:+.3f}", flush=True)

    static = [round(float(tau0), 3)] * (K + 1)
    out = ("held-out system-level Kendall-tau over GEPA self-improvement cycles (system-tau selection) "
           f"(start={start}, cal={len(cal)}, hold={len(hold)}, K={K}, N={N}, {len(SYSTEMS)} systems):\n"
           f"  full        : {[round(float(x), 3) for x in full]}\n"
           f"  static      : {static}\n"
           f"  no-calib    : {[round(float(x), 3) for x in nocal]}\n\n"
           f"start criteria:\n{start_crit}\n\nevolved criteria (final):\n{P}\n")
    print("\n" + out, flush=True)
    (DATA / f"self_improve_{start}_sys88.txt").write_text(out)


if __name__ == "__main__":
    a = sys.argv[1:]
    nums = [int(x) for x in a[:3]] if len(a) >= 3 else []
    st = a[3] if len(a) >= 4 else "weak"
    nn = int(a[4]) if len(a) >= 5 else 5
    (main(*nums, start=st, N=nn) if nums else main(start=st, N=nn))
