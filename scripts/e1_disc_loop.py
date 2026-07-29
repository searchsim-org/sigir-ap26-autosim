"""E1-strong (no-invisible-walls #6: test the STRONG version of the idea): run the metabolism
loop with the DISCRIMINATOR as the selection signal (prompt-level DIAL), instead of the coarse
CSG-D1. Each cycle: reflect on the most-detectable simulated utterances, propose style edits,
keep the one that best lowers a real-vs-sim discriminator AUC on calibration; gate on no-regression.
Track HELD-OUT discriminator AUC over cycles (lower = more human). Answers: how far can
prompt-level self-improvement push realism on the metric that actually matters?

Run: .venv/bin/python scripts/e1_disc_loop.py [n_cal] [n_hold] [K] [N]
"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from autosim.runtime import load_env, build_generator, build_judge

YEAR = os.environ.get("AUTOSIM_YEAR", "2023")
DATA = Path(f"data/ikat/{YEAR}")
DEFAULT_STYLE = "Write naturally as this user would."


def load_turns():
    topics = json.loads((DATA / f"{YEAR}_test_topics.json").read_text())
    out = []
    for tp in topics:
        ptkb = list(tp.get("ptkb", {}).values())
        hist = []
        for t in tp.get("turns", []):
            out.append({"ptkb": ptkb, "history": "\n".join(hist[-4:]) or "(start of conversation)",
                        "real": t.get("utterance", ""),
                        "resolved": t.get("resolved_utterance", t.get("utterance", ""))})
            hist.append(f"User: {t.get('utterance', '')}")
            if t.get("response"):
                hist.append(f"Assistant: {t['response'][:200]}")
    return [r for r in out if r["real"]]


def gen_utt(gen, style, rec):
    ask = (f"You are a real user in an ongoing conversational web-search session.\n"
           f"Your background:\n- " + "\n- ".join(rec["ptkb"][:8]) + "\n\n"
           f"Conversation so far:\n{rec['history']}\n\n"
           f"Your next information need (express it as YOUR next message, in your own words): "
           f"{rec['resolved']}\n\n{style}\n"
           f"Write ONLY your next message to the assistant (no quotes, no preamble).")
    return gen.complete(ask, temperature=0.7).text.strip().strip('"').splitlines()[0][:300]


def gen_arm(gen, style, recs, workers=12):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda r: gen_utt(gen, style, r), recs))


def disc(real, sim):
    """TF-IDF char-ngram real-vs-sim discriminator: return (5-fold AUC, p_sim scorer)."""
    texts = real + sim
    y = np.array([1] * len(real) + [0] * len(sim))
    V = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    X = V.fit_transform(texts)
    auc = float(cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc").mean())
    clf = LogisticRegression(max_iter=2000).fit(X, y)

    def p_sim(utts):
        return clf.predict_proba(V.transform(utts))[:, 0]     # P(class 0 = sim)
    return auc, p_sim


def reflect_n(judge, style, flagged, n):
    ex = " / ".join(flagged[:6])
    base = ("You tune a STYLE INSTRUCTION for a simulated conversational-search user so its messages are "
            f"INDISTINGUISHABLE from REAL users to a detector.\nCurrent instruction:\n{style}\n\n"
            f"A detector still flags these simulated messages as fake: {ex}\n\n"
            "Rewrite the instruction (1-3 sentences) so the messages read like real users and evade the "
            "detector (natural phrasing, realistic length variation, not robotically uniform). Output ONLY the instruction.")
    out = []
    for _ in range(n):
        c = judge.complete(base, temperature=0.8).text.strip()
        if 10 < len(c) < 500 and c not in out:
            out.append(c)
    return out or [style]


def main(n_cal=60, n_hold=60, K=3, N=4):
    env = load_env()
    gen = build_generator(env)
    judge = build_judge(env)
    turns = load_turns()
    random.Random(13).shuffle(turns)
    cal, hold = turns[:n_cal], turns[n_cal:n_cal + n_hold]
    real_cal = [r["real"] for r in cal]
    real_hold = [r["real"] for r in hold]

    def hold_auc(style):
        return disc(real_hold, gen_arm(gen, style, hold))[0]

    style = DEFAULT_STYLE
    traj = [round(hold_auc(style), 3)]
    print(f"cal={len(cal)} hold={len(hold)} K={K} N={N} | held-out discriminator AUC (lower=more human)", flush=True)
    print(f"[k=0 default] hold_auc={traj[-1]}", flush=True)
    for k in range(K):
        sim_cal = gen_arm(gen, style, cal)
        cur, p_sim = disc(real_cal, sim_cal)
        flagged = [q for q, _ in sorted(zip(sim_cal, p_sim(sim_cal)), key=lambda t: -t[1])][:6]
        cands = reflect_n(judge, style, flagged, N)
        scored = [(disc(real_cal, gen_arm(gen, c, cal))[0], c) for c in cands]
        best_auc, best = min(scored, key=lambda t: t[0])
        if best_auc <= cur:
            style = best
        traj.append(round(hold_auc(style), 3))
        print(f"[k={k + 1}] cur_cal_auc={cur:.3f} best_cal_auc={best_auc:.3f} "
              f"{'ACCEPT' if best_auc <= cur else 'reject'} hold_auc={traj[-1]}", flush=True)
    print(f"\nheld-out discriminator AUC over cycles: {traj}", flush=True)
    print(f"final style instruction:\n{style}", flush=True)
    (DATA / "e1_disc_loop.json").write_text(json.dumps({"hold_auc_traj": traj, "final_style": style}, indent=1))
    print("saved -> data/ikat/2023/e1_disc_loop.json", flush=True)


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:5]]
    main(*a) if a else main()
