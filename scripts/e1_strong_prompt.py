"""E1-strong-v2 (no-invisible-walls: BREAK the 'prompt-level is bounded' ceiling by testing the
STRONG version). The crude greedy loop plateaued at 0.968; SOTA (GEPA) shows prompt-level can beat
weight-level RL. Here we add the untried strong levers:
  - FEW-SHOT conditioning on REAL user utterances (disjoint from the eval set -> no leak)
  - reflective optimization of a style instruction ON TOP of few-shot (N=6, K=4)
  - selection on a real-vs-sim discriminator (the non-hand-writable objective)
and measure how LOW prompt-level can drive held-out discriminator AUC. If it approaches
weight-level territory, "weight-level is the frontier" is falsified.

Arms (held-out real-vs-sim AUC, lexical + octen-embedding; lower = more human):
  default | loop_final(CSG-D1 style) | fewshot(real examples) | fewshot+opt(reflective)
Run: .venv/bin/python scripts/e1_strong_prompt.py [n_cal] [n_hold] [K] [N]
"""
import json
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from autosim.runtime import load_env, build_generator, build_judge

YEAR = os.environ.get("AUTOSIM_YEAR", "2023")
DATA = Path(f"data/ikat/{YEAR}")
EMB_MODEL = "octen-embedding-8b"
DEFAULT_STYLE = "Write naturally as this user would."
LOOP_FINAL = ("Average 14 words per message. Occasionally include polite phrasing (approx. 10% of the "
              "time) and occasionally use very short, terse phrases of 3 words or fewer (approx. 8% of the time).")


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


def gen_utt(gen, style, rec, shots=None):
    ex = ""
    if shots:
        ex = "Here is how REAL users in this study phrase their messages (match this style closely):\n" + \
             "\n".join(f"- {s}" for s in shots) + "\n\n"
    ask = (f"You are a real user in an ongoing conversational web-search session.\n"
           f"Your background:\n- " + "\n- ".join(rec["ptkb"][:8]) + "\n\n"
           f"{ex}Conversation so far:\n{rec['history']}\n\n"
           f"Your next information need (express it as YOUR next message, in your own words): "
           f"{rec['resolved']}\n\n{style}\n"
           f"Write ONLY your next message to the assistant (no quotes, no preamble).")
    return gen.complete(ask, temperature=0.7).text.strip().strip('"').splitlines()[0][:300]


def gen_arm(gen, style, recs, shot_pool=None, kshot=6, workers=12):
    def one(r):
        shots = random.Random(hash(r["resolved"]) & 0xffff).sample(shot_pool, min(kshot, len(shot_pool))) if shot_pool else None
        return gen_utt(gen, style, r, shots)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, recs))


def disc(real, sim):
    texts = real + sim
    y = np.array([1] * len(real) + [0] * len(sim))
    V = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    X = V.fit_transform(texts)
    auc = float(cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc").mean())
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    return auc, (lambda utts: clf.predict_proba(V.transform(utts))[:, 0])


def embed(texts, env):
    ep = env["CUSTOM_LLM_ENDPOINT"].rstrip("/"); key = env["CUSTOM_LLM_API_KEY"]
    out = []
    for i in range(0, len(texts), 48):
        body = json.dumps({"model": EMB_MODEL, "input": texts[i:i + 48]}).encode()
        req = urllib.request.Request(ep + "/v1/embeddings", body,
                                     {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        out += [d["embedding"] for d in json.loads(urllib.request.urlopen(req, timeout=90).read())["data"]]
    return np.array(out, dtype=float)


def emb_auc(real, sim, env):
    Xr, Xs = embed(real, env), embed(sim, env)
    X = np.vstack([Xr, Xs]); y = np.array([1] * len(real) + [0] * len(sim))
    return float(cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc").mean())


def reflect_n(judge, style, flagged, n):
    ex = " / ".join(flagged[:6])
    base = ("You tune a STYLE INSTRUCTION for a simulated conversational-search user so its messages are "
            f"INDISTINGUISHABLE from real users to a detector.\nCurrent instruction:\n{style}\n\n"
            f"A detector still flags these simulated messages as fake: {ex}\n\nName the specific giveaways "
            "(length, phrasing, formality, uniformity) and rewrite the instruction (1-3 sentences) to remove them. "
            "Output ONLY the instruction.")
    out = []
    for _ in range(n):
        c = judge.complete(base, temperature=0.8).text.strip()
        if 10 < len(c) < 500 and c not in out:
            out.append(c)
    return out or [style]


def main(n_cal=60, n_hold=60, K=4, N=6):
    env = load_env()
    gen = build_generator(env); judge = build_judge(env)
    turns = load_turns(); random.Random(13).shuffle(turns)
    cal, hold, shot_recs = turns[:n_cal], turns[n_cal:n_cal + n_hold], turns[n_cal + n_hold:n_cal + n_hold + 40]
    real_cal = [r["real"] for r in cal]; real_hold = [r["real"] for r in hold]
    shot_pool = [r["real"] for r in shot_recs]                      # DISJOINT from hold -> no leak
    print(f"cal={len(cal)} hold={len(hold)} shots_pool={len(shot_pool)} K={K} N={N}", flush=True)

    def evalarm(style, shots):
        sim = gen_arm(gen, style, hold, shot_pool=shots)
        return round(disc(real_hold, sim)[0], 3), sim

    res = {}
    for name, style, shots in [("default", DEFAULT_STYLE, None), ("loop_final", LOOP_FINAL, None),
                               ("fewshot", DEFAULT_STYLE, shot_pool)]:
        a, sim = evalarm(style, shots)
        res[name] = {"lex_auc": a, "sim": sim}
        print(f"  {name:16s} lexical hold_auc={a}", flush=True)

    # fewshot + reflective optimization of the style, selecting on the discriminator (few-shot kept on)
    style = DEFAULT_STYLE
    for k in range(K):
        sim_cal = gen_arm(gen, style, cal, shot_pool=shot_pool)
        cur, p_sim = disc(real_cal, sim_cal)
        flagged = [q for q, _ in sorted(zip(sim_cal, p_sim(sim_cal)), key=lambda t: -t[1])][:6]
        cands = reflect_n(judge, style, flagged, N)
        scored = [(disc(real_cal, gen_arm(gen, c, cal, shot_pool=shot_pool))[0], c) for c in cands]
        best_auc, best = min(scored, key=lambda t: t[0])
        if best_auc <= cur:
            style = best
        print(f"  [fewshot+opt k={k + 1}] cur={cur:.3f} best={best_auc:.3f} {'ACCEPT' if best_auc <= cur else 'reject'}", flush=True)
    a, sim = evalarm(style, shot_pool)
    res["fewshot_opt"] = {"lex_auc": a, "sim": sim, "style": style}
    print(f"  {'fewshot_opt':16s} lexical hold_auc={a}", flush=True)

    # confirm the winners on the independent octen-embedding discriminator
    print("\n-- octen-embedding discriminator (confirm) --", flush=True)
    out = {}
    for name in ["default", "loop_final", "fewshot", "fewshot_opt"]:
        ea = round(emb_auc(real_hold, res[name]["sim"], env), 3)
        out[name] = {"lex_auc": res[name]["lex_auc"], "emb_auc": ea}
        print(f"  {name:16s} lex={res[name]['lex_auc']}  emb={ea}", flush=True)
    (DATA / "e1_strong_prompt.json").write_text(json.dumps({"results": out, "final_style": res["fewshot_opt"].get("style")}, indent=1))
    print("saved -> data/ikat/2023/e1_strong_prompt.json", flush=True)


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:5]]
    main(*a) if a else main()
