"""E1-fair: score the E1 arms against a NON-hand-writable objective (a learned real-vs-sim
discriminator), the metric the SOTA loops actually optimize (DIAL) -- not the 3-stat CSG-D1
a human can copy into a prompt.

For each style arm we generate simulated iKAT user utterances, embed real vs sim (gateway
jina-embeddings-v3), and train a logistic real-vs-sim classifier (5-fold AUC). AUC->0.5 means
the sim is indistinguishable from real (the loop truly closed the gap); AUC->1.0 means trivially
separable. The question the sharp reviewer / the user raised: on THIS objective, does a terse
one-liner still match the loop, or does the hand-writable win vanish?

Run: .venv/bin/python scripts/e1_discriminator.py
"""
import json
import os
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from autosim.runtime import load_env, build_generator

YEAR = os.environ.get("AUTOSIM_YEAR", "2023")
DATA = Path(f"data/ikat/{YEAR}")
EMB_MODEL = "octen-embedding-8b"
NTURNS = 120
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


def embed(texts, env):
    ep = env["CUSTOM_LLM_ENDPOINT"].rstrip("/"); key = env["CUSTOM_LLM_API_KEY"]
    out = []
    for i in range(0, len(texts), 48):
        body = json.dumps({"model": EMB_MODEL, "input": texts[i:i + 48]}).encode()
        req = urllib.request.Request(ep + "/v1/embeddings", body,
                                     {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        for attempt in range(4):
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=90).read()); break
            except Exception:
                if attempt == 3:
                    raise
        out += [d["embedding"] for d in r["data"]]
    return np.array(out, dtype=float)


def auc(real_emb, sim_emb):
    X = np.vstack([real_emb, sim_emb]); y = np.array([1] * len(real_emb) + [0] * len(sim_emb))
    return float(cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc").mean())


def main():
    env = load_env()
    gen = build_generator(env)
    turns = load_turns()
    random.Random(13).shuffle(turns)
    turns = turns[:NTURNS]
    real = [r["real"] for r in turns]
    arms = {
        "default": "Write naturally as this user would.",
        "naive_terse": "Write a short, terse search message, about 12 words, and skip pleasantries.",
        "oracle_stats": "Average about 13 words per message. About 2% of the time include polite phrasing. "
                        "About 8% of the time use a very short phrase of 3 words or fewer.",
        "loop_final": LOOP_FINAL,
    }
    print(f"real n={len(real)}; discriminator = jina-v3 embed + logistic 5-fold AUC (0.5=indistinguishable)", flush=True)
    real_emb = embed(real, env)
    results = {}
    saved = {"real": real}
    for name, style in arms.items():
        sim = gen_arm(gen, style, turns)
        a = auc(real_emb, embed(sim, env))
        results[name] = round(a, 3); saved[name] = sim
        print(f"  {name:14s} discriminator AUC(real vs sim) = {a:.3f}", flush=True)
    (DATA / "e1_discriminator.json").write_text(json.dumps({"auc": results, "n": len(real)}, indent=1))
    (DATA / "e1_discriminator_utts.json").write_text(json.dumps(saved, ensure_ascii=False))
    lo = min(results.values())
    print(f"\nlowest AUC (most real) = {lo}; all arms still >~0.8 => prompt-level cannot fool the "
          f"discriminator (points to weight-level); a spread => the loop/one-liner differ on the rich metric", flush=True)
    print("saved -> data/ikat/2023/e1_discriminator.json", flush=True)


if __name__ == "__main__":
    main()
