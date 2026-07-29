"""Experiment A / Stage 2 (the demonstration): adversarially evolve AutoSim's generation
style to fool a sim-vs-real discriminator, in TWO arms:
  - NAIVE   : pick the style edit that best fools the discriminator (maximize the proxy reward)
  - COVERAGE: pick the edit that fools it while preserving diversity (reward - lambda*div_drop)
Track, per round: discriminator held-out AUC (down = sim looks more real) AND diversity
(distinct-2, length entropy, embedding dispersion). Expectation (the objective-inversion
thesis, shown not claimed): NAIVE lowers AUC but COLLAPSES diversity (tails die); COVERAGE
lowers AUC while holding diversity.

Reuses the needs generated in Stage 1. Usage: python expA_adversarial.py [K] [rounds_gen_N]
"""
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

HOME = os.path.expanduser("~")
RUNS = f"{HOME}/v2sim/runs"
GEN, JUDGE = "qwen3-next-80b-a3b-instruct", "gemma4-31b-it"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
GN = int(sys.argv[2]) if len(sys.argv) > 2 else 300
N_CAND, CAND_BATCH, LAMBDA = 4, 60, 1.0
DEFAULT_STYLE = "Write naturally as this user would."
EMB = None


def load_env(path=f"{HOME}/v2sim/.env"):
    e = {}
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); e[k] = v.strip().strip('"').strip("'")
    return e


def chat(client, model, prompt, temp):
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=temp)
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""


def gen_query(client, need, style):
    p = ("You are a real user of a Chinese lifestyle/social search app (like Xiaohongshu). "
         f"You want to find information about: {need}. {style} "
         "Write ONLY the exact search query you would type, in Chinese. No quotes, no explanation.")
    return chat(client, GEN, p, 0.7).splitlines()[0].strip().strip('"')[:60] if True else ""


def gen_batch(client, needs, style, workers=24):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(lambda nd: gen_query(client, nd, style), needs))
    return [q for q in out if q]


def diversity(qs):
    chars = "".join(qs)
    bi = [chars[i:i + 2] for i in range(len(chars) - 1)]
    d2 = len(set(bi)) / max(1, len(bi))
    L = np.array([len(q) for q in qs]); p = np.array(list(Counter(L).values())) / len(L)
    ent = float(-(p * np.log(p)).sum())
    V = EMB.encode(qs, convert_to_numpy=True, batch_size=128, show_progress_bar=False)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    disp = float(1 - (Vn @ Vn.mean(0)).mean())
    return {"distinct2": round(d2, 4), "len_entropy": round(ent, 3), "embed_dispersion": round(disp, 4)}


def fit_disc(real, sim):
    """Return (held-out AUC, scorer q->P(sim)) using MiniLM embeddings + logistic."""
    Xr, Xs = EMB.encode(real, convert_to_numpy=True, show_progress_bar=False), EMB.encode(sim, convert_to_numpy=True, show_progress_bar=False)
    X = np.vstack([Xr, Xs]); y = np.array([1] * len(real) + [0] * len(sim))  # 1=real,0=sim
    auc = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc").mean()
    clf = LogisticRegression(max_iter=2000).fit(X, y)

    def p_sim(qs):
        V = EMB.encode(qs, convert_to_numpy=True, show_progress_bar=False)
        return clf.predict_proba(V)[:, 0]  # P(class 0 = sim)
    return float(auc), p_sim


def propose_styles(client, style, flagged, n=N_CAND):
    ex = " / ".join(flagged[:6])
    base = (f"You tune a STYLE INSTRUCTION for a simulated Chinese search user so its queries look like REAL users. "
            f"Current instruction:\n{style}\n\nA detector still flags these simulated queries as fake: {ex}\n\n"
            "Write an improved 1-2 sentence style instruction so the simulated queries are indistinguishable from real "
            "terse, messy user queries. Output ONLY the instruction.")
    out = []
    for _ in range(n):
        c = chat(client, JUDGE, base, 0.9)
        if 8 < len(c) < 400 and c not in out:
            out.append(c)
    return out or [style]


def run_arm(client, arm, needs, real):
    gen_needs = needs[:GN]
    style, traj = DEFAULT_STYLE, []
    for k in range(K):
        sim = gen_batch(client, gen_needs, style)
        auc, p_sim = fit_disc(real, sim)
        dv = diversity(sim)
        traj.append({"round": k, "auc": round(auc, 3), **dv, "style": style[:80]})
        print(f"  [{arm} k={k}] AUC={auc:.3f} distinct2={dv['distinct2']} "
              f"len_ent={dv['len_entropy']} disp={dv['embed_dispersion']}", flush=True)
        if k == K - 1:
            break
        flagged = [q for q, ps in sorted(zip(sim, p_sim(sim)), key=lambda t: -t[1])][:6]
        cands = propose_styles(client, style, flagged)
        scored = []
        for c in cands:
            b = gen_batch(client, gen_needs[:CAND_BATCH], c, workers=16)
            if not b:
                continue
            reward = 1 - float(p_sim(b).mean())        # higher = fools D better
            div = diversity(b)["embed_dispersion"]
            obj = reward if arm == "naive" else reward + LAMBDA * div  # coverage keeps dispersion
            scored.append((obj, c))
        if scored:
            style = max(scored, key=lambda t: t[0])[1]
    return traj


def main():
    global EMB
    env = load_env()
    client = OpenAI(base_url=env["CUSTOM_LLM_ENDPOINT"].rstrip("/") + "/v1", api_key=env["CUSTOM_LLM_API_KEY"])
    st1 = json.load(open(f"{RUNS}/expA_stage1.json"))
    real, needs = st1["real"], st1["needs"]
    EMB = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
    print(f"loaded {len(real)} real, {len(needs)} needs", flush=True)
    out = {}
    for arm in ["naive", "coverage"]:
        print(f"\n=== ARM: {arm} ===", flush=True)
        out[arm] = run_arm(client, arm, needs, real)
    suffix = sys.argv[3] if len(sys.argv) > 3 else ""
    json.dump(out, open(f"{RUNS}/expA_adversarial{suffix}.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved -> ~/v2sim/runs/expA_adversarial{suffix}.json", flush=True)
    for arm in out:
        t = out[arm]
        print(f"{arm:9s}: AUC {t[0]['auc']}->{t[-1]['auc']}  "
              f"dispersion {t[0]['embed_dispersion']}->{t[-1]['embed_dispersion']}  "
              f"distinct2 {t[0]['distinct2']}->{t[-1]['distinct2']}", flush=True)


if __name__ == "__main__":
    main()
