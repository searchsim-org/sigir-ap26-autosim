"""Experiment A / Stage 2 (robust rewrite): adversarially evolve AutoSim's generation
style to fool a sim-vs-real discriminator, in TWO arms, with proper variance + a
coverage objective that actually diverges from naive.

Two fixes over expA_adversarial.py, forced by what the first sweep revealed:
  (1) The gateway is DETERMINISTIC per prompt (verified). So re-running with different
      RNG seeds gives byte-identical trajectories -> zero variance. Real error bars come
      from resampling the QUERY SUBSET. Each seed takes a DISJOINT chunk of the need pool,
      giving independent replicates (sampling variance over the query population).
  (2) The old coverage term used embed_dispersion, which barely moves across candidate
      styles (0.80-0.84), so coverage never diverged from naive. The new coverage penalty
      measures SHORTFALL BELOW REAL diversity (distinct-2 + length entropy, normalized) --
      quantities that DO vary and that collapse kills. Coverage = fool the discriminator
      WITHOUT letting diversity fall below the real users'. This is the concrete
      non-Goodhartable reward: tied to the real behavior distribution, not to a scalar mean.

Per round we track: discriminator held-out AUC (down = sim looks more real), and diversity
(distinct-2, length entropy, embedding dispersion). Expectation (objective-inversion thesis,
shown not claimed): NAIVE collapses diversity; COVERAGE holds it near real -- across seeds.

Usage: python expA_adversarial2.py [K] [subset_size] [suffix] [seed]
Reuses needs from expA_stage1.json. Writes expA_adv2{suffix}.json (full styles + final sim).
"""
import json
import os
import random
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
GN = int(sys.argv[2]) if len(sys.argv) > 2 else 250          # subset size per seed
SUFFIX = sys.argv[3] if len(sys.argv) > 3 else ""
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
N_CAND, CAND_BATCH, LAMBDA = 3, 100, 1.0
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
    return chat(client, GEN, p, 0.7).splitlines()[0].strip().strip('"')[:60]


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


def collapse_penalty(cd, rp):
    """Normalized shortfall of candidate diversity BELOW the real users' diversity."""
    d2 = max(0.0, rp["distinct2"] - cd["distinct2"]) / max(rp["distinct2"], 1e-6)
    le = max(0.0, rp["len_entropy"] - cd["len_entropy"]) / max(rp["len_entropy"], 1e-6)
    return 0.5 * d2 + 0.5 * le


def fit_disc(real, sim):
    Xr = EMB.encode(real, convert_to_numpy=True, show_progress_bar=False)
    Xs = EMB.encode(sim, convert_to_numpy=True, show_progress_bar=False)
    X = np.vstack([Xr, Xs]); y = np.array([1] * len(real) + [0] * len(sim))
    auc = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc").mean()
    clf = LogisticRegression(max_iter=2000).fit(X, y)

    def p_sim(qs):
        return clf.predict_proba(EMB.encode(qs, convert_to_numpy=True, show_progress_bar=False))[:, 0]
    return float(auc), p_sim


ANGLES = ["Make them much shorter and more fragmentary, like a real quick search.",
          "Use colloquial slang, abbreviations and occasional typos.",
          "Drop punctuation and function words; keep only content words.",
          "Mimic fast mobile typing with pinyin shortcuts and no spaces.",
          "Use the topic-specific jargon and product names real users type."]


def propose_styles(client, style, flagged, n=N_CAND):
    """Distinct prompt ANGLE per candidate so the (deterministic) gateway returns DIFFERENT
    styles -- otherwise identical prompts yield identical candidates and the search collapses
    to a single forced path (no real choice, arms cannot diverge)."""
    ex = " / ".join(flagged[:6])
    out = []
    for i in range(n):
        base = (f"You tune a STYLE INSTRUCTION for a simulated Chinese search user so its queries look like REAL users. "
                f"Current instruction:\n{style}\n\nA detector still flags these simulated queries as fake: {ex}\n\n"
                f"{ANGLES[i % len(ANGLES)]} "
                "Write an improved 1-2 sentence style instruction so the simulated queries are indistinguishable from "
                "real terse, messy user queries. Output ONLY the instruction.")
        c = chat(client, JUDGE, base, 0.9)
        if 8 < len(c) < 400 and c not in out:
            out.append(c)
    return out or [style]


def run_arm(client, arm, needs_sub, real_sub, real_prof):
    style, traj = DEFAULT_STYLE, []
    sim = []
    for k in range(K):
        sim = gen_batch(client, needs_sub, style)
        auc, p_sim = fit_disc(real_sub, sim)
        dv = diversity(sim)
        traj.append({"round": k, "auc": round(auc, 3), **dv, "style": style})
        print(f"  [seed{SEED} {arm} k={k}] AUC={auc:.3f} distinct2={dv['distinct2']} "
              f"len_ent={dv['len_entropy']} disp={dv['embed_dispersion']}", flush=True)
        if k == K - 1:
            break
        flagged = [q for q, ps in sorted(zip(sim, p_sim(sim)), key=lambda t: -t[1])][:6]
        cands = propose_styles(client, style, flagged)
        scored = []
        for c in cands:
            b = gen_batch(client, needs_sub[:CAND_BATCH], c, workers=16)
            if not b:
                continue
            reward = 1 - float(p_sim(b).mean())               # higher = fools D better
            pen = collapse_penalty(diversity(b), real_prof)   # >0 if diversity fell below real
            obj = reward if arm == "naive" else reward - LAMBDA * pen
            scored.append((obj, c))
        if scored:
            style = max(scored, key=lambda t: t[0])[1]
    return {"traj": traj, "final_style": style, "final_sim": sim[:200]}


def main():
    global EMB
    env = load_env()
    client = OpenAI(base_url=env["CUSTOM_LLM_ENDPOINT"].rstrip("/") + "/v1", api_key=env["CUSTOM_LLM_API_KEY"])
    st1 = json.load(open(f"{RUNS}/expA_stage1.json"))
    real_all, needs_all = st1["real"], st1["needs"]
    # DISJOINT subset per seed: fixed shuffle, then chunk [SEED*GN : (SEED+1)*GN]
    order = list(range(len(real_all))); random.Random(0).shuffle(order)
    idx = order[SEED * GN:(SEED + 1) * GN]
    real_sub = [real_all[i] for i in idx]; needs_sub = [needs_all[i] for i in idx]
    EMB = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
    real_prof = diversity(real_sub)
    print(f"seed {SEED}: {len(real_sub)} real/needs (disjoint chunk); real_prof={real_prof}", flush=True)
    out = {"seed": SEED, "real_prof": real_prof, "idx_range": [SEED * GN, (SEED + 1) * GN]}
    for arm in ["naive", "coverage"]:
        print(f"\n=== seed{SEED} ARM: {arm} ===", flush=True)
        out[arm] = run_arm(client, arm, needs_sub, real_sub, real_prof)
    json.dump(out, open(f"{RUNS}/expA_adv2{SUFFIX}.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved -> ~/v2sim/runs/expA_adv2{SUFFIX}.json", flush=True)
    for arm in ["naive", "coverage"]:
        t = out[arm]["traj"]
        print(f"{arm:9s}: AUC {t[0]['auc']}->{t[-1]['auc']}  "
              f"distinct2 {t[0]['distinct2']}->{t[-1]['distinct2']}  "
              f"len_ent {t[0]['len_entropy']}->{t[-1]['len_entropy']}", flush=True)


if __name__ == "__main__":
    main()
