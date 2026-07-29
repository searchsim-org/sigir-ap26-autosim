"""Experiment A pilot: how far is default-AutoSim from REAL Qilin user queries?
Trains a sim-vs-real discriminator (multilingual embeddings + logistic, 5-fold AUC)
and reports length stats. AUC near 1.0 = the behavioral gap the discriminator exploits
(a richer, harder-to-game signal than v1's 3-feature CSG-D1). Runs on a GPU node.

Usage (on a GPU node, msmarco env): python expA_pilot.py [N]
"""
import glob
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

HOME = os.path.expanduser("~")
QILIN = f"{HOME}/v2sim/data/qilin/search_train"
GEN_MODEL = "qwen3-next-80b-a3b-instruct"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500


def load_env(path=f"{HOME}/v2sim/.env"):
    env = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip().strip('"').strip("'")
    return env


def real_queries(n):
    df = pd.read_parquet(sorted(glob.glob(QILIN + "/*.parquet"))[0])
    qs = [str(q) for q in df["query"].dropna().tolist()]
    qs = [q for q in qs if 1 <= len(q) <= 60 and q != "nan"]
    random.seed(0)
    return random.sample(qs, min(n, len(qs)))


def gen_default(client, need):
    ask = ("You are a real user of a Chinese lifestyle/social search app (like Xiaohongshu). "
           f"You want to find information about: {need}. "
           "Write ONLY the exact search query you would type, in Chinese, as you naturally would. "
           "No quotes, no explanation.")
    try:
        r = client.chat.completions.create(
            model=GEN_MODEL, messages=[{"role": "user", "content": ask}], temperature=0.7)
        return (r.choices[0].message.content or "").strip().strip('"').splitlines()[0][:60]
    except Exception:
        return ""


def main():
    env = load_env()
    client = OpenAI(base_url=env["CUSTOM_LLM_ENDPOINT"].rstrip("/") + "/v1",
                    api_key=env["CUSTOM_LLM_API_KEY"])
    real = real_queries(N)
    print(f"real queries: {len(real)}; sample: {real[:3]}", flush=True)

    with ThreadPoolExecutor(max_workers=16) as ex:
        sim = list(ex.map(lambda q: gen_default(client, q), real))
    sim = [s for s in sim if s]
    print(f"sim (default AutoSim): {len(sim)}; sample: {sim[:3]}", flush=True)

    rl = np.array([len(q) for q in real]); sl = np.array([len(q) for q in sim])
    print(f"char-len  real {rl.mean():.1f}+/-{rl.std():.1f}   sim {sl.mean():.1f}+/-{sl.std():.1f}", flush=True)

    emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
    X = emb.encode(real + sim, convert_to_numpy=True, batch_size=128, show_progress_bar=False)
    y = np.array([1] * len(real) + [0] * len(sim))
    auc = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc").mean()
    print(f"\nDISCRIMINATOR 5-fold AUC (real vs default-AutoSim) = {auc:.3f}", flush=True)
    print("(AUC->1.0 = trivially separable = big behavioral gap; the reward to close)", flush=True)

    os.makedirs(f"{HOME}/v2sim/runs", exist_ok=True)
    json.dump({"real": real, "sim_default": sim, "auc": auc,
               "len_real": [rl.mean(), rl.std()], "len_sim": [sl.mean(), sl.std()]},
              open(f"{HOME}/v2sim/runs/expA_pilot.json", "w"), ensure_ascii=False, indent=1)
    print("saved -> ~/v2sim/runs/expA_pilot.json", flush=True)


if __name__ == "__main__":
    main()
