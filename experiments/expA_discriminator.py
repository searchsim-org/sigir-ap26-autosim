"""Experiment A / Stage 1: a TRAINED sim-vs-real discriminator on real Qilin queries,
plus baseline diversity metrics (the quantities that mode-collapse will later kill).

Pipeline (on a GPU node, msmarco env, A100):
  1. sample N real Qilin queries
  2. abstract each into a neutral information-need (judge=gemma) so generation is
     topic-matched but style-controllable (not an echo of the real query)
  3. generate default-style AutoSim queries (gen=qwen) from those needs
  4. train an xlm-roberta discriminator real-vs-sim -> held-out AUC/acc (the gap)
  5. diversity of real vs default-sim: distinct-1/2 (char), length entropy, embedding dispersion
  6. save needs + queries for the adversarial stage.

Usage: python expA_discriminator.py [N]
"""
import glob
import json
import math
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HOME = os.path.expanduser("~")
QILIN = f"{HOME}/v2sim/data/qilin/search_train"
RUNS = f"{HOME}/v2sim/runs"
GEN, JUDGE = "qwen3-next-80b-a3b-instruct", "gemma4-31b-it"
DISC = "xlm-roberta-base"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000


def load_env(path=f"{HOME}/v2sim/.env"):
    e = {}
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); e[k] = v.strip().strip('"').strip("'")
    return e


def real_queries(n):
    df = pd.read_parquet(sorted(glob.glob(QILIN + "/*.parquet"))[0])
    qs = [str(q) for q in df["query"].dropna().tolist() if 1 <= len(str(q)) <= 60 and str(q) != "nan"]
    random.seed(0); return random.sample(qs, min(n, len(qs)))


def chat(client, model, prompt, temp):
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=temp)
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""


def abstract_need(client, q):
    p = (f"A user searched a Chinese lifestyle app with this query: 「{q}」. "
         "In ONE short neutral sentence (Chinese), describe what information they want. "
         "Output ONLY the description.")
    return chat(client, JUDGE, p, 0.0).splitlines()[0][:80] or q


def gen_query(client, need, style):
    p = ("You are a real user of a Chinese lifestyle/social search app (like Xiaohongshu). "
         f"You want to find information about: {need}. {style} "
         "Write ONLY the exact search query you would type, in Chinese. No quotes, no explanation.")
    return chat(client, GEN, p, 0.7).splitlines()[0].strip().strip('"')[:60]


def diversity(qs, emb):
    chars = "".join(qs)
    d1 = len(set(chars)) / max(1, len(chars))
    bi = [chars[i:i + 2] for i in range(len(chars) - 1)]
    d2 = len(set(bi)) / max(1, len(bi))
    L = np.array([len(q) for q in qs])
    p = np.array(list(Counter(L).values())) / len(L)
    ent = float(-(p * np.log(p)).sum())
    V = emb.encode(qs, convert_to_numpy=True, batch_size=128, show_progress_bar=False)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    disp = float(1 - (Vn @ Vn.mean(0)).mean())  # mean cosine distance to centroid
    return {"distinct1": round(d1, 4), "distinct2": round(d2, 4),
            "len_mean": round(L.mean(), 2), "len_entropy": round(ent, 3), "embed_dispersion": round(disp, 4)}


def train_discriminator(real, sim):
    tok = AutoTokenizer.from_pretrained(DISC)
    model = AutoModelForSequenceClassification.from_pretrained(DISC, num_labels=2).cuda()
    texts = real + sim
    labels = [1] * len(real) + [0] * len(sim)
    idx = list(range(len(texts))); random.Random(1).shuffle(idx)
    cut = int(0.8 * len(idx)); tr, te = idx[:cut], idx[cut:]

    def batchify(ids, bs=32):
        for i in range(0, len(ids), bs):
            b = ids[i:i + bs]
            enc = tok([texts[j] for j in b], padding=True, truncation=True, max_length=32, return_tensors="pt")
            yield {k: v.cuda() for k, v in enc.items()}, torch.tensor([labels[j] for j in b]).cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for ep in range(3):
        random.Random(ep).shuffle(tr)
        for enc, y in batchify(tr):
            opt.zero_grad(); out = model(**enc, labels=y); out.loss.backward(); opt.step()
    model.eval(); probs, ys = [], []
    with torch.no_grad():
        for enc, y in batchify(te):
            p = torch.softmax(model(**enc).logits, -1)[:, 1]
            probs += p.tolist(); ys += y.tolist()
    auc = roc_auc_score(ys, probs)
    acc = np.mean([(p > 0.5) == bool(t) for p, t in zip(probs, ys)])
    return float(auc), float(acc)


def main():
    env = load_env()
    client = OpenAI(base_url=env["CUSTOM_LLM_ENDPOINT"].rstrip("/") + "/v1", api_key=env["CUSTOM_LLM_API_KEY"])
    real = real_queries(N)
    print(f"real: {len(real)}", flush=True)
    with ThreadPoolExecutor(max_workers=24) as ex:
        needs = list(ex.map(lambda q: abstract_need(client, q), real))
        sim = list(ex.map(lambda nd: gen_query(client, nd, "Write naturally as this user would."), needs))
    keep = [(r, n, s) for r, n, s in zip(real, needs, sim) if s]
    real, needs, sim = [k[0] for k in keep], [k[1] for k in keep], [k[2] for k in keep]
    print(f"generated default-sim: {len(sim)}; sample real/sim: {real[0]} || {sim[0]}", flush=True)

    emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
    dv_real, dv_sim = diversity(real, emb), diversity(sim, emb)
    auc, acc = train_discriminator(real, sim)
    print(f"\n== TRAINED discriminator (xlm-roberta) real vs default-AutoSim ==", flush=True)
    print(f"   held-out AUC={auc:.3f}  acc={acc:.3f}", flush=True)
    print(f"== diversity ==\n   real: {dv_real}\n   sim : {dv_sim}", flush=True)

    os.makedirs(RUNS, exist_ok=True)
    json.dump({"real": real, "needs": needs, "sim_default": sim,
               "discriminator": {"auc": auc, "acc": acc},
               "diversity": {"real": dv_real, "sim_default": dv_sim}},
              open(f"{RUNS}/expA_stage1.json", "w"), ensure_ascii=False, indent=1)
    print("saved -> ~/v2sim/runs/expA_stage1.json", flush=True)


if __name__ == "__main__":
    main()
