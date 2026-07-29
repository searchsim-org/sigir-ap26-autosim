"""Experiment B: the realism->validity link on real Qilin behavior.

Claim: a user simulator whose QUERY DISTRIBUTION collapses cannot rank retrieval
systems the way real users do. Combined with Exp A (naive self-improvement CAUSES
collapse), this shows naive self-improvement DEGRADES the evaluator while a
coverage-preserving objective keeps it valid.

Protocol (real behavior, no assessor needed):
  - M real Qilin queries, each with its real SERP clicks (= relevance).
  - Corpus = SERP notes + random notes; 6 retrieval systems with a quality spread.
  - Ground-truth system ranking = nDCG@10 of each system on the REAL queries (real clicks).
  - Drive the evaluation with query distributions of decreasing diversity:
      real (upper bound) -> default-AutoSim -> collapsed to K in {50,10,3} modes.
    Each generated/collapsed query inherits its source real query's clicked notes as
    relevance. Compute the induced system ranking and its Kendall tau vs the real ranking.
  - Expectation: tau degrades monotonically as the query distribution collapses.

Runs on a GPU node (msmarco env, A100). Usage: python exp_B.py [M] [n_random_notes]
"""
import glob
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from openai import OpenAI
from rank_bm25 import BM25Okapi
from scipy.stats import kendalltau
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

HOME = os.path.expanduser("~")
QROOT = f"{HOME}/v2sim/data/qilin"
GEN, JUDGE = "qwen3-next-80b-a3b-instruct", "gemma4-31b-it"
M = int(sys.argv[1]) if len(sys.argv) > 1 else 600
NRAND = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
random.seed(0); np.random.seed(0)
EMB = None


def load_env(path=f"{HOME}/v2sim/.env"):
    e = {}
    for ln in open(path):
        ln = ln.strip()
        if ln and "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1); e[k] = v.strip().strip('"').strip("'")
    return e


def chat(client, model, prompt, temp):
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=temp)
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""


def toks(s):  # char bigrams for Chinese BM25
    s = str(s); return [s[i:i + 2] for i in range(len(s) - 1)] or [s]


def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg(ranked_ids, rel, k=10):
    g = [rel.get(i, 0) for i in ranked_ids[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idc = dcg(ideal)
    return dcg(g) / idc if idc > 0 else 0.0


def main():
    env = load_env()
    client = OpenAI(base_url=env["CUSTOM_LLM_ENDPOINT"].rstrip("/") + "/v1", api_key=env["CUSTOM_LLM_API_KEY"])
    # --- load queries + SERP relevance ---
    s = pd.read_parquet(sorted(glob.glob(f"{QROOT}/search_train/*.parquet"))[0])
    s = s[s["query"].notna()].reset_index(drop=True)
    idx = random.sample(range(len(s)), M)
    queries, rels, serp_notes = [], [], set()
    for i in idx:
        row = s.iloc[i]; det = row.search_result_details_with_idx
        rel = {}
        for d in det:
            nid = int(d["note_idx"]); serp_notes.add(nid)
            score = (2 if (d.get("click") and (d.get("collect") or d.get("like") or (d.get("page_time", 0) or 0) > 10)) else (1 if d.get("click") else 0))
            if score: rel[nid] = max(rel.get(nid, 0), score)
        if rel:
            queries.append(str(row["query"])); rels.append(rel)
    print(f"queries with clicks: {len(queries)}", flush=True)

    # --- corpus: SERP notes + random notes ---
    notes = pd.concat([pd.read_parquet(f)[["note_idx", "note_title", "note_content"]]
                       for f in sorted(glob.glob(f"{QROOT}/notes/*.parquet"))], ignore_index=True)
    notes["note_idx"] = notes["note_idx"].astype(int)
    rand_ids = set(np.random.choice(notes["note_idx"].values, size=min(NRAND, len(notes)), replace=False).tolist())
    corpus_ids = sorted(serp_notes | rand_ids)
    nmap = notes.set_index("note_idx")
    text = {nid: (str(nmap.loc[nid, "note_title"]) + " " + str(nmap.loc[nid, "note_content"]))[:256]
            for nid in corpus_ids if nid in nmap.index}
    corpus_ids = [c for c in corpus_ids if c in text]
    docs = [text[c] for c in corpus_ids]
    print(f"corpus: {len(corpus_ids)} notes", flush=True)

    # --- indexes ---
    bm25 = BM25Okapi([toks(d) for d in docs])
    dmat = EMB.encode(docs, convert_to_numpy=True, batch_size=256, show_progress_bar=False, normalize_embeddings=True)

    def retrieve(q, system, k=10):
        if system == "random":
            return random.sample(corpus_ids, k)
        qq = q[:2] if "trunc" in system else q
        if system.startswith("bm25"):
            sc = bm25.get_scores(toks(qq)); top = np.argsort(-sc)[:k]
            return [corpus_ids[j] for j in top]
        if system.startswith("dense"):
            qv = EMB.encode([qq], convert_to_numpy=True, normalize_embeddings=True)[0]
            sc = dmat @ qv; top = np.argsort(-sc)[:k]
            return [corpus_ids[j] for j in top]
        if system == "hybrid":  # RRF of bm25 + dense
            b = np.argsort(-bm25.get_scores(toks(q)))[:50]
            qv = EMB.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
            d = np.argsort(-(dmat @ qv))[:50]
            rr = {}
            for r, j in enumerate(b): rr[j] = rr.get(j, 0) + 1 / (60 + r)
            for r, j in enumerate(d): rr[j] = rr.get(j, 0) + 1 / (60 + r)
            top = sorted(rr, key=lambda j: -rr[j])[:k]
            return [corpus_ids[j] for j in top]
    SYSTEMS = ["dense", "hybrid", "bm25", "bm25_trunc", "dense_trunc", "random"]

    def system_ranking(qlist):
        sc = {}
        for sysn in SYSTEMS:
            vals = [ndcg(retrieve(q, sysn), rel) for q, rel in zip(qlist, rels)]
            sc[sysn] = float(np.mean(vals))
        return sc

    real_scores = system_ranking(queries)
    real_order = [real_scores[x] for x in SYSTEMS]
    print("real system nDCG:", {k: round(v, 4) for k, v in real_scores.items()}, flush=True)

    # --- default-AutoSim queries (from abstracted intents) ---
    def abstract(q):
        return chat(client, JUDGE, f"A user searched a Chinese lifestyle app with: 「{q}」. In ONE short neutral Chinese sentence, say what they want. Output ONLY it.", 0.0).splitlines()[0][:80] or q

    def gen(nd):
        return chat(client, GEN, f"You are a real Chinese lifestyle-app user. You want: {nd}. Write ONLY the exact search query you'd type, in Chinese.", 0.7).splitlines()[0].strip().strip('"')[:60]
    with ThreadPoolExecutor(max_workers=24) as ex:
        needs = list(ex.map(abstract, queries))
        simq = list(ex.map(gen, needs))
    simq = [x if x else queries[i] for i, x in enumerate(simq)]

    # --- collapse operator: map each sim query to its k-means cluster medoid ---
    def collapse(qs, K):
        V = EMB.encode(qs, convert_to_numpy=True, normalize_embeddings=True)
        km = KMeans(n_clusters=min(K, len(set(qs))), n_init=3, random_state=0).fit(V)
        out = []
        for lab in km.labels_:
            members = [i for i in range(len(qs)) if km.labels_[i] == lab]
            c = km.cluster_centers_[lab]
            medoid = members[int(np.argmax(V[members] @ c))]
            out.append(qs[medoid])
        return out

    dists = {"real": queries, "default_sim": simq,
             "collapse_K50": collapse(simq, 50), "collapse_K10": collapse(simq, 10), "collapse_K3": collapse(simq, 3)}
    results = {}
    for name, ql in dists.items():
        sc = system_ranking(ql); order = [sc[x] for x in SYSTEMS]
        tau = kendalltau(real_order, order)[0]
        distinct = len(set(ql)) / len(ql)
        results[name] = {"tau_vs_real": round(float(tau), 3), "distinct_q": round(distinct, 3),
                         "sys_ndcg": {k: round(v, 4) for k, v in sc.items()}}
        print(f"  {name:14s} distinct={distinct:.3f}  tau(vs real system ranking)={tau:+.3f}", flush=True)

    os.makedirs(f"{HOME}/v2sim/runs", exist_ok=True)
    json.dump({"systems": SYSTEMS, "real_scores": real_scores, "results": results, "M": len(queries)},
              open(f"{HOME}/v2sim/runs/exp_B.json", "w"), ensure_ascii=False, indent=1)
    print("\nsaved -> ~/v2sim/runs/exp_B.json", flush=True)


if __name__ == "__main__":
    EMB = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
    main()
