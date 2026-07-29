"""Exp C validity, finer-grained (test whether the null tau is a coarse-metric artifact).

Two upgrades over exp_C_validity.py:
  1. 10 retrieval systems (tau resolution 1/45 vs 1/15) with a real quality spread:
     bm25(bigram), bm25(unigram), bm25(title), dense(MiniLM), dense(Qwen3-Emb-0.6B),
     hybrid(bm25+MiniLM), hybrid(bm25+Qwen), bm25_trunc, dense_trunc, random.
  2. PER-QUERY nDCG correlation: for each system, Pearson corr over the 600 queries between
     nDCG(sim query_i) and nDCG(real query_i). A more-realistic simulator should reproduce
     WHICH queries are hard/easy -> higher per-query correlation, even if the system RANKING
     (tau) is saturated. This is the finer realism->validity signal.

Fast BM25 via bm25s. Clean per-arm base reload. Usage: python exp_C_validity2.py [suffix]
"""
import glob
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import bm25s
import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from peft import PeftModel
from scipy.stats import kendalltau, pearsonr
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

HOME = os.path.expanduser("~")
QROOT = f"{HOME}/v2sim/data/qilin"
RUNS = f"{HOME}/v2sim/runs"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
JUDGE = "gemma4-31b-it"
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "_r0"
M = int(sys.argv[2]) if len(sys.argv) > 2 else 600
NRAND = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
SEEDS = int(os.environ.get("EXPC_VSEEDS", 1))
random.seed(0); np.random.seed(0)
DEV = "cuda"


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


def big(s):
    s = str(s); return [s[i:i + 2] for i in range(len(s) - 1)] or [s]


def uni(s):
    return list(str(s)) or [str(s)]


def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg(ranked, rel, k=10):
    idc = dcg(sorted(rel.values(), reverse=True)[:k])
    return dcg([rel.get(i, 0) for i in ranked[:k]]) / idc if idc > 0 else 0.0


def prompt_for(need):
    return ("You are a real user of a Chinese lifestyle/social search app (like Xiaohongshu). "
            f"You want to find information about: {need}. "
            "Write ONLY the exact search query you would type, in Chinese. No quotes, no explanation.")


@torch.no_grad()
def gen_local(model, tok, needs, bs=48):
    tok.padding_side = "left"
    texts = [tok.apply_chat_template([{"role": "user", "content": prompt_for(n)}], tokenize=False, add_generation_prompt=True) for n in needs]
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(DEV)
        g = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=32, pad_token_id=tok.pad_token_id)
        for d in tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
            out.append((d.strip().splitlines()[0].strip().strip('"')[:60]) if d.strip() else "")
    return out


def main():
    env = load_env()
    client = OpenAI(base_url=env["CUSTOM_LLM_ENDPOINT"].rstrip("/") + "/v1", api_key=env["CUSTOM_LLM_API_KEY"])
    s = pd.read_parquet(sorted(glob.glob(f"{QROOT}/search_train/*.parquet"))[0])
    s = s[s["query"].notna()].reset_index(drop=True)
    idx = random.sample(range(len(s)), M)
    queries, rels, serp = [], [], set()
    for i in idx:
        row = s.iloc[i]; rel = {}
        for d in row.search_result_details_with_idx:
            nid = int(d["note_idx"]); serp.add(nid)
            sc = 2 if (d.get("click") and (d.get("collect") or d.get("like") or (d.get("page_time", 0) or 0) > 10)) else (1 if d.get("click") else 0)
            if sc: rel[nid] = max(rel.get(nid, 0), sc)
        if rel:
            queries.append(str(row["query"])); rels.append(rel)
    notes = pd.concat([pd.read_parquet(f)[["note_idx", "note_title", "note_content"]] for f in sorted(glob.glob(f"{QROOT}/notes/*.parquet"))], ignore_index=True)
    notes["note_idx"] = notes["note_idx"].astype(int)
    rand_ids = set(np.random.choice(notes["note_idx"].values, size=min(NRAND, len(notes)), replace=False).tolist())
    cids = sorted(serp | rand_ids)
    nmap = notes.set_index("note_idx")
    title = {c: str(nmap.loc[c, "note_title"]) for c in cids if c in nmap.index}
    full = {c: (str(nmap.loc[c, "note_title"]) + " " + str(nmap.loc[c, "note_content"]))[:256] for c in cids if c in nmap.index}
    cids = [c for c in cids if c in full]
    docs_full = [full[c] for c in cids]; docs_title = [title[c] for c in cids]
    print(f"queries={len(queries)} corpus={len(cids)}", flush=True)

    # indexes
    def bm25_index(tokfn, docs):
        r = bm25s.BM25(); r.index([tokfn(d) for d in docs]); return r
    IDX = {"big": bm25_index(big, docs_full), "uni": bm25_index(uni, docs_full), "title": bm25_index(big, docs_title)}
    mini = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEV)
    Dmini = mini.encode(docs_full, convert_to_numpy=True, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    Dmini_t = mini.encode(docs_title, convert_to_numpy=True, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    print("indexes built", flush=True)

    def bm25_topk(idx, qtok, k):
        res, _ = idx.retrieve(qtok, k=k, show_progress=False)
        return res  # (n,k) doc indices into cids

    def dense_topk(enc_model, D, qs, k):
        Q = enc_model.encode(qs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return np.argsort(-(Q @ D.T), axis=1)[:, :k]

    def rrf(a, b, k):  # a,b: (n,K) doc-idx rankings -> (n,k) fused
        out = []
        for ra, rb in zip(a, b):
            sc = {}
            for r, j in enumerate(ra): sc[j] = sc.get(j, 0) + 1 / (60 + r)
            for r, j in enumerate(rb): sc[j] = sc.get(j, 0) + 1 / (60 + r)
            out.append(sorted(sc, key=lambda j: -sc[j])[:k])
        return out

    SYSTEMS = ["bm25", "bm25_uni", "bm25_title", "dense_mini", "dense_mini_title",
               "hybrid_mini", "hybrid_uni", "bm25_trunc", "dense_trunc", "random"]

    def rankings(qlist, k=10):
        """Return {system: list-of (per-query top-k corpus_id lists)}."""
        big_t = [big(q) for q in qlist]; uni_t = [uni(q) for q in qlist]; tr_t = [big(q[:2]) for q in qlist]
        bm_big = bm25_topk(IDX["big"], big_t, 50)
        bm_uni = bm25_topk(IDX["uni"], uni_t, 50)
        dn_mini = dense_topk(mini, Dmini, qlist, 50)
        dn_mini_t = dense_topk(mini, Dmini_t, qlist, k)
        R = {}
        R["bm25"] = [[cids[j] for j in row[:k]] for row in bm_big]
        R["bm25_uni"] = [[cids[j] for j in row[:k]] for row in bm_uni]
        R["bm25_title"] = [[cids[j] for j in row[:k]] for row in bm25_topk(IDX["title"], big_t, k)]
        R["dense_mini"] = [[cids[j] for j in row[:k]] for row in dn_mini[:, :k]]
        R["dense_mini_title"] = [[cids[j] for j in row] for row in dn_mini_t]
        R["hybrid_mini"] = [[cids[j] for j in row] for row in rrf(bm_big, dn_mini, k)]
        R["hybrid_uni"] = [[cids[j] for j in row] for row in rrf(bm_uni, dn_mini, k)]
        R["bm25_trunc"] = [[cids[j] for j in row[:k]] for row in bm25_topk(IDX["big"], tr_t, k)]
        R["dense_trunc"] = [[cids[j] for j in row[:k]] for row in dense_topk(mini, Dmini, [q[:2] for q in qlist], k)]
        R["random"] = [random.sample(cids, k) for _ in qlist]
        return R

    def per_query_ndcg(R):
        return {sysn: np.array([ndcg(R[sysn][i], rels[i]) for i in range(len(queries))]) for sysn in SYSTEMS}

    real_R = rankings(queries); real_pq = per_query_ndcg(real_R)
    real_mean = {k: float(v.mean()) for k, v in real_pq.items()}
    real_order = [real_mean[x] for x in SYSTEMS]
    print("real system nDCG:", {k: round(v, 3) for k, v in real_mean.items()}, flush=True)

    # needs (gateway; same wording as before -> cached)
    def abstract(q):
        return chat(client, JUDGE, f"A user searched a Chinese lifestyle app with: 「{q}」. In ONE short neutral Chinese sentence, say what they want. Output ONLY it.", 0.0).splitlines()[0][:80] or q
    with ThreadPoolExecutor(max_workers=24) as ex:
        needs = list(ex.map(abstract, queries))

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    results = {"systems": SYSTEMS, "real_mean_ndcg": real_mean}
    arms = {"default": None, "naive": f"{RUNS}/expC_naive{SUFFIX}_adapter", "coverage": f"{RUNS}/expC_coverage{SUFFIX}_adapter"}
    for arm, adapter in arms.items():
        if adapter and not os.path.isdir(adapter):
            print(f"  skip {arm}", flush=True); continue
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
        model = m if adapter is None else PeftModel.from_pretrained(m, adapter).eval()
        taus, pqfs, dists = [], [], []
        for si in range(SEEDS):
            torch.manual_seed(1000 + si)                                   # generation-seed CI
            gq = gen_local(model, tok, needs)
            gq = [q if q else queries[i] for i, q in enumerate(gq)]
            R = rankings(gq); pq = per_query_ndcg(R)
            order = [float(pq[x].mean()) for x in SYSTEMS]
            taus.append(kendalltau(real_order, order)[0])
            corrs = [pearsonr(pq[x], real_pq[x])[0] for x in SYSTEMS if real_pq[x].std() > 1e-9 and pq[x].std() > 1e-9]
            pqfs.append(float(np.mean(corrs)))
            dists.append(len(set(gq)) / len(gq))
        results[arm] = {"tau_10sys_mean": round(float(np.mean(taus)), 3), "tau_10sys_std": round(float(np.std(taus)), 3),
                        "per_query_fidelity_mean": round(float(np.mean(pqfs)), 4), "per_query_fidelity_std": round(float(np.std(pqfs)), 4),
                        "distinct_q": round(float(np.mean(dists)), 3), "seeds": SEEDS, "pqf_per_seed": [round(p, 4) for p in pqfs]}
        print(f"  {arm:9s} tau={np.mean(taus):+.3f}+/-{np.std(taus):.3f}  pqf={np.mean(pqfs):.4f}+/-{np.std(pqfs):.4f}  (seeds={SEEDS})", flush=True)
        del model, m; torch.cuda.empty_cache()

    json.dump(results, open(f"{RUNS}/exp_C_validity2{SUFFIX}.json", "w"), ensure_ascii=False, indent=1)
    print(f"saved -> ~/v2sim/runs/exp_C_validity2{SUFFIX}.json", flush=True)


if __name__ == "__main__":
    main()
