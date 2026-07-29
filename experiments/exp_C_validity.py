"""Exp C -> validity knockout: run each self-improved generator through Exp B's exact
real-behavior system-ranking machinery and report sim2real Kendall tau per arm.

Rebuilds Exp B's corpus + real click-based system ranking (same seeds => identical setup),
abstracts the 600 click-queries into needs, then for each arm generates a query per need
with that arm's generator and computes tau(system ranking induced, real ranking):
  - default  : base Qwen2.5-3B (no adapter)
  - naive    : base + naive LoRA adapter    (expected: collapsed -> tau DROPS)
  - coverage : base + coverage LoRA adapter (expected: preserved -> tau HOLDS/RISES)

Usage: python exp_C_validity.py [suffix]   (suffix matches the adapters, e.g. _r0)
Mirrors exp_B.py's harness exactly so real_order is identical.
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
import torch
from openai import OpenAI
from peft import PeftModel
from rank_bm25 import BM25Okapi
from scipy.stats import kendalltau
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

HOME = os.path.expanduser("~")
QROOT = f"{HOME}/v2sim/data/qilin"
RUNS = f"{HOME}/v2sim/runs"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
GEN, JUDGE = "qwen3-next-80b-a3b-instruct", "gemma4-31b-it"
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "_r0"
M, NRAND = 600, 20000
random.seed(0); np.random.seed(0)          # identical to exp_B.py
DEV = "cuda"
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


def toks(s):
    s = str(s); return [s[i:i + 2] for i in range(len(s) - 1)] or [s]


def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg(ranked_ids, rel, k=10):
    g = [rel.get(i, 0) for i in ranked_ids[:k]]
    idc = dcg(sorted(rel.values(), reverse=True)[:k])
    return dcg(g) / idc if idc > 0 else 0.0


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
        g = g[:, enc["input_ids"].shape[1]:]
        for d in tok.batch_decode(g, skip_special_tokens=True):
            out.append((d.strip().splitlines()[0].strip().strip('"')[:60]) if d.strip() else "")
    return out


def main():
    global EMB
    env = load_env()
    client = OpenAI(base_url=env["CUSTOM_LLM_ENDPOINT"].rstrip("/") + "/v1", api_key=env["CUSTOM_LLM_API_KEY"])
    # --- rebuild Exp B setup (identical seeds) ---
    s = pd.read_parquet(sorted(glob.glob(f"{QROOT}/search_train/*.parquet"))[0])
    s = s[s["query"].notna()].reset_index(drop=True)
    idx = random.sample(range(len(s)), M)
    queries, rels, serp_notes = [], [], set()
    for i in idx:
        row = s.iloc[i]; rel = {}
        for d in row.search_result_details_with_idx:
            nid = int(d["note_idx"]); serp_notes.add(nid)
            sc = 2 if (d.get("click") and (d.get("collect") or d.get("like") or (d.get("page_time", 0) or 0) > 10)) else (1 if d.get("click") else 0)
            if sc: rel[nid] = max(rel.get(nid, 0), sc)
        if rel:
            queries.append(str(row["query"])); rels.append(rel)
    notes = pd.concat([pd.read_parquet(f)[["note_idx", "note_title", "note_content"]] for f in sorted(glob.glob(f"{QROOT}/notes/*.parquet"))], ignore_index=True)
    notes["note_idx"] = notes["note_idx"].astype(int)
    rand_ids = set(np.random.choice(notes["note_idx"].values, size=min(NRAND, len(notes)), replace=False).tolist())
    corpus_ids = sorted(serp_notes | rand_ids)
    nmap = notes.set_index("note_idx")
    text = {nid: (str(nmap.loc[nid, "note_title"]) + " " + str(nmap.loc[nid, "note_content"]))[:256] for nid in corpus_ids if nid in nmap.index}
    corpus_ids = [c for c in corpus_ids if c in text]
    docs = [text[c] for c in corpus_ids]
    print(f"queries={len(queries)} corpus={len(corpus_ids)}", flush=True)

    EMB = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEV)
    bm25 = BM25Okapi([toks(d) for d in docs])
    dmat = EMB.encode(docs, convert_to_numpy=True, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
    SYSTEMS = ["dense", "hybrid", "bm25", "bm25_trunc", "dense_trunc", "random"]

    def retrieve(q, system, k=10):
        if system == "random":
            return random.sample(corpus_ids, k)
        qq = q[:2] if "trunc" in system else q
        if system.startswith("bm25"):
            return [corpus_ids[j] for j in np.argsort(-bm25.get_scores(toks(qq)))[:k]]
        if system.startswith("dense"):
            qv = EMB.encode([qq], convert_to_numpy=True, normalize_embeddings=True)[0]
            return [corpus_ids[j] for j in np.argsort(-(dmat @ qv))[:k]]
        if system == "hybrid":
            b = np.argsort(-bm25.get_scores(toks(q)))[:50]
            qv = EMB.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
            d = np.argsort(-(dmat @ qv))[:50]
            rr = {}
            for r, j in enumerate(b): rr[j] = rr.get(j, 0) + 1 / (60 + r)
            for r, j in enumerate(d): rr[j] = rr.get(j, 0) + 1 / (60 + r)
            return [corpus_ids[j] for j in sorted(rr, key=lambda j: -rr[j])[:k]]

    def system_ranking(qlist):
        return {sysn: float(np.mean([ndcg(retrieve(q, sysn), rel) for q, rel in zip(qlist, rels)])) for sysn in SYSTEMS}

    real_scores = system_ranking(queries)
    real_order = [real_scores[x] for x in SYSTEMS]
    print("real:", {k: round(v, 4) for k, v in real_scores.items()}, flush=True)

    # --- abstract the click-queries into needs (gateway, deterministic) ---
    def abstract(q):
        return chat(client, JUDGE, f"A user searched a Chinese lifestyle app with: 「{q}」. In ONE short neutral Chinese sentence, say what they want. Output ONLY it.", 0.0).splitlines()[0][:80] or q
    with ThreadPoolExecutor(max_workers=24) as ex:
        needs = list(ex.map(abstract, queries))

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    results = {"real_scores": real_scores}
    arms = {"default": None, "naive": f"{RUNS}/expC_naive{SUFFIX}_adapter", "coverage": f"{RUNS}/expC_coverage{SUFFIX}_adapter"}
    for arm, adapter in arms.items():
        if adapter and not os.path.isdir(adapter):
            print(f"  skip {arm}: no adapter at {adapter}", flush=True); continue
        # reload a FRESH base each arm -- PeftModel injects LoRA into base in-place, so reusing
        # one base object across arms would stack/corrupt adapters.
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
        model = m if adapter is None else PeftModel.from_pretrained(m, adapter).eval()
        gq = gen_local(model, tok, needs)
        gq = [q if q else queries[i] for i, q in enumerate(gq)]
        sc = system_ranking(gq); order = [sc[x] for x in SYSTEMS]
        tau = kendalltau(real_order, order)[0]
        distinct = len(set(gq)) / len(gq)
        results[arm] = {"tau_vs_real": round(float(tau), 3), "distinct_q": round(distinct, 3), "sys_ndcg": {k: round(v, 4) for k, v in sc.items()}}
        print(f"  {arm:9s} distinct={distinct:.3f}  tau={tau:+.3f}", flush=True)
        del model, m; torch.cuda.empty_cache()

    json.dump(results, open(f"{RUNS}/exp_C_validity{SUFFIX}.json", "w"), ensure_ascii=False, indent=1)
    print(f"saved -> ~/v2sim/runs/exp_C_validity{SUFFIX}.json", flush=True)


if __name__ == "__main__":
    main()
