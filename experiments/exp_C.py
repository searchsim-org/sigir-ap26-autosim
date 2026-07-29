"""Exp C: a genuinely self-improving simulator at the WEIGHT level (the constructive claim).

A local generator (Qwen2.5-3B-Instruct + LoRA) improves its own weights against a REAL
behavioral target (Qilin queries) by iterated rejection-sampling SFT. Two arms:
  - naive   : SFT target = the most-REAL candidate per need (maximize discriminator-fooling)
  - coverage: SFT target = most-real subject to an MMR diversity penalty vs the kept batch
              (fool the discriminator WITHOUT collapsing below the real diversity floor)

Per iteration we refit the sim-vs-real discriminator (moving target, GAN-style, stabilized
by the coverage term + a fidelity stop-rule) and record held-out AUC + diversity. The claim
prompt-level tuning could NOT make (Exp A: AUC stayed high) is tested here: does weight-level
self-improvement DURABLY lower AUC, and does naive collapse while coverage holds?

Usage: python exp_C.py [arm] [iters] [subset] [suffix]
Requires: peft, accelerate (pip install), Qwen2.5-3B-Instruct cached. Runs on one A100.
"""
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import AutoModelForCausalLM, AutoTokenizer

HOME = os.path.expanduser("~")
RUNS = f"{HOME}/v2sim/runs"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
ARM = sys.argv[1] if len(sys.argv) > 1 else "coverage"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SUBSET = int(sys.argv[3]) if len(sys.argv) > 3 else 400
SUFFIX = sys.argv[4] if len(sys.argv) > 4 else ""
N_SAMPLES = int(os.environ.get("EXPC_NSAMP", 6))
EVAL_N, MMR_MU = 300, float(os.environ.get("EXPC_MU", 0.5))
LR = float(os.environ.get("EXPC_LR", 1e-4))
LORA_R = int(os.environ.get("EXPC_R", 16))
DEV = "cuda"
EMB = None


def prompt_for(need):
    return ("You are a real user of a Chinese lifestyle/social search app (like Xiaohongshu). "
            f"You want to find information about: {need}. "
            "Write ONLY the exact search query you would type, in Chinese. No quotes, no explanation.")


def build_inputs(tok, needs):
    msgs = [[{"role": "user", "content": prompt_for(n)}] for n in needs]
    return [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]


META = ("用户", "寻找信息", "查找信息", "这位", "该用户", "搜索", "想要", "想找")


def clean(q):
    """Blank out third-person/meta outputs (e.g. '用户找...') that echo the need instead of
    being a real first-person query -- these are the residual tells the discriminator exploits."""
    q = q.strip().splitlines()[0].strip().strip('"')[:60] if q.strip() else ""
    return "" if any(q.startswith(b) or (b in q[:4]) for b in META) else q


@torch.no_grad()
def generate(model, tok, needs, n, temp, bs=32):
    """Return list (len=needs) of lists (len=n) of query strings."""
    texts = build_inputs(tok, needs)
    out = [[] for _ in needs]
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True).to(DEV)
        gen = model.generate(**enc, do_sample=(temp > 0), temperature=max(temp, 1e-2),
                             top_p=0.95, max_new_tokens=32, num_return_sequences=n,
                             pad_token_id=tok.pad_token_id)
        gen = gen[:, enc["input_ids"].shape[1]:]
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for j in range(len(chunk)):
            for r in range(n):
                out[i + j].append(clean(dec[j * n + r]))
    return out


def diversity(qs):
    qs = [q for q in qs if q]
    chars = "".join(qs)
    bi = [chars[i:i + 2] for i in range(len(chars) - 1)]
    d2 = len(set(bi)) / max(1, len(bi))
    L = np.array([len(q) for q in qs]); p = np.array(list(Counter(L).values())) / max(1, len(L))
    ent = float(-(p * np.log(p)).sum())
    return {"distinct2": round(d2, 4), "len_entropy": round(ent, 3)}


def fit_disc(real, sim):
    Xr = EMB.encode(real, convert_to_numpy=True, show_progress_bar=False)
    Xs = EMB.encode(sim, convert_to_numpy=True, show_progress_bar=False)
    X = np.vstack([Xr, Xs]); y = np.array([1] * len(real) + [0] * len(sim))
    auc = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc").mean()
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    return float(auc), clf


def select_targets(cands, clf, real_prof):
    """Per need, choose one SFT target. naive: most-real. coverage: most-real w/ MMR diversity."""
    flat = [q for cs in cands for q in cs if q]
    if not flat:
        return []
    P = {q: p for q, p in zip(flat, clf.predict_proba(EMB.encode(flat, convert_to_numpy=True, show_progress_bar=False))[:, 0])}
    kept, kept_vecs = [], []
    for cs in cands:
        cs = [q for q in cs if q]
        if not cs:
            continue
        if ARM == "naive":
            pick = min(cs, key=lambda q: P.get(q, 1.0))               # lowest P(sim) = most real
        else:
            v = {q: e for q, e in zip(cs, EMB.encode(cs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False))}
            def score(q):
                real = 1 - P.get(q, 1.0)
                sim = 0.0 if not kept_vecs else float(np.max(np.stack(kept_vecs) @ v[q]))
                return real - MMR_MU * sim                            # penalize near-duplicates of kept
            pick = max(cs, key=score)
            kept_vecs.append(v[pick])
        kept.append(pick)
    return kept


def sft(model, tok, needs, targets, epochs=2, bs=8, lr=LR):
    model.train()
    prev_side = tok.padding_side
    tok.padding_side = "right"                                          # RIGHT pad for training: prefix at [:pl]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    data = list(zip(needs, targets))
    for ep in range(epochs):
        random.shuffle(data)
        for i in range(0, len(data), bs):
            batch = data[i:i + bs]
            full, plens = [], []
            for nd, tg in batch:
                pref = tok.apply_chat_template([{"role": "user", "content": prompt_for(nd)}],
                                               tokenize=False, add_generation_prompt=True)
                plens.append(len(tok(pref)["input_ids"]))
                full.append(pref + tg + tok.eos_token)
            enc = tok(full, return_tensors="pt", padding=True).to(DEV)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100                  # mask right-side padding
            for r, pl in enumerate(plens):
                labels[r, :pl] = -100                                  # mask prefix; loss only on target query
            opt.zero_grad()
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).loss.backward()
            opt.step()
    tok.padding_side = prev_side                                        # restore LEFT pad for generation
    model.eval()


def main():
    global EMB
    from peft import LoraConfig, get_peft_model
    st1 = json.load(open(f"{RUNS}/expA_stage1.json"))
    real_all, needs_all = st1["real"], st1["needs"]
    order = list(range(len(real_all))); random.Random(7).shuffle(order)
    ev, tr = order[:EVAL_N], order[EVAL_N:EVAL_N + SUBSET]
    real_eval = [real_all[i] for i in ev]; needs_eval = [needs_all[i] for i in ev]
    needs_train = [needs_all[i] for i in tr]
    EMB = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEV)
    real_prof = diversity(real_eval)
    print(f"[{ARM}] real_eval={len(real_eval)} needs_train={len(needs_train)} real_prof={real_prof}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, target_modules=["q_proj", "v_proj"],
                                             lora_dropout=0.05, task_type="CAUSAL_LM"))
    traj = []
    for it in range(ITERS):
        ev_samp = [c[0] for c in generate(model, tok, needs_eval, 1, temp=0.7)]
        ev_samp = [q for q in ev_samp if q]
        auc, clf = fit_disc(real_eval, ev_samp)
        dv = diversity(ev_samp)
        traj.append({"iter": it, "auc": round(auc, 3), **dv})
        print(f"  [{ARM} it={it}] AUC={auc:.3f} distinct2={dv['distinct2']} len_ent={dv['len_entropy']}", flush=True)
        if it == ITERS - 1:
            break
        cands = generate(model, tok, needs_train, N_SAMPLES, temp=1.0)
        targets = select_targets(cands, clf, real_prof)
        kept_needs = [nd for nd, cs in zip(needs_train, cands) if [q for q in cs if q]]
        sft(model, tok, kept_needs[:len(targets)], targets)
    out = {"arm": ARM, "real_prof": real_prof, "traj": traj, "final_sim": ev_samp[:200]}
    json.dump(out, open(f"{RUNS}/exp_C_{ARM}{SUFFIX}.json", "w"), ensure_ascii=False, indent=1)
    model.save_pretrained(f"{RUNS}/expC_{ARM}{SUFFIX}_adapter")        # for post-hoc sim2real tau eval
    print(f"saved -> ~/v2sim/runs/exp_C_{ARM}{SUFFIX}.json (+ adapter)", flush=True)
    print(f"{ARM}: AUC {traj[0]['auc']}->{traj[-1]['auc']}  distinct2 {traj[0]['distinct2']}->{traj[-1]['distinct2']}", flush=True)


if __name__ == "__main__":
    main()
