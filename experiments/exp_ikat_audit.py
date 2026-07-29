"""EXP-2 discriminator audit (reviewer-hardened). Puts confidence + power around the AUC claims.
- Embedding-LR AUC for weight & default with 5-fold mean+/-SD and a label-permutation p-value
  (is the AUC distinguishable from 0.5?), so "at chance" is tested, not asserted.
- XLM-RoBERTa with TURN-GROUPED folds: a turn's real + its samples go to the same fold -> no per-turn
  leakage, which would otherwise inflate the default AUC and exaggerate the gap.
- POWER check: real-vs-default (easy, should be ~1) and real-vs-politeness-perturbed-real (a small KNOWN
  shift) -> confirms the discriminator can detect a small difference at this n, so weight~0.5 is not blindness.
- Generic-collapse check: distinct-1/2, mean pairwise token-Jaccard (self-BLEU proxy), length entropy.
Run on a GPU node (GPU): python exp_ikat_audit.py
"""
import json
import math
import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from exp_ikat_weight import generate, MODEL, DATA, RUNS, DEV, TAG

DISC = "xlm-roberta-base"
RNG = np.random.default_rng(0)


def embed(EMB, texts):
    return EMB.encode([t or " " for t in texts], convert_to_numpy=True, show_progress_bar=False)


def foldwise_auc(X, y, seed=1):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return np.array(aucs)


def perm_p(X, y, observed, n=300):
    null = np.array([foldwise_auc(X, RNG.permutation(y)).mean() for _ in range(n)])
    return float((np.abs(null - 0.5) >= abs(observed - 0.5)).mean())


def emb_auc(EMB, real, sim):
    X = np.vstack([embed(EMB, real), embed(EMB, sim)])
    y = np.array([1] * len(real) + [0] * len(sim))
    a = foldwise_auc(X, y)
    return float(a.mean()), float(a.std()), perm_p(X, y, a.mean())


def grouped_xlmr_auc(texts, labels, gids, folds=5):
    """Group folds by gids so a turn's real + its samples never split across train/test (no leakage)."""
    tok = AutoTokenizer.from_pretrained(DISC)
    uniq = sorted(set(gids))
    random.Random(1).shuffle(uniq)
    fold_of = {g: i % folds for i, g in enumerate(uniq)}
    aucs = []
    for f in range(folds):
        te = [i for i, g in enumerate(gids) if fold_of[g] == f]
        tr = [i for i, g in enumerate(gids) if fold_of[g] != f]
        model = AutoModelForSequenceClassification.from_pretrained(DISC, num_labels=2).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        model.train()
        for ep in range(4):
            random.Random(ep).shuffle(tr)
            for i in range(0, len(tr), 16):
                b = tr[i:i + 16]
                enc = tok([texts[j] for j in b], padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEV)
                yb = torch.tensor([labels[j] for j in b]).to(DEV)
                opt.zero_grad(); model(**enc, labels=yb).loss.backward(); opt.step()
        model.eval(); probs, ys = [], []
        with torch.no_grad():
            for i in range(0, len(te), 16):
                b = te[i:i + 16]
                enc = tok([texts[j] for j in b], padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEV)
                probs += torch.softmax(model(**enc).logits, -1)[:, 1].tolist(); ys += [labels[j] for j in b]
        aucs.append(roc_auc_score(ys, probs)); del model; torch.cuda.empty_cache()
    return float(np.mean(aucs)), float(np.std(aucs))


def diversity(utts):
    toks = [u.lower().split() for u in utts if u]
    uni = set(t for u in toks for t in u)
    bi = set((u[i], u[i + 1]) for u in toks for i in range(len(u) - 1))
    n1 = sum(len(u) for u in toks); n2 = sum(max(0, len(u) - 1) for u in toks)
    sets = [set(u) for u in toks if u]; pairs = 0; jac = 0.0
    for _ in range(2000):
        a, b = int(RNG.integers(0, len(sets))), int(RNG.integers(0, len(sets)))
        if a == b:
            continue
        u = sets[a] | sets[b]; jac += (len(sets[a] & sets[b]) / len(u)) if u else 0.0; pairs += 1
    lens = np.array([len(u) for u in toks], float); h = 0.0
    for c in np.unique(lens):
        p = (lens == c).mean(); h -= p * math.log(p + 1e-12)
    return {"distinct1": round(len(uni) / max(n1, 1), 3), "distinct2": round(len(bi) / max(n2, 1), 3),
            "mean_pairwise_jaccard": round(jac / max(pairs, 1), 3), "length_entropy": round(float(h), 3),
            "mean_words": round(float(lens.mean()), 1)}


def main():
    recs = [json.loads(l) for l in (DATA / "ikat_turns.jsonl").read_text().splitlines() if l.strip()]
    random.Random(13).shuffle(recs)
    ev = recs[250:]
    real = [r["real"] for r in ev]
    tids = [r.get("tid", str(i)) for i, r in enumerate(ev)]
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
    wmodel = PeftModel.from_pretrained(base, str(RUNS / f"ikat_weight_adapter{TAG}")).eval()
    gw = generate(wmodel, tok, ev, 2, 0.7)
    weight = [c for r in gw for c in r if c]
    wtids = [tids[i] for i, r in enumerate(gw) for c in r if c]
    with wmodel.disable_adapter():
        gd = generate(wmodel, tok, ev, 2, 0.7)
    default = [c for r in gd for c in r if c]
    dtids = [tids[i] for i, r in enumerate(gd) for c in r if c]
    del base, wmodel; torch.cuda.empty_cache()

    EMB = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEV)
    out = {"model": MODEL, "n_turns": len(ev), "n_weight_utts": len(weight), "n_default_utts": len(default)}

    a, s, p = emb_auc(EMB, real, weight)
    out["emb_weight"] = {"auc": round(a, 3), "sd": round(s, 3), "perm_p_vs_0.5": round(p, 3)}
    print(f"emb-LR real-vs-WEIGHT  AUC={a:.3f}+/-{s:.3f}  perm_p(!=0.5)={p:.3f}", flush=True)
    a, s, p = emb_auc(EMB, real, default)
    out["emb_default"] = {"auc": round(a, 3), "sd": round(s, 3), "perm_p_vs_0.5": round(p, 3)}
    print(f"emb-LR real-vs-DEFAULT AUC={a:.3f}+/-{s:.3f}  perm_p(!=0.5)={p:.3f}", flush=True)

    pert = ["Hi, could you please help me: " + r for r in real]
    a, s, p = emb_auc(EMB, real, pert)
    out["power_real_vs_perturbed"] = {"auc": round(a, 3), "sd": round(s, 3), "perm_p_vs_0.5": round(p, 3)}
    print(f"POWER real-vs-POLITENESS-PERTURB AUC={a:.3f}+/-{s:.3f} perm_p={p:.3f} (want detectable > 0.5)", flush=True)

    aw, sw = grouped_xlmr_auc(real + weight, [1] * len(real) + [0] * len(weight), tids + wtids)
    out["xlmr_weight_grouped"] = {"auc": round(aw, 3), "sd": round(sw, 3)}
    print(f"xlm-roberta (turn-grouped) real-vs-WEIGHT  AUC={aw:.3f}+/-{sw:.3f}", flush=True)
    ad, sd_ = grouped_xlmr_auc(real + default, [1] * len(real) + [0] * len(default), tids + dtids)
    out["xlmr_default_grouped"] = {"auc": round(ad, 3), "sd": round(sd_, 3)}
    print(f"xlm-roberta (turn-grouped) real-vs-DEFAULT AUC={ad:.3f}+/-{sd_:.3f}", flush=True)

    out["diversity_weight"] = diversity(weight)
    out["diversity_real"] = diversity(real)
    print(f"diversity weight={out['diversity_weight']}  real={out['diversity_real']}", flush=True)

    json.dump(out, open(RUNS / f"exp_ikat_audit{TAG}.json", "w"), indent=1)
    print(f"saved -> ~/v2sim/runs/exp_ikat_audit{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
