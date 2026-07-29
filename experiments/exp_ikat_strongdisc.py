"""Hardening: confirm the weight-level indistinguishability (AUC~0.45) on a STRONG *trained*
discriminator (xlm-roberta fine-tuned), not just embedding+logistic. If a fine-tuned transformer
still cannot separate the weight-level simulator from real users, the substrate law's headline is
bulletproof. Held-out iKAT turns (disjoint from SFT train). Run on a GPU node: python exp_ikat_strongdisc.py
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from exp_ikat_weight import generate, MODEL, DATA, RUNS, DEV, TAG

DISC = "xlm-roberta-base"


def train_disc(real, sim, folds=5):
    tok = AutoTokenizer.from_pretrained(DISC)
    texts = real + sim
    labels = [1] * len(real) + [0] * len(sim)
    idx = list(range(len(texts))); random.Random(1).shuffle(idx)
    n = len(idx); fold = n // folds
    aucs = []
    for f in range(folds):
        te = idx[f * fold:(f + 1) * fold] if f < folds - 1 else idx[f * fold:]
        tr = [i for i in idx if i not in set(te)]
        model = AutoModelForSequenceClassification.from_pretrained(DISC, num_labels=2).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        model.train()
        for ep in range(4):
            random.Random(ep).shuffle(tr)
            for i in range(0, len(tr), 16):
                b = tr[i:i + 16]
                enc = tok([texts[j] for j in b], padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEV)
                y = torch.tensor([labels[j] for j in b]).to(DEV)
                opt.zero_grad(); model(**enc, labels=y).loss.backward(); opt.step()
        model.eval(); probs, ys = [], []
        with torch.no_grad():
            for i in range(0, len(te), 16):
                b = te[i:i + 16]
                enc = tok([texts[j] for j in b], padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEV)
                probs += torch.softmax(model(**enc).logits, -1)[:, 1].tolist()
                ys += [labels[j] for j in b]
        aucs.append(roc_auc_score(ys, probs))
        del model; torch.cuda.empty_cache()
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    recs = [json.loads(l) for l in (DATA / "ikat_turns.jsonl").read_text().splitlines() if l.strip()]
    random.Random(13).shuffle(recs)
    ev = recs[250:]
    real = [r["real"] for r in ev]
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
    wmodel = PeftModel.from_pretrained(base, str(RUNS / f"ikat_weight_adapter{TAG}")).eval()
    # 2 samples/turn to bulk the small held-out set
    weight = [c for r in generate(wmodel, tok, ev, 2, 0.7) for c in r if c]
    with wmodel.disable_adapter():
        default = [c for r in generate(wmodel, tok, ev, 2, 0.7) for c in r if c]
    del base, wmodel; torch.cuda.empty_cache()
    reals = real + real                                     # match the 2-sample sim size
    print(f"held-out turns={len(real)}; discriminator train set ~{len(reals)}+{len(weight)}", flush=True)
    a_d, s_d = train_disc(reals[:len(default)], default)
    print(f"xlm-roberta real-vs-DEFAULT  AUC={a_d:.3f} +/- {s_d:.3f}", flush=True)
    a_w, s_w = train_disc(reals[:len(weight)], weight)
    print(f"xlm-roberta real-vs-WEIGHT   AUC={a_w:.3f} +/- {s_w:.3f}", flush=True)
    json.dump({"model": MODEL, "n_turns": len(real), "auc_default": round(a_d, 3), "auc_weight": round(a_w, 3),
               "std_default": round(s_d, 3), "std_weight": round(s_w, 3)},
              open(RUNS / f"exp_ikat_strongdisc{TAG}.json", "w"), indent=1)
    print(f"saved -> ~/v2sim/runs/exp_ikat_strongdisc{TAG}.json  (strong-disc, model={MODEL})", flush=True)


if __name__ == "__main__":
    main()
