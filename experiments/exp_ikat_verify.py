"""Verify the weight-level iKAT result is genuine, not mode-collapse or memorization.
Loads the saved adapter, regenerates eval utterances (weight vs base), and checks:
  (1) diversity (distinct-1 word / distinct-2 char / unique-ratio / length stats) vs REAL and base,
  (2) verbatim copy of TRAIN real utterances (memorization) and of the EVAL target,
  (3) an independent LEXICAL discriminator AUC (confirm the octen result),
  (4) sample utterances for eyeballing.
Run on a GPU node: python exp_ikat_verify.py
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from exp_ikat_weight import build_prompt, generate, MODEL, DATA, RUNS, DEV


def diversity(utts):
    utts = [u for u in utts if u]
    chars = "".join(utts)
    bi = [chars[i:i + 2] for i in range(len(chars) - 1)]
    toks = " ".join(utts).split()
    L = np.array([len(u.split()) for u in utts])
    return {"distinct1_word": round(len(set(toks)) / max(1, len(toks)), 3),
            "distinct2_char": round(len(set(bi)) / max(1, len(bi)), 3),
            "uniq_ratio": round(len(set(utts)) / max(1, len(utts)), 3),
            "len_mean": round(float(L.mean()), 1), "len_std": round(float(L.std()), 1)}


def lex_auc(real, sim):
    texts = real + [s or " " for s in sim]
    y = np.array([1] * len(real) + [0] * len(sim))
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(texts)
    return float(cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc").mean())


def main():
    recs = [json.loads(l) for l in (DATA / "ikat_turns.jsonl").read_text().splitlines() if l.strip()]
    random.Random(13).shuffle(recs)
    train, ev = recs[:250], recs[250:]
    real_ev = [r["real"] for r in ev]
    train_real = set(r["real"] for r in train)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
    wmodel = PeftModel.from_pretrained(base, str(RUNS / "ikat_weight_adapter")).eval()
    weight_utts = [c[0] for c in generate(wmodel, tok, ev, 1, 0.7)]
    with wmodel.disable_adapter():
        base_utts = [c[0] for c in generate(wmodel, tok, ev, 1, 0.7)]

    print("== diversity (distinct/unique/length) ==", flush=True)
    print(f"  REAL    {diversity(real_ev)}", flush=True)
    print(f"  base    {diversity(base_utts)}", flush=True)
    print(f"  weight  {diversity(weight_utts)}", flush=True)
    memo = sum(1 for u in weight_utts if u in train_real) / len(weight_utts)
    copytgt = sum(1 for u, r in zip(weight_utts, real_ev) if u.strip() == r.strip()) / len(weight_utts)
    print(f"\n== leakage ==\n  verbatim-copy of TRAIN real: {memo:.3f}   copy of EVAL target: {copytgt:.3f}", flush=True)
    print(f"\n== independent LEXICAL discriminator AUC (real_ev vs sim) ==\n"
          f"  base   {lex_auc(real_ev, base_utts):.3f}\n  weight {lex_auc(real_ev, weight_utts):.3f}", flush=True)
    print("\n== samples (real || weight-sim) ==", flush=True)
    for i in range(8):
        print(f"  R: {real_ev[i][:70]!r}\n  W: {weight_utts[i][:70]!r}", flush=True)
    json.dump({"diversity": {"real": diversity(real_ev), "base": diversity(base_utts), "weight": diversity(weight_utts)},
               "verbatim_train": memo, "copy_target": copytgt,
               "lex_auc": {"base": lex_auc(real_ev, base_utts), "weight": lex_auc(real_ev, weight_utts)}},
              open(RUNS / "exp_ikat_verify.json", "w"), indent=1)
    print("\nsaved -> ~/v2sim/runs/exp_ikat_verify.json", flush=True)


if __name__ == "__main__":
    main()
