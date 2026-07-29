"""ACTION-B: the strongest prompt baseline the reviewers asked for -> FEW-SHOT prompting with real terse
utterances as in-context exemplars (base model, no weight updates), audited with the same leakage-safe
discriminators. If few-shot ALSO saturates (cannot reach chance), "prompt-level saturates ~0.85" is
earned against the strongest prompt; if it reaches chance, the claim needs qualification.
Run on a GPU node: python exp_ikat_fewshot.py
"""
import json
import random

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_ikat_weight import generate, MODEL, DATA, RUNS, DEV, TAG
from exp_ikat_audit import emb_auc, grouped_xlmr_auc


def main():
    recs = [json.loads(l) for l in (DATA / "ikat_turns.jsonl").read_text().splitlines() if l.strip()]
    random.Random(13).shuffle(recs)
    train, ev = recs[:250], recs[250:]
    real = [r["real"] for r in ev]
    tids = [r.get("tid", str(i)) for i, r in enumerate(ev)]
    # few-shot exemplars: real, terse utterances drawn from the TRAIN split (disjoint from eval)
    exs = [r["real"] for r in train if 4 <= len(r["real"].split()) <= 18][:8]
    style = ("Real users in this setting write terse, elliptical messages. Here are real examples:\n- "
             + "\n- ".join(exs) + "\nWrite your next message in exactly that terse, real style.")
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
    gw = generate(base, tok, ev, 2, 0.7, style)
    fs = [c for r in gw for c in r if c]
    ftids = [tids[i] for i, r in enumerate(gw) for c in r if c]
    del base
    torch.cuda.empty_cache()

    EMB = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEV)
    a, s, p = emb_auc(EMB, real, fs)
    aw, sw = grouped_xlmr_auc(real + fs, [1] * len(real) + [0] * len(fs), tids + ftids)
    out = {"model": MODEL, "arm": "few_shot_prompt", "n_exemplars": len(exs),
           "emb_auc": round(a, 3), "emb_sd": round(s, 3), "emb_perm_p_vs_0.5": round(p, 3),
           "xlmr_grouped_auc": round(aw, 3), "xlmr_sd": round(sw, 3),
           "mean_words": round(float(np.mean([len(x.split()) for x in fs])), 1)}
    print(out, flush=True)
    json.dump(out, open(RUNS / f"exp_ikat_fewshot{TAG}.json", "w"), indent=1)
    print(f"saved -> ~/v2sim/runs/exp_ikat_fewshot{TAG}.json  DONE_FEWSHOT", flush=True)


if __name__ == "__main__":
    main()
