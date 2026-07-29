"""Generate default-sim and weight-sim utterances for the HELD-OUT iKAT turns (disjoint from the
weight model's SFT train), for the realism->validity test. Saves {tid, real, resolved, default_sim,
weight_sim} so local retrieval can compare how each query retrieves vs the real one.
Run on a GPU node: python exp_ikat_rvgen.py"""
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from exp_ikat_weight import build_prompt, generate, MODEL, DATA, RUNS, DEV

INFILE = sys.argv[1] if len(sys.argv) > 1 else "ikat_turns.jsonl"
OUTFILE = sys.argv[2] if len(sys.argv) > 2 else "ikat_rv_queries.jsonl"
ALL = len(sys.argv) > 3 and sys.argv[3] == "all"               # 'all' => whole file is held-out (e.g. 2024)
# prompt-level terse control (the paper's one-liner): tests whether a terse PROMPT alone removes the
# effectiveness bias, or whether weight-level realism is needed. Same base model as default/weight.
TERSE_STYLE = "Write a short, terse message, about 12 words, and skip pleasantries."


def main():
    recs = [json.loads(l) for l in (DATA / INFILE).read_text().splitlines() if l.strip()]
    if ALL:
        ev = recs                                              # whole year held-out (model trained on 2023 only)
    else:
        random.Random(13).shuffle(recs)
        ev = recs[250:]                                        # 2023 held-out, disjoint from SFT train
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(0)                                       # reproducible sampling across the arms
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
    wmodel = PeftModel.from_pretrained(base, str(RUNS / "ikat_weight_adapter")).eval()
    weight = [c[0] for c in generate(wmodel, tok, ev, 1, 0.7)]
    with wmodel.disable_adapter():
        default = [c[0] for c in generate(wmodel, tok, ev, 1, 0.7)]
        terse = [c[0] for c in generate(wmodel, tok, ev, 1, 0.7, TERSE_STYLE)]     # prompt-level terse control
    out = [{"tid": r["tid"], "real": r["real"], "resolved": r["resolved"], "default_sim": d,
            "terse_prompt_sim": t, "weight_sim": w}
           for r, d, t, w in zip(ev, default, terse, weight)]
    (RUNS / OUTFILE).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out))
    print(f"wrote {len(out)} -> ~/v2sim/runs/{OUTFILE}", flush=True)
    for x in out[:4]:
        print(f"  {x['tid']} | real: {x['real'][:45]!r} | weight: {x['weight_sim'][:45]!r} | default: {x['default_sim'][:45]!r}", flush=True)


if __name__ == "__main__":
    main()
