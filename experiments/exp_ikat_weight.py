"""SOTA move: WEIGHT-LEVEL adversarial self-improvement of the iKAT user simulator (DIAL/v2 recipe),
to break the ~0.88 prompt-level discriminator band. Same base model (Qwen2.5-3B) carries both a
prompt-level baseline and the weight-level loop, so the prompt-vs-weight comparison is apples-to-apples.

Arms (held-out real-vs-sim discriminator AUC; lower = more human):
  base_default     : 3B, neutral instruction (no weights changed)
  base_promptstyle : 3B, the best prompt-level style instruction (prompt-level ceiling)
  warm_sft         : 3B + LoRA SFT on REAL iKAT utterances (weight-level warm start)
  adv_it{k}        : + adversarial rejection-sampling vs a frozen discriminator (DIAL-style)

Discriminator: local sentence-transformer (loop selection, fast) + octen-embedding via gateway
(eval reporting, comparable to the prompt-level numbers). Runs on a GPU node (msmarco env, A100).
Usage: python exp_ikat_weight.py [iters] [n_samples]
"""
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import AutoModelForCausalLM, AutoTokenizer

HOME = os.path.expanduser("~")
DATA = Path(f"{HOME}/v2sim/data/ikat")
RUNS = Path(f"{HOME}/v2sim/runs")
MODEL = os.environ.get("AUTOSIM_WEIGHT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
TAG = os.environ.get("AUTOSIM_WEIGHT_TAG", "")   # e.g. "_7b" -> model-tagged artifacts, keeps 3B run intact
EMB_MODEL = "octen-embedding-8b"
def _argint(i, d):
    try:
        return int(sys.argv[i])
    except (IndexError, ValueError):
        return d


ITERS = _argint(1, 4)
N = _argint(2, 6)
DEV = "cuda"
PROMPT_STYLE = ("Average 14 words per message. Occasionally include polite phrasing (approx. 10% of the "
                "time) and occasionally use very short, terse phrases of 3 words or fewer (approx. 8% of the time).")
EMB = None


def load_env(path=f"{HOME}/v2sim/.env"):
    e = {}
    for ln in open(path):
        ln = ln.strip()
        if ln and "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1); e[k] = v.strip().strip('"').strip("'")
    return e


def build_prompt(rec, style="Write naturally as this user would."):
    return ("You are a real user in an ongoing conversational web-search session.\n"
            "Your background:\n- " + "\n- ".join(rec["ptkb"][:8]) + "\n\n"
            f"Conversation so far:\n{rec['history']}\n\n"
            f"Your next information need (express it as YOUR next message, in your own words): {rec['resolved']}\n\n"
            f"{style}\nWrite ONLY your next message to the assistant (no quotes, no preamble).")


def chat_text(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, recs, n=1, temp=0.7, style="Write naturally as this user would.", bs=32):
    tok.padding_side = "left"
    texts = [chat_text(tok, build_prompt(r, style)) for r in recs]
    out = [[] for _ in recs]
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to(DEV)
        g = model.generate(**enc, do_sample=(temp > 0), temperature=max(temp, 1e-2), top_p=0.95,
                           max_new_tokens=48, num_return_sequences=n, pad_token_id=tok.pad_token_id)
        g = g[:, enc["input_ids"].shape[1]:]
        dec = tok.batch_decode(g, skip_special_tokens=True)
        for j in range(len(enc["input_ids"])):
            for r in range(n):
                d = dec[j * n + r].strip()
                out[i + j].append(d.splitlines()[0].strip().strip('"')[:200] if d else "")
    return out


def sft(model, tok, prompts, targets, epochs=3, bs=8, lr=1e-4):
    model.train()
    prev = tok.padding_side
    tok.padding_side = "right"
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    data = [(p, t) for p, t in zip(prompts, targets) if t]
    for ep in range(epochs):
        random.shuffle(data)
        for i in range(0, len(data), bs):
            batch = data[i:i + bs]
            full, plens = [], []
            for pr, tg in batch:
                pref = chat_text(tok, pr)
                plens.append(len(tok(pref)["input_ids"]))
                full.append(pref + tg + tok.eos_token)
            enc = tok(full, return_tensors="pt", padding=True).to(DEV)
            lab = enc["input_ids"].clone()
            lab[enc["attention_mask"] == 0] = -100
            for r, pl in enumerate(plens):
                lab[r, :pl] = -100
            opt.zero_grad()
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=lab).loss.backward()
            opt.step()
    tok.padding_side = prev
    model.eval()


def emb_gateway(texts, env):
    ep = env["CUSTOM_LLM_ENDPOINT"].rstrip("/"); key = env["CUSTOM_LLM_API_KEY"]
    out = []
    for i in range(0, len(texts), 48):
        body = json.dumps({"model": EMB_MODEL, "input": [t or " " for t in texts[i:i + 48]]}).encode()
        req = urllib.request.Request(ep + "/v1/embeddings", body,
                                     {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        out += [d["embedding"] for d in json.loads(urllib.request.urlopen(req, timeout=90).read())["data"]]
    return np.array(out, dtype=float)


def auc_from_emb(Xr, Xs):
    X = np.vstack([Xr, Xs]); y = np.array([1] * len(Xr) + [0] * len(Xs))
    return float(cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc").mean())


def local_disc(real, sim):
    Xr = EMB.encode(real, convert_to_numpy=True, show_progress_bar=False)
    Xs = EMB.encode([s or " " for s in sim], convert_to_numpy=True, show_progress_bar=False)
    X = np.vstack([Xr, Xs]); y = np.array([1] * len(Xr) + [0] * len(Xs))
    auc = auc_from_emb(Xr, Xs)
    clf = LogisticRegression(max_iter=2000).fit(X, y)

    def p_sim(utts):
        return clf.predict_proba(EMB.encode([u or " " for u in utts], convert_to_numpy=True, show_progress_bar=False))[:, 0]
    return auc, p_sim


def main():
    global EMB
    env = load_env()
    recs = [json.loads(l) for l in (DATA / "ikat_turns.jsonl").read_text().splitlines() if l.strip()]
    random.Random(13).shuffle(recs)
    train, ev = recs[:250], recs[250:]
    real_ev = [r["real"] for r in ev]
    real_tr = [r["real"] for r in train]
    EMB = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=DEV)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                             lora_dropout=0.05, task_type="CAUSAL_LM"))
    print(f"train={len(train)} eval={len(ev)} ITERS={ITERS} N={N}", flush=True)
    Xr_ev = emb_gateway(real_ev, env)                                   # octen eval embeddings (once)
    traj = []

    def report(name, style=None, adapter_on=True):
        with model.disable_adapter() if not adapter_on else _nullctx():
            sim = [c[0] for c in generate(model, tok, ev, 1, 0.7, style or "Write naturally as this user would.")]
        loc = local_disc(real_ev, sim)[0]
        oct_auc = auc_from_emb(Xr_ev, emb_gateway(sim, env))
        traj.append({"arm": name, "local_auc": round(loc, 3), "octen_auc": round(oct_auc, 3)})
        print(f"  {name:16s} local_auc={loc:.3f}  octen_auc={oct_auc:.3f}", flush=True)

    import contextlib
    @contextlib.contextmanager
    def _nullctx():
        yield

    report("base_default", adapter_on=False)
    report("base_promptstyle", style=PROMPT_STYLE, adapter_on=False)

    # weight-level warm start: SFT the LoRA on REAL utterances
    sft(model, tok, [build_prompt(r) for r in train], real_tr, epochs=3)
    report("warm_sft")

    # adversarial rejection-sampling vs a frozen discriminator (DIAL-style)
    for it in range(ITERS):
        cands = generate(model, tok, train, n=N, temp=1.0)
        cand_lists = [[c for c in cs if c] for cs in cands]
        flat = [c for cs in cand_lists for c in cs]
        _, p_sim = local_disc(real_tr, flat)                            # frozen within the round
        ps = p_sim(flat); pos = 0; targets = []                         # batched scoring
        for cs in cand_lists:
            if cs:
                targets.append(cs[int(np.argmin(ps[pos:pos + len(cs)]))]); pos += len(cs)   # keep most-real
            else:
                targets.append("")
        sft(model, tok, [build_prompt(r) for r in train], targets, epochs=2)
        report(f"adv_it{it + 1}")

    model.save_pretrained(str(RUNS / f"ikat_weight_adapter{TAG}"))
    json.dump({"model": MODEL, "traj": traj}, open(RUNS / f"exp_ikat_weight{TAG}.json", "w"), indent=1)
    print(f"\ntrajectory: {[(t['arm'], t['octen_auc']) for t in traj]}", flush=True)
    print(f"saved -> ~/v2sim/runs/exp_ikat_weight{TAG}.json (+ adapter{TAG}) [model={MODEL}]", flush=True)


if __name__ == "__main__":
    main()
