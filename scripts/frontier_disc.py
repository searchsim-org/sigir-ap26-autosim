"""LEVER-2: frontier-LLM discriminator (a no-training, strong-model probe complementing the trained
classifiers). For each held-out turn, show a frontier LLM the real user's next message and a
simulator's, in random order, and ask which is the ACTUAL HUMAN. Accuracy ~0.5 => even a strong LLM
cannot tell. Run for two model families x {default_sim, weight_sim}.
Run: .venv/bin/python scripts/frontier_disc.py
"""
import json
import math
import random
import urllib.request
from pathlib import Path

from autosim.runtime import load_env

env = load_env()
EP = env["CUSTOM_LLM_ENDPOINT"].rstrip("/")
KEY = env["CUSTOM_LLM_API_KEY"]
MODELS = ["qwen3-next-80b-a3b-instruct", "gemma4-31b-it"]     # two families as independent judges

PROMPT = ("Two messages were each written as the next turn of a user in a conversational web-search "
          "session. One was written by an ACTUAL HUMAN user; the other by an AI simulator imitating a "
          "user. Decide which is the actual human. Reply with ONLY the single letter A or B.\n\n"
          "A: {a}\n\nB: {b}\n\nThe actual human wrote message:")


def chat(model, prompt):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": 4}).encode()
    req = urllib.request.Request(EP + "/v1/chat/completions", body,
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read())["choices"][0]["message"]["content"]


def load_turns():
    turns = []
    for y in ["2023", "2024"]:
        f = Path(f"data/ikat/{y}/ikat_rv_queries.jsonl")
        for l in f.read_text().splitlines():
            if l.strip():
                turns.append(json.loads(l))
    return turns


def run(model, sim_key, turns, seed=0):
    rng = random.Random(seed)
    correct = tot = abstain = 0
    for t in turns:
        real, sim = t.get("real", ""), t.get(sim_key, "")
        if not real or not sim:
            continue
        swap = rng.random() < 0.5
        a, b = (sim, real) if swap else (real, sim)
        human = "B" if swap else "A"
        try:
            ans = chat(model, PROMPT.format(a=a, b=b)).strip().upper()
        except Exception:
            abstain += 1
            continue
        letter = next((c for c in ans if c in "AB"), None)
        if letter is None:
            abstain += 1
            continue
        correct += (letter == human)
        tot += 1
    acc = correct / tot if tot else float("nan")
    # binomial two-sided p vs chance 0.5
    if tot:
        from statistics import NormalDist
        z = (correct - 0.5 * tot) / math.sqrt(0.25 * tot)
        p = 2 * (1 - NormalDist().cdf(abs(z)))
    else:
        p = float("nan")
    return {"acc": round(acc, 3), "n": tot, "abstain": abstain, "p_vs_chance": round(p, 3)}


def main():
    turns = load_turns()
    print(f"held-out turns with utterances: {len(turns)}", flush=True)
    out = {"models": MODELS, "results": {}}
    for model in MODELS:
        out["results"][model] = {}
        for sim_key in ["default_sim", "weight_sim"]:
            r = run(model, sim_key, turns)
            out["results"][model][sim_key] = r
            print(f"  {model:28s} real-vs-{sim_key:12s} acc={r['acc']:.3f} (n={r['n']}, p={r['p_vs_chance']}, abstain={r['abstain']})", flush=True)
    Path("data/ikat/frontier_disc.json").write_text(json.dumps(out, indent=1))
    print("saved -> data/ikat/frontier_disc.json", flush=True)


if __name__ == "__main__":
    main()
