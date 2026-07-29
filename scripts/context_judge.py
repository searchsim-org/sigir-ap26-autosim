"""ACTION-A: firm up the context-aware human-likeness judge. Run a PANEL of strong models through the same
with-context discrimination task a human annotator sees (conversation history + two candidate next
messages; pick the real human), over the 60 study pairs (30 real-vs-weight, 30 real-vs-default). Reports
accuracy per model x condition with binomial CIs + a pooled estimate. Complements the context-free
frontier judges (at chance) and the single AI annotator (weight 20/30).
Run: .venv/bin/python scripts/context_judge.py
"""
import json
import math
import urllib.request
from pathlib import Path
from statistics import NormalDist

from autosim.runtime import load_env

env = load_env()
EP = env["CUSTOM_LLM_ENDPOINT"].rstrip("/")
KEY = env["CUSTOM_LLM_API_KEY"]
MODELS = ["qwen3-next-80b-a3b-instruct", "gemma4-31b-it", "qwen36-35b", "qwen35-35b"]
pairs = json.loads(Path("docs/paper-v3/human_study/pairs_key.json").read_text())

PROMPT = ("In a web-search conversation, one of the two candidate next user messages was written by an "
          "ACTUAL HUMAN user and the other by an AI simulator imitating a user. Using the conversation so "
          "far, decide which is the human.\n\nConversation so far:\n{hist}\n\nA: {a}\n\nB: {b}\n\n"
          "Reply with ONLY the single letter A or B (the actual human):")


def chat(model, prompt):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": 4}).encode()
    req = urllib.request.Request(EP + "/v1/chat/completions", body,
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]


def stat(k, n):
    if not n:
        return {"acc": float("nan"), "n": 0}
    p = k / n
    se = math.sqrt(max(p * (1 - p), 1e-9) / n)
    z = (k - 0.5 * n) / math.sqrt(0.25 * n)
    pv = 2 * (1 - NormalDist().cdf(abs(z)))
    return {"acc": round(p, 3), "n": n, "ci95": [round(max(0, p - 1.96 * se), 3), round(min(1, p + 1.96 * se), 3)],
            "p_vs_chance": round(pv, 3)}


def main():
    out = {"models": MODELS, "n_pairs": len(pairs), "results": {}, "pooled": {}}
    agg = {"default": [0, 0], "weight": [0, 0]}
    for model in MODELS:
        by = {"default": [0, 0], "weight": [0, 0]}
        for pr in pairs:
            try:
                ans = chat(model, PROMPT.format(hist=(pr.get("history") or "")[-600:], a=pr["a"], b=pr["b"]))
            except Exception:
                continue
            letter = next((c for c in (ans or "").strip().upper() if c in "AB"), None)
            if letter is None:
                continue
            c = pr["condition"]
            by[c][1] += 1
            by[c][0] += int(letter == pr["real_side"])
        out["results"][model] = {c: stat(*by[c]) for c in by}
        for c in by:
            agg[c][0] += by[c][0]; agg[c][1] += by[c][1]
        for c in ["default", "weight"]:
            r = out["results"][model][c]
            print(f"  {model:28s} {c:8s} acc={r['acc']} n={r['n']} CI{r.get('ci95')} p={r.get('p_vs_chance')}", flush=True)
    for c in ["default", "weight"]:
        out["pooled"][c] = stat(*agg[c])
        r = out["pooled"][c]
        print(f"  POOLED {c:8s} acc={r['acc']} n={r['n']} CI{r.get('ci95')} p={r.get('p_vs_chance')}", flush=True)
    Path("data/ikat/context_judge.json").write_text(json.dumps(out, indent=1))
    print("saved -> data/ikat/context_judge.json", flush=True)


if __name__ == "__main__":
    main()
