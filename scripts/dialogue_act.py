"""LEVER-3b: a policy-level behavioral axis -> conversational MOVE (a decision, not style). Classify each
next-user message as NEW (new information request), REFINE (drill into current topic), FEEDBACK (react to
results), or CLARIFY (answer/ask a clarification), via a gateway LLM. Compare the move DISTRIBUTION of
real vs default_sim vs weight_sim (total-variation distance from real). If the default (prompt-level) sim's
move mix diverges from real where the weight-level sim matches, the substrate law extends to move policy.
Run: .venv/bin/python scripts/dialogue_act.py
"""
import json
import urllib.request
from pathlib import Path

from autosim.runtime import load_env

env = load_env()
EP = env["CUSTOM_LLM_ENDPOINT"].rstrip("/")
KEY = env["CUSTOM_LLM_API_KEY"]
MODEL = "gemma4-31b-it"                                        # judge family, different from generators
ACTS = ["NEW", "REFINE", "FEEDBACK", "CLARIFY"]

PROMPT = ("In a web-search conversation, classify the USER's next message as exactly one of:\n"
          "NEW = a new information request on a different topic/aspect;\n"
          "REFINE = drill into or narrow the current topic;\n"
          "FEEDBACK = react to or comment on results;\n"
          "CLARIFY = answer or ask a clarifying question.\n"
          "Conversation so far:\n{hist}\n\nUSER's next message: {msg}\n\nReply with ONE word (NEW/REFINE/FEEDBACK/CLARIFY):")


def classify(hist, msg):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": PROMPT.format(hist=hist[-500:], msg=msg)}],
                       "temperature": 0, "max_tokens": 4}).encode()
    req = urllib.request.Request(EP + "/v1/chat/completions", body,
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=90).read())["choices"][0]["message"]["content"].upper()
    for a in ACTS:
        if a in r:
            return a
    return "NEW"


def dist(labels):
    n = len(labels)
    return {a: round(labels.count(a) / n, 3) for a in ACTS} if n else {}


def tv(p, q):
    return round(0.5 * sum(abs(p.get(a, 0) - q.get(a, 0)) for a in ACTS), 3)


def main():
    turns = []
    for y in ["2023", "2024"]:
        f = Path(f"data/ikat/{y}/ikat_rv_queries.jsonl")
        if f.exists():
            turns += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    print(f"turns: {len(turns)}", flush=True)
    labels = {k: [] for k in ["real", "default_sim", "weight_sim"]}
    for t in turns:
        h = t.get("history", "") or ""
        for k in labels:
            m = (t.get(k) or "").strip()
            if m:
                labels[k].append(classify(h, m))
    out = {"n": len(turns), "distribution": {}, "tv_from_real": {}}
    for k in labels:
        out["distribution"][k] = dist(labels[k])
    for k in ["default_sim", "weight_sim"]:
        out["tv_from_real"][k] = tv(out["distribution"][k], out["distribution"]["real"])
    print("move distribution:", flush=True)
    for k in labels:
        print(f"  {k:12s} {out['distribution'][k]}", flush=True)
    print(f"total-variation distance from REAL:  default={out['tv_from_real']['default_sim']}  "
          f"weight={out['tv_from_real']['weight_sim']}  (lower = closer to real move policy)", flush=True)
    Path("data/ikat/dialogue_act.json").write_text(json.dumps(out, indent=1))
    print("saved -> data/ikat/dialogue_act.json", flush=True)


if __name__ == "__main__":
    main()
