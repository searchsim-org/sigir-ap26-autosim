"""DA M1 response: confirm the conversational-move result is not a single-classifier artifact. Classify
each utterance with TWO judge families, report inter-classifier agreement (% and Cohen's kappa) on the
real utterances, and the per-classifier move-distribution TV-distance from real for default vs weight
(the conclusion should hold under both judges).
Run: .venv/bin/python scripts/dialogue_act_agree.py
"""
import json
import urllib.request
from pathlib import Path

from autosim.runtime import load_env

env = load_env()
EP = env["CUSTOM_LLM_ENDPOINT"].rstrip("/")
KEY = env["CUSTOM_LLM_API_KEY"]
JUDGES = ["gemma4-31b-it", "qwen3-next-80b-a3b-instruct"]
ACTS = ["NEW", "REFINE", "FEEDBACK", "CLARIFY"]
PROMPT = ("In a web-search conversation, classify the USER's next message as exactly one of:\n"
          "NEW = a new information request on a different topic/aspect;\nREFINE = drill into or narrow the "
          "current topic;\nFEEDBACK = react to or comment on results;\nCLARIFY = answer or ask a clarifying "
          "question.\nConversation so far:\n{hist}\n\nUSER's next message: {msg}\n\n"
          "Reply with ONE word (NEW/REFINE/FEEDBACK/CLARIFY):")


def classify(model, hist, msg):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": PROMPT.format(hist=hist[-500:], msg=msg)}],
                       "temperature": 0, "max_tokens": 4}).encode()
    req = urllib.request.Request(EP + "/v1/chat/completions", body,
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    r = (json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"] or "").upper()
    return next((a for a in ACTS if a in r), "NEW")


def dist(ls):
    return {a: ls.count(a) / len(ls) for a in ACTS} if ls else {}


def tv(p, q):
    return round(0.5 * sum(abs(p.get(a, 0) - q.get(a, 0)) for a in ACTS), 3)


def main():
    turns = []
    for y in ["2023", "2024"]:
        f = Path(f"data/ikat/{y}/ikat_rv_queries.jsonl")
        if f.exists():
            turns += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    lab = {j: {k: [] for k in ["real", "default_sim", "weight_sim"]} for j in JUDGES}
    pair = {j: [] for j in JUDGES}                                 # aligned real-utterance labels for agreement
    for t in turns:
        h = t.get("history", "") or ""
        for k in lab[JUDGES[0]]:
            m = (t.get(k) or "").strip()
            if not m:
                continue
            for j in JUDGES:
                c = classify(j, h, m)
                lab[j][k].append(c)
                if k == "real":
                    pair[j].append(c)
    a, b = pair[JUDGES[0]], pair[JUDGES[1]]
    n = min(len(a), len(b))
    po = sum(a[i] == b[i] for i in range(n)) / n
    da, db = dist(a), dist(b)
    pe = sum(da.get(x, 0) * db.get(x, 0) for x in ACTS)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    out = {"n_real": n, "agreement": round(po, 3), "cohens_kappa": round(kappa, 3), "per_judge": {}}
    print(f"inter-classifier agreement on real utterances: {po:.3f}  Cohen's kappa={kappa:.3f}", flush=True)
    for j in JUDGES:
        rd = dist(lab[j]["real"])
        out["per_judge"][j] = {"tv_default": tv(dist(lab[j]["default_sim"]), rd),
                               "tv_weight": tv(dist(lab[j]["weight_sim"]), rd),
                               "real_dist": {k: round(v, 3) for k, v in rd.items()}}
        print(f"  {j:28s} TV(default,real)={out['per_judge'][j]['tv_default']}  "
              f"TV(weight,real)={out['per_judge'][j]['tv_weight']}", flush=True)
    Path("data/ikat/dialogue_act_agree.json").write_text(json.dumps(out, indent=1))
    print("saved -> data/ikat/dialogue_act_agree.json", flush=True)


if __name__ == "__main__":
    main()
