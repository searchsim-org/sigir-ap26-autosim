"""Real numbers for the behavioral-breakdown table (Table 3): style features of
real iKAT user utterances vs AutoSim with default vs evolved style instruction.
"""
import json
import random

from autosim.runtime import load_env, build_generator
from autosim.metrics.dimensions import _style_features

DATA = "data/ikat/2023/2023_test_topics.json"
DEFAULT = "Write naturally as this user would."
EVOLVED = ("Average 14 words per message. Occasionally include polite phrasing (approx. 10% of the "
           "time) and occasionally use very short, terse phrases of 3 words or fewer (approx. 8%).")


def load_turns():
    topics = json.loads(open(DATA).read())
    out = []
    for tp in topics:
        ptkb = list(tp.get("ptkb", {}).values())
        hist = []
        for t in tp.get("turns", []):
            out.append({"ptkb": ptkb, "history": "\n".join(hist[-4:]) or "(start)",
                        "real": t.get("utterance", ""),
                        "resolved": t.get("resolved_utterance", t.get("utterance", ""))})
            hist.append(f"User: {t.get('utterance', '')}")
            if t.get("response"):
                hist.append(f"Assistant: {t['response'][:200]}")
    return [r for r in out if r["real"]]


def gen_utt(gen, style, rec):
    ask = (f"You are a real user in an ongoing conversational web-search session.\n"
           f"Your background:\n- " + "\n- ".join(rec["ptkb"][:8]) + "\n\n"
           f"Conversation so far:\n{rec['history']}\n\n"
           f"Your next information need (express it as YOUR next message): {rec['resolved']}\n\n{style}\n"
           f"Write ONLY your next message (no quotes, no preamble).")
    return gen.complete(ask, temperature=0.7).text.strip().strip('"').splitlines()[0][:300]


def feats(utts):
    return _style_features([{"text": u} for u in utts])


def main(n=30):
    env = load_env()
    gen = build_generator(env)
    turns = load_turns()
    random.Random(99).shuffle(turns)
    sample = turns[:n]
    real = [t["real"] for t in sample]
    default = [gen_utt(gen, DEFAULT, t) for t in sample]
    evolved = [gen_utt(gen, EVOLVED, t) for t in sample]
    print(f"(n={n} held-out turns)")
    for name, utts in [("real (human)", real), ("AutoSim default", default), ("AutoSim evolved", evolved)]:
        f = feats(utts)
        print(f"{name:18s}  avg_words={f['avg_words']:.1f}  polite%={100 * f['polite_frac']:.0f}  "
              f"short%={100 * f['short_frac']:.0f}")


if __name__ == "__main__":
    main()
