"""E1 (best-paper hardening): is the metabolism loop's behavioral self-improvement beatable by a
TRIVIAL terse prompt? A sharp reviewer says "a one-line 'be terse' prompt = your whole loop."

We compare held-out behavioral CSG (distributional style realism vs real iKAT users) AND the raw
style features (avg words / % polite / % very-short) across:
  - default              : the neutral instruction
  - naive_terse          : a practitioner's one-liner guess
  - naive_keyword        : a second one-liner guess (steelman the baseline)
  - oracle_stats         : the REAL measured hold-set stats hand-written into a prompt (hardest baseline)
  - loop_final           : the style the metabolism loop LEARNED from data (self_improve_behavior.txt)

Same split as self_improve_behavior (seed 13, cal=40, hold=40) so numbers are comparable.
Read: if a one-liner matches loop_final's CSG, the machinery is under-motivated (report honestly, and
the loop's value reduces to AUTOMATED discovery of the target). If the loop wins on the full
distribution while one-liners miss politeness/short-frac, the self-improvement is justified.

Run: .venv/bin/python scripts/e1_terse_baseline.py
"""
import json
import os
import random
from pathlib import Path

from autosim.runtime import load_env, build_generator
from autosim.metrics.dimensions import _style_features, d1_comm_style

YEAR = os.environ.get("AUTOSIM_YEAR", "2023")
DATA = Path(f"data/ikat/{YEAR}")
LOOP_FINAL = ("Average 14 words per message. Occasionally include polite phrasing (approx. 10% of the "
              "time) and occasionally use very short, terse phrases of 3 words or fewer (approx. 8% of the time).")


def load_turns():  # copied from self_improve_behavior.py for a self-contained, identical split
    topics = json.loads((DATA / f"{YEAR}_test_topics.json").read_text())
    out = []
    for tp in topics:
        ptkb = list(tp.get("ptkb", {}).values())
        hist = []
        for t in tp.get("turns", []):
            out.append({"ptkb": ptkb, "history": "\n".join(hist[-4:]) or "(start of conversation)",
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
           f"Your next information need (express it as YOUR next message, in your own words): "
           f"{rec['resolved']}\n\n{style}\n"
           f"Write ONLY your next message to the assistant (no quotes, no preamble).")
    return gen.complete(ask, temperature=0.7).text.strip().strip('"').splitlines()[0][:300]


def feats(utts):
    return _style_features([{"text": u} for u in utts])


def csg(sim, real):
    return d1_comm_style([{"text": u} for u in sim], [{"text": u} for u in real])


def main():
    env = load_env()
    gen = build_generator(env)
    turns = load_turns()
    random.Random(13).shuffle(turns)
    hold = turns[40:80]
    real_hold = [r["real"] for r in hold]
    fr = feats(real_hold)
    print(f"REAL hold: avg {fr['avg_words']:.1f}w, {100 * fr['polite_frac']:.0f}% polite, "
          f"{100 * fr['short_frac']:.0f}% very-short (n={len(hold)})", flush=True)

    oracle = (f"Average about {fr['avg_words']:.0f} words per message. About {100 * fr['polite_frac']:.0f}% "
              f"of the time include polite phrasing. About {100 * fr['short_frac']:.0f}% of the time use a "
              f"very short phrase of 3 words or fewer.")
    arms = {
        "default": "Write naturally as this user would.",
        "naive_terse": "Write a short, terse search message, about 12 words, and skip pleasantries.",
        "naive_keyword": "Reply like a real user typing fast: very brief, blunt, almost keyword-style.",
        "oracle_stats": oracle,
        "loop_final": LOOP_FINAL,
    }
    results = {"real": {"avg_words": fr["avg_words"], "polite_frac": fr["polite_frac"], "short_frac": fr["short_frac"], "n": len(hold)}}
    for name, style in arms.items():
        sim = [gen_utt(gen, style, r) for r in hold]
        c = float(csg(sim, real_hold))
        fs = feats(sim)
        results[name] = {"csg": round(c, 3), "avg_words": round(fs["avg_words"], 1),
                         "polite_frac": round(fs["polite_frac"], 3), "short_frac": round(fs["short_frac"], 3),
                         "style": style}
        print(f"  {name:14s} CSG={c:.3f}  avg={fs['avg_words']:.1f}w  "
              f"polite={100 * fs['polite_frac']:.0f}%  short={100 * fs['short_frac']:.0f}%", flush=True)
    (DATA / "e1_terse_baseline.json").write_text(json.dumps(results, indent=1))
    best_naive = max(results[a]["csg"] for a in ["naive_terse", "naive_keyword"])
    print(f"\nloop_final CSG={results['loop_final']['csg']}  best_naive_oneliner CSG={best_naive}  "
          f"oracle_stats CSG={results['oracle_stats']['csg']}", flush=True)
    print("VERDICT: loop beats one-liners" if results["loop_final"]["csg"] > best_naive + 0.03
          else "VERDICT: one-liner matches loop -> self-improvement value is automated discovery, report honestly", flush=True)
    print("saved -> data/ikat/2023/e1_terse_baseline.json", flush=True)


if __name__ == "__main__":
    main()
