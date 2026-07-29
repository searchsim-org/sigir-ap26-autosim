"""Behavioral Claim B: self-improve AutoSim's USER-UTTERANCE realism (not the assessor).

AutoSim generates the next conversational-search user utterance; CSG = distributional
style match (verbosity / politeness / terseness; USI-style D1) vs REAL iKAT user
utterances. The metabolism loop evolves the generation STYLE instruction over cycles
(the high-leverage surface where the LLM's known over-verbose/over-polite gap lives),
selecting on calibration CSG (a stable distributional signal) and gating; we show
held-out CSG RISES (full) vs static vs no-calibration.

Anchor: real iKAT 2023 raw user utterances (human-authored). Generator (qwen) writes
utterances; judge (gemma) proposes style instructions. No fakes.

Run: .venv/bin/python scripts/self_improve_behavior.py [n_cal] [n_hold] [K] [N]
"""
import json
import os
import random
import sys
from pathlib import Path

from autosim.runtime import load_env, build_generator, build_judge
from autosim.metrics.dimensions import _style_features, d1_comm_style

YEAR = os.environ.get("AUTOSIM_YEAR", "2023")
DATA = Path(f"data/ikat/{YEAR}")
DEFAULT_STYLE = "Write naturally as this user would."   # neutral; LLMs tend to over-explain/over-polite


def load_turns():
    """[{ptkb, history, real, resolved}] over all topic turns (with running history)."""
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


def gen_all(gen, style, recs):
    return [gen_utt(gen, style, r) for r in recs]


def csg(sim_utts, real_utts):
    return d1_comm_style([{"text": u} for u in sim_utts], [{"text": u} for u in real_utts])


def style_gap(sim_utts, real_utts):
    fs = _style_features([{"text": u} for u in sim_utts])
    fr = _style_features([{"text": u} for u in real_utts])
    return (f"Real users: avg {fr['avg_words']:.1f} words/msg, {100 * fr['polite_frac']:.0f}% polite, "
            f"{100 * fr['short_frac']:.0f}% very short (<=3 words). "
            f"Your simulation: avg {fs['avg_words']:.1f} words, {100 * fs['polite_frac']:.0f}% polite, "
            f"{100 * fs['short_frac']:.0f}% short.")


def reflect_n(judge, style, gap, n):
    base = (f"You tune the STYLE INSTRUCTION given to a user simulator so its messages match how REAL "
            f"search users write.\nCurrent style instruction:\n{style}\n\nObserved gap:\n{gap}\n\n"
            f"Write an improved style instruction (1-3 sentences) so the simulated user matches the real "
            f"distribution (message length, politeness, terseness). Output ONLY the instruction.")
    out = []
    for _ in range(n):
        c = judge.complete(base, temperature=0.8).text.strip()
        if 10 < len(c) < 500 and c not in out:
            out.append(c)
    return out or [style]


def main(n_cal=40, n_hold=40, K=2, N=4):
    env = load_env()
    gen = build_generator(env)
    judge = build_judge(env)
    turns = load_turns()
    random.Random(13).shuffle(turns)
    cal, hold = turns[:n_cal], turns[n_cal:n_cal + n_hold]
    real_cal = [r["real"] for r in cal]
    real_hold = [r["real"] for r in hold]
    print(f"cal={len(cal)} hold={len(hold)} K={K} N={N} | behavioral CSG (style realism)", flush=True)

    def hold_csg(style):
        return csg(gen_all(gen, style, hold), real_hold)

    csg0 = hold_csg(DEFAULT_STYLE)

    # FULL: evolve style instruction; select best by calibration CSG; gate
    P, full = DEFAULT_STYLE, [csg0]
    for k in range(K):
        sim_cal = gen_all(gen, P, cal)
        cur = csg(sim_cal, real_cal)
        cands = reflect_n(judge, P, style_gap(sim_cal, real_cal), N)
        scored = [(c, csg(gen_all(gen, c, cal), real_cal)) for c in cands]
        bc, best = max(scored, key=lambda x: x[1])
        if best >= cur:
            P = bc
        full.append(hold_csg(P))
        print(f"[full k={k + 1}] cur_csg={cur:.3f} best_csg={best:.3f} "
              f"{'ACCEPT' if best >= cur else 'reject'} hold_csg={full[-1]:.3f}", flush=True)

    # NO-CALIBRATION: blind style edits
    P2, nocal = DEFAULT_STYLE, [csg0]
    for k in range(K):
        P2 = reflect_n(judge, P2, "(no calibration feedback provided)", N)[0]
        nocal.append(hold_csg(P2))
        print(f"[nocal k={k + 1}] hold_csg={nocal[-1]:.3f}", flush=True)

    static = [round(float(csg0), 3)] * (K + 1)
    out = ("held-out behavioral CSG (style realism; higher = closer to real users) over cycles "
           f"(cal={len(cal)}, hold={len(hold)}, K={K}, N={N}):\n"
           f"  full     : {[round(float(x), 3) for x in full]}\n"
           f"  static   : {static}\n"
           f"  no-calib : {[round(float(x), 3) for x in nocal]}\n\n"
           f"final style instruction:\n{P}\n")
    print("\n" + out, flush=True)
    (DATA / "self_improve_behavior.txt").write_text(out)


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:5]]
    main(*a) if a else main()
