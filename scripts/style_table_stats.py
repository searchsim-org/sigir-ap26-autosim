"""P1.8: Table 3 with dispersion + significance. Generates default vs evolved user
utterances over n held-out turns, and reports avg-words / polite% / very-short% with
bootstrap 95% CIs, plus KS and permutation tests of (evolved vs real) and (default vs
real) message-length distributions. Answers 'is evolved indistinguishable from human?'

Run: .venv/bin/python scripts/style_table_stats.py [n]
Out: data/ikat/2023/style_table_stats.txt
"""
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scipy.stats import ks_2samp

from autosim.runtime import load_env, build_generator
from scripts.self_improve_behavior import load_turns, gen_utt
from autosim.metrics.dimensions import _POLITE

import os
DATA = Path(f"data/ikat/{os.environ.get('AUTOSIM_YEAR', '2023')}")
OUTNAME = "style_table_stats.txt" if os.environ.get("AUTOSIM_YEAR", "2023") == "2023" else f"style_table_stats_{os.environ['AUTOSIM_YEAR']}.txt"
DEFAULT = "Write naturally as this user would."
EVOLVED = ("Average 14 words per message. Occasionally include polite phrasing (approx. 10% of the "
           "time) and occasionally use very short, terse phrases of 3 words or fewer (approx. 8%).")
WORKERS = 24


def words(u):
    return len(u.split())


def polite(u):
    return any(w in u.lower() for w in _POLITE)


def boot_ci(vals, fn, B=5000, seed=0):
    rng = random.Random(seed); n = len(vals); out = []
    for _ in range(B):
        out.append(fn([vals[rng.randrange(n)] for _ in range(n)]))
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B)]


def perm_mean_diff(a, b, R=20000, seed=0):
    obs = abs(statistics.mean(a) - statistics.mean(b))
    pool = a + b; na = len(a); rng = random.Random(seed); hits = 0
    for _ in range(R):
        rng.shuffle(pool)
        if abs(statistics.mean(pool[:na]) - statistics.mean(pool[na:])) >= obs - 1e-9:
            hits += 1
    return (hits + 1) / (R + 1)


def main(n=80):
    env = load_env(); gen = build_generator(env)
    turns = load_turns(); random.Random(99).shuffle(turns)
    sample = turns[:n]
    real = [t["real"] for t in sample]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        default = list(ex.map(lambda r: gen_utt(gen, DEFAULT, r), sample))
        evolved = list(ex.map(lambda r: gen_utt(gen, EVOLVED, r), sample))

    lines = [f"Style table with CIs + tests (n={n} held-out turns)"]
    rows = [("real (human)", real), ("AutoSim default", default), ("AutoSim evolved", evolved)]
    for name, utts in rows:
        w = [words(u) for u in utts]
        wl, wh = boot_ci(w, statistics.mean)
        pf = 100 * sum(polite(u) for u in utts) / len(utts)
        sf = 100 * sum(words(u) <= 3 for u in utts) / len(utts)
        lines.append(f"  {name:16s} avg_words={statistics.mean(w):5.1f} [{wl:.1f},{wh:.1f}]  "
                     f"polite%={pf:4.0f}  short%={sf:4.0f}")
    rw = [words(u) for u in real]; ew = [words(u) for u in evolved]; dw = [words(u) for u in default]
    ks_re = ks_2samp(rw, ew); ks_rd = ks_2samp(rw, dw)
    lines += [
        "  word-length distribution tests:",
        f"    real vs evolved : KS D={ks_re.statistic:.3f} p={ks_re.pvalue:.3f}  "
        f"perm mean-diff p={perm_mean_diff(rw, ew):.3f}  "
        f"(evolved {'INDISTINGUISHABLE' if ks_re.pvalue > 0.05 else 'differs'} from human)",
        f"    real vs default : KS D={ks_rd.statistic:.3f} p={ks_rd.pvalue:.4f}  "
        f"perm mean-diff p={perm_mean_diff(rw, dw):.4f}  "
        f"(default {'differs' if ks_rd.pvalue < 0.05 else 'indistinguishable'} from human)",
    ]
    out = "\n".join(lines)
    print(out, flush=True)
    (DATA / OUTNAME).write_text(out + "\n")


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:2]]
    main(*a) if a else main()
