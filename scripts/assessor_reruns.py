"""Re-score the cached pool passages many times to quantify (a) the LLM-sampling
variance of the headline tau (P1.5) and (b) whether persona-conditioning actually
buys the tau over a plain relevance judge (P1.6).

Uses data/ikat/2023/assessor_inputs.json (cached top-5 passage TEXTS per turn x
system) -> only assessor LLM calls, no retrieval. Reference nDCG is recomputed
from pool_cache_v2.jsonl + qrels (deterministic).

Run: .venv/bin/python scripts/assessor_reruns.py
Out: data/ikat/2023/assessor_reruns.txt (+ .json raw)
"""
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scipy.stats import kendalltau

from autosim.runtime import load_env, build_generator
from autosim.eval.assessor import _parse_unit, _FORMAT
from autosim.data.ikat_real import load_qrels, load_ikat_topics
from autosim.eval.trec_metrics import system_ndcg, system_ndcg_condensed

DATA = Path("data/ikat/2023")
SYSTEMS = ["bm25(raw)", "bm25(resolved)", "bm25(ptkb)",
           "splade(raw)", "splade(resolved)", "splade(ptkb)",
           "rrf(resolved)", "rrf(ptkb)",
           "bm25+rerank", "splade+rerank", "rrf+rerank", "splade(ptkb)+rerank"]
WORKERS = 24


def reference_ndcg():
    done = {json.loads(l)["tid"]: json.loads(l)
            for l in (DATA / "pool_cache_v2.jsonl").read_text().splitlines() if l.strip()}
    qrels = load_qrels(DATA / "2023-qrels.all-turns.txt")
    runs = {n: {} for n in SYSTEMS}
    for tid, rec in done.items():
        for n in SYSTEMS:
            if n in rec["rankings"]:
                runs[n][tid] = [(sid, "", 0.0) for sid in rec["rankings"][n]]
    ndcg = {n: system_ndcg(runs[n], qrels, 5)[0] for n in SYSTEMS}
    ndcg_c = {n: system_ndcg_condensed(runs[n], qrels, 5)[0] for n in SYSTEMS}
    return ndcg, ndcg_c


def persona_ask(ptkb, resolved, psgs):
    persona = "\n".join(f"- {s}" for s in ptkb[:10]) or "- (no profile)"
    p = "\n".join(f"[{i + 1}] {t[:300]}" for i, t in enumerate(psgs[:5])) or "(none)"
    crit = ("Considering who YOU are, judge how well these passages let you answer your question. "
            "Reward passages that specifically and faithfully address it; discount tangential or "
            "keyword-only matches.")
    return (f"You are simulating a specific user in a conversational search session.\n"
            f"About you:\n{persona}\n\nYou asked: \"{resolved}\"\n\n"
            f"The system returned these passages:\n{p}\n\n{crit}\n{_FORMAT}")


def nopersona_ask(ptkb, resolved, psgs):
    p = "\n".join(f"[{i + 1}] {t[:300]}" for i, t in enumerate(psgs[:5])) or "(none)"
    crit = ("Judge how well these passages answer the question. Reward passages that "
            "specifically and faithfully address it; discount tangential or keyword-only matches.")
    return (f"You are a relevance assessor.\n\nQuestion: \"{resolved}\"\n\n"
            f"Passages:\n{p}\n\n{crit}\n{_FORMAT}")


def run_pass(gen, inp, topics, ask_fn, temp):
    tasks = [(tid, n) for tid in inp for n in SYSTEMS if n in inp[tid] and tid in topics]

    def one(t):
        tid, n = t
        ask = ask_fn(topics[tid]["ptkb"], topics[tid]["resolved"], inp[tid][n])
        try:
            return t, _parse_unit(gen.complete(ask, temperature=temp).text.strip())
        except Exception:
            return t, None

    sums = {n: [0.0, 0] for n in SYSTEMS}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for (tid, n), v in ex.map(one, tasks):
            if v is not None:
                sums[n][0] += v
                sums[n][1] += 1
    return {n: (sums[n][0] / sums[n][1] if sums[n][1] else 0.0) for n in SYSTEMS}


def taus(autosim, ndcg, ndcg_c):
    tc = kendalltau([ndcg_c[n] for n in SYSTEMS], [autosim[n] for n in SYSTEMS])[0]
    tr = kendalltau([ndcg[n] for n in SYSTEMS], [autosim[n] for n in SYSTEMS])[0]
    return tc, tr


def main():
    env = load_env()
    gen = build_generator(env)
    inp = json.loads((DATA / "assessor_inputs.json").read_text())
    topics = load_ikat_topics(DATA / "2023_test_topics.json")
    ndcg, ndcg_c = reference_ndcg()

    configs = [("persona@T0", persona_ask, 0.0, 5),
               ("persona@T0.7", persona_ask, 0.7, 5),
               ("nopersona@T0", nopersona_ask, 0.0, 3)]
    raw = {}
    lines = [f"Assessor re-runs: tau vs nDCG@5 over repeated passes ({len(inp)} turns x 12 systems)"]
    for name, fn, temp, reps in configs:
        tcs, trs, sysmeans = [], [], []
        for r in range(reps):
            au = run_pass(gen, inp, topics, fn, temp)
            tc, tr = taus(au, ndcg, ndcg_c)
            tcs.append(tc); trs.append(tr); sysmeans.append(au)
            print(f"  {name} pass {r + 1}/{reps}: cond_tau={tc:+.3f} raw_tau={tr:+.3f}", flush=True)
        raw[name] = {"cond_tau": tcs, "raw_tau": trs, "sysmeans": sysmeans}
        sd = statistics.stdev(tcs) if len(tcs) > 1 else 0.0
        sdr = statistics.stdev(trs) if len(trs) > 1 else 0.0
        lines.append(f"{name:14s} cond_tau {statistics.mean(tcs):+.3f} +/- {sd:.3f}  "
                     f"raw_tau {statistics.mean(trs):+.3f} +/- {sdr:.3f}  (reps={reps}, "
                     f"min={min(tcs):+.3f} max={max(tcs):+.3f})")
    out = "\n".join(lines)
    print("\n" + out, flush=True)
    (DATA / "assessor_reruns.txt").write_text(out + "\n")
    (DATA / "assessor_reruns.json").write_text(json.dumps(raw, indent=2))


if __name__ == "__main__":
    main()
