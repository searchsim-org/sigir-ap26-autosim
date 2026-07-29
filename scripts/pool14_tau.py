"""14-system validity: the 12 constructed systems PLUS the 2 real official iKAT-2023
organizer runs (bm25_rm3 auto + manual), all scored by AutoSim in ONE fresh pass so
they are directly comparable. Tests whether AutoSim ranks a pool that *includes real
participant systems* like NIST.

Needs assessor_inputs.json (12 constructed) + official_inputs.json (2 official) +
pool_cache_v2.jsonl + the official run files. API: 176 x 14 ~ 2.5k assessor calls.
Run: .venv/bin/python scripts/pool14_tau.py
Out: data/ikat/2023/pool14_tau.txt
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scipy.stats import kendalltau

from autosim.runtime import load_env, build_generator
from autosim.eval.assessor import _parse_unit, _FORMAT
from autosim.data.ikat_real import load_qrels, load_ikat_topics, load_ikat_run
from autosim.eval.trec_metrics import system_ndcg, system_ndcg_condensed

DATA = Path("data/ikat/2023")
CONSTRUCTED = ["bm25(raw)", "bm25(resolved)", "bm25(ptkb)",
               "splade(raw)", "splade(resolved)", "splade(ptkb)",
               "rrf(resolved)", "rrf(ptkb)",
               "bm25+rerank", "splade+rerank", "rrf+rerank", "splade(ptkb)+rerank"]
OFFICIAL = {
    "bm25_rm3(auto)*": "ret_bm25_rm3--type_automatic--num_ptkb_3--k_1000--num_psg_3.official.run.json",
    "bm25_rm3(manual)*": "ret_bm25_rm3--type_manual--num_ptkb_3--k_1000--num_psg_3.official.run.json",
}
WORKERS = 24


def persona_ask(ptkb, resolved, psgs):
    persona = "\n".join(f"- {s}" for s in ptkb[:10]) or "- (no profile)"
    p = "\n".join(f"[{i + 1}] {t[:300]}" for i, t in enumerate(psgs[:5])) or "(none)"
    crit = ("Considering who YOU are, judge how well these passages let you answer your question. "
            "Reward passages that specifically and faithfully address it; discount tangential or "
            "keyword-only matches.")
    return (f"You are simulating a specific user in a conversational search session.\n"
            f"About you:\n{persona}\n\nYou asked: \"{resolved}\"\n\n"
            f"The system returned these passages:\n{p}\n\n{crit}\n{_FORMAT}")


def boot_ci(x, y, B=10000, seed=0):
    rng = random.Random(seed); n = len(x); taus = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        t = kendalltau([x[i] for i in idx], [y[i] for i in idx])[0]
        if t == t:
            taus.append(t)
    taus.sort()
    return taus[int(0.025 * len(taus))], taus[int(0.975 * len(taus))]


def main():
    qrels = load_qrels(DATA / "2023-qrels.all-turns.txt")
    topics = load_ikat_topics(DATA / "2023_test_topics.json")
    cache = {json.loads(l)["tid"]: json.loads(l)
             for l in (DATA / "pool_cache_v2.jsonl").read_text().splitlines() if l.strip()}
    judged = list(cache)

    # reference nDCG: constructed (from cache rankings) + official (from run files)
    runs = {n: {tid: [(sid, "", 0.0) for sid in cache[tid]["rankings"][n]]
                for tid in judged if n in cache[tid]["rankings"]} for n in CONSTRUCTED}
    inp = json.loads((DATA / "assessor_inputs.json").read_text())          # 12 constructed texts
    off_in = json.loads((DATA / "official_inputs.json").read_text())        # 2 official texts
    for name, fname in OFFICIAL.items():
        _, ranked = load_ikat_run(DATA / fname)
        runs[name] = {tid: ranked[tid] for tid in judged if tid in ranked}
        for tid in judged:
            if tid in off_in and name.replace("*", "") in off_in[tid]:
                inp.setdefault(tid, {})[name] = off_in[tid][name.replace("*", "")]

    systems = CONSTRUCTED + list(OFFICIAL)
    ndcg = {n: system_ndcg(runs[n], qrels, 5)[0] for n in systems}
    ndcg_c = {n: system_ndcg_condensed(runs[n], qrels, 5)[0] for n in systems}

    # fresh AutoSim pass over all 14 systems
    env = load_env(); gen = build_generator(env)
    tasks = [(tid, n) for tid in judged for n in systems if tid in inp and n in inp[tid]]

    def one(t):
        tid, n = t
        try:
            return t, _parse_unit(gen.complete(
                persona_ask(topics[tid]["ptkb"], topics[tid]["resolved"], inp[tid][n]),
                temperature=0.0).text.strip())
        except Exception:
            return t, None
    acc = {n: [0.0, 0] for n in systems}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for (tid, n), v in ex.map(one, tasks):
            if v is not None:
                acc[n][0] += v; acc[n][1] += 1
    autosim = {n: (acc[n][0] / acc[n][1] if acc[n][1] else 0.0) for n in systems}

    order = sorted(systems, key=lambda n: -ndcg_c[n])
    lines = [f"14-system pool (12 constructed + 2 real official*), {len(judged)} turns",
             f"{'system':22s} {'nDCG@5':>8s} {'cond':>8s} {'AutoSim':>8s}"]
    for n in order:
        lines.append(f"{n:22s} {ndcg[n]:8.4f} {ndcg_c[n]:8.4f} {autosim[n]:8.4f}")
    for label, ref in [("condensed nDCG@5", ndcg_c), ("raw nDCG@5", ndcg)]:
        x = [ref[n] for n in systems]; y = [autosim[n] for n in systems]
        tau, p = kendalltau(x, y)
        lo, hi = boot_ci(x, y)
        lines.append(f"Kendall tau (AutoSim, {label:16s}) = {tau:+.3f} (p={p:.3f})  "
                     f"system-bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
    out = "\n".join(lines)
    print("\n" + out, flush=True)
    (DATA / "pool14_tau.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()
