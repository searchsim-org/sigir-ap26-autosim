"""E2 (backbone-swap robustness): re-run the sim-to-real validity with a DIFFERENT assessor model
family, reusing the cached retrievals (no re-retrieval). Fetches the cached top-5 passage texts,
re-scores satisfaction with AUTOSIM_GEN_MODEL (set to deepseek-v31-4bit), and recomputes Kendall tau.
Reports the original (Qwen, from cache) tau alongside, so validity's model-independence is testable.

Run: AUTOSIM_GEN_MODEL=deepseek-v31-4bit AUTOSIM_YEAR=2024 .venv/bin/python scripts/reassess_backbone.py
"""
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scipy.stats import kendalltau

from autosim.runtime import load_env, build_generator, build_retrieval, GENERATOR_MODEL
from autosim.data.ikat_real import load_ikat_topics, load_qrels
from autosim.eval.trec_metrics import system_ndcg_condensed
from autosim.eval.assessor import AutoSimAssessor

YEAR = os.environ.get("AUTOSIM_YEAR", "2024")
DATA = Path(f"data/ikat/{YEAR}")
CACHE = DATA / (f"pool_cache_{YEAR}.jsonl" if YEAR != "2023" else "pool_cache_v2.jsonl")
SYSTEMS = ["bm25(raw)", "bm25(resolved)", "bm25(ptkb)", "splade(raw)", "splade(resolved)", "splade(ptkb)",
           "rrf(resolved)", "rrf(ptkb)", "bm25+rerank", "splade+rerank", "rrf+rerank", "splade(ptkb)+rerank"]


def _retry(fn, tries=4, base=2.0):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                return 0.0
            time.sleep(base * (i + 1))


def main():
    env = load_env()
    bm25 = build_retrieval(env)
    assessor = AutoSimAssessor(build_generator(env))     # model = AUTOSIM_GEN_MODEL (deepseek)
    print(f"re-assessing {YEAR} with model = {GENERATOR_MODEL}", flush=True)
    qrels = load_qrels(next(DATA.glob("*qrels*")))
    topics = load_ikat_topics(DATA / f"{YEAR}_test_topics.json")
    cache = [json.loads(l) for l in CACHE.read_text().splitlines() if l.strip()]
    cache = [r for r in cache if r["tid"] in topics and r["tid"] in qrels and any(v > 0 for v in qrels[r["tid"]].values())]
    print(f"cached judged turns: {len(cache)}", flush=True)

    # fetch all needed passage texts once (batched mget)
    ids = sorted({i for r in cache for n in SYSTEMS for i in r["rankings"].get(n, [])[:5]})
    texts = {}
    for i in range(0, len(ids), 200):
        texts.update(bm25.mget_texts(ids[i:i + 200]))
    print(f"fetched {len(texts)} passage texts", flush=True)

    # re-assess every (turn, system) in parallel with the new model
    jobs = [(r["tid"], n) for r in cache for n in SYSTEMS]
    cachemap = {r["tid"]: r for r in cache}

    def assess(job):
        tid, n = job
        ids5 = cachemap[tid]["rankings"].get(n, [])[:5]
        tx = [texts.get(i, "") for i in ids5]
        s = _retry(lambda: assessor.score_turn(topics[tid]["ptkb"], topics[tid]["resolved"], tx, k=5))
        return (tid, n, s)

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(assess, jobs))
    ds = {n: [] for n in SYSTEMS}
    for tid, n, s in results:
        ds[n].append(s)

    runs = {n: {r["tid"]: [(i, "", 0.0) for i in r["rankings"].get(n, [])] for r in cache} for n in SYSTEMS}
    ndcg_c = {n: system_ndcg_condensed(runs[n], qrels, 5)[0] for n in SYSTEMS}
    deepseek = {n: (sum(ds[n]) / len(ds[n]) if ds[n] else 0.0) for n in SYSTEMS}
    qwen = {n: (sum(r["autosim"][n] for r in cache) / len(cache)) for n in SYSTEMS}

    tau_ds, p_ds = kendalltau([ndcg_c[n] for n in SYSTEMS], [deepseek[n] for n in SYSTEMS])
    tau_qw, p_qw = kendalltau([ndcg_c[n] for n in SYSTEMS], [qwen[n] for n in SYSTEMS])
    tau_agree, p_agree = kendalltau([qwen[n] for n in SYSTEMS], [deepseek[n] for n in SYSTEMS])
    print(f"\n== backbone-swap validity ({YEAR}, {len(cache)} turns, {len(SYSTEMS)} systems) ==", flush=True)
    print(f"  default (Qwen, cache) : tau={tau_qw:+.3f} p={p_qw:.3f}", flush=True)
    print(f"  swapped ({GENERATOR_MODEL}): tau={tau_ds:+.3f} p={p_ds:.3f}", flush=True)
    print(f"  default-vs-swapped agreement: tau={tau_agree:+.3f} p={p_agree:.3f}", flush=True)
    json.dump({"year": YEAR, "n": len(cache), "model": GENERATOR_MODEL,
               "tau_qwen": round(float(tau_qw), 3), "p_qwen": round(float(p_qw), 3),
               "tau_deepseek": round(float(tau_ds), 3), "p_deepseek": round(float(p_ds), 3),
               "tau_agreement": round(float(tau_agree), 3)},
              open(DATA / "reassess_backbone.json", "w"), indent=1)
    print(f"saved -> {DATA/'reassess_backbone.json'}", flush=True)


if __name__ == "__main__":
    main()
