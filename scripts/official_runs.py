"""Fold the 2 REAL official iKAT-2023 organizer runs (bm25_rm3 auto + manual) into
the picture: compute their nDCG@5 (condensed + raw) on the judged turns, and cache
their top-5 passage TEXTS so AutoSim can later score them (-> a 14-system pool with
2 genuine participant systems, not only constructed ones).

Local only (no API). Run: .venv/bin/python scripts/official_runs.py
Out: data/ikat/2023/official_ndcg.txt + official_inputs.json
"""
import json
from pathlib import Path

from autosim.data.ikat_real import load_ikat_run, load_qrels, load_ikat_topics
from autosim.eval.trec_metrics import system_ndcg, system_ndcg_condensed

DATA = Path("data/ikat/2023")
RUNS = {
    "bm25_rm3(auto)": "ret_bm25_rm3--type_automatic--num_ptkb_3--k_1000--num_psg_3.official.run.json",
    "bm25_rm3(manual)": "ret_bm25_rm3--type_manual--num_ptkb_3--k_1000--num_psg_3.official.run.json",
}


def main():
    qrels = load_qrels(DATA / "2023-qrels.all-turns.txt")
    topics = load_ikat_topics(DATA / "2023_test_topics.json")
    judged = [t for t in qrels if any(v > 0 for v in qrels[t].values()) and t in topics]

    inputs = {}        # {tid: {sysname: [top5 texts]}}
    lines = [f"Official iKAT-2023 runs on {len(judged)} judged turns:"]
    for name, fname in RUNS.items():
        _, ranked = load_ikat_run(DATA / fname)
        run = {tid: [(pid, txt, sc) for pid, txt, sc in ranked[tid]]
               for tid in judged if tid in ranked}
        cov = sum(1 for tid in judged if tid in ranked)
        nd = system_ndcg(run, qrels, 5)[0]
        ndc = system_ndcg_condensed(run, qrels, 5)[0]
        # judged-coverage of top-5 (how much of nDCG is real vs interpolated)
        jc = []
        for tid in run:
            top5 = [pid for pid, _, _ in run[tid][:5]]
            jc.append(sum(1 for pid in top5 if pid in qrels.get(tid, {})) / max(1, len(top5)))
        lines.append(f"  {name:18s} nDCG@5={nd:.4f}  cond_nDCG@5={ndc:.4f}  "
                     f"turn_cov={cov}/{len(judged)}  top5_judged={sum(jc) / len(jc):.2f}")
        for tid in run:
            inputs.setdefault(tid, {})[name] = [txt for _, txt, _ in run[tid][:5]]
    out = "\n".join(lines)
    print(out, flush=True)
    (DATA / "official_ndcg.txt").write_text(out + "\n")
    (DATA / "official_inputs.json").write_text(json.dumps(inputs))
    print(f"cached top-5 texts for {len(inputs)} turns -> official_inputs.json", flush=True)


if __name__ == "__main__":
    main()
