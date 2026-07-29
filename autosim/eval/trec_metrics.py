"""nDCG@k over graded qrels (exponential gain). System ORDERING by nDCG is the
reference ranking AutoSim is validated against; gain convention does not affect
ordering materially.
"""
import math


def _dcg(gains: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: list[str], qrel_map: dict[str, int], k: int = 5) -> float:
    gains = [qrel_map.get(pid, 0) for pid in ranked_ids[:k]]
    ideal = sorted(qrel_map.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def system_ndcg(run: dict[str, list], qrels: dict[str, dict[str, int]], k: int = 5):
    """Mean nDCG@k over judged turns. Returns (mean_ndcg, n_judged_turns)."""
    scores = []
    for tid, ranked in run.items():
        if tid in qrels and any(v > 0 for v in qrels[tid].values()):
            scores.append(ndcg_at_k([pid for pid, _, _ in ranked], qrels[tid], k))
    return (sum(scores) / len(scores) if scores else 0.0), len(scores)


def system_ndcg_condensed(run: dict[str, list], qrels: dict[str, dict[str, int]], k: int = 5):
    """Condensed nDCG@k (Sakai): drop unjudged passages before scoring -> fair to
    non-pooled systems (a reranker surfacing unjudged-relevant passages is not penalised)."""
    scores = []
    for tid, ranked in run.items():
        if tid in qrels and any(v > 0 for v in qrels[tid].values()):
            judged_only = [pid for pid, _, _ in ranked if pid in qrels[tid]]
            scores.append(ndcg_at_k(judged_only, qrels[tid], k))
    return (sum(scores) / len(scores) if scores else 0.0), len(scores)
