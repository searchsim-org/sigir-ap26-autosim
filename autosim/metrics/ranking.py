from scipy.stats import kendalltau


def induced_ranking(pref_scores: dict[str, float]) -> list[str]:
    return [s for s, _ in sorted(pref_scores.items(), key=lambda kv: -kv[1])]


def ranking_tau(pref_scores: dict[str, float], official_order: list[str]) -> float:
    sim_rank = {s: i for i, s in enumerate(induced_ranking(pref_scores))}
    off_rank = {s: i for i, s in enumerate(official_order)}
    common = [s for s in official_order if s in sim_rank]
    tau, _ = kendalltau([sim_rank[s] for s in common], [off_rank[s] for s in common])
    return float(tau)
