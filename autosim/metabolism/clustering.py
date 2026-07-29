def cluster_divergences(divergences: list[dict]) -> list[dict]:
    """Group per-turn sim-vs-real divergences by dimension (P1 simple clustering)."""
    by_dim: dict[str, list[dict]] = {}
    for d in divergences:
        by_dim.setdefault(d["dim"], []).append(d)
    clusters = []
    for dim, items in by_dim.items():
        avg = sum(i["delta"] for i in items) / len(items)
        clusters.append({"dim": dim, "items": items, "avg_delta": avg,
                         "summary": f"{dim} off by {avg:+.2f} over {len(items)} turns"})
    return sorted(clusters, key=lambda c: c["avg_delta"])   # worst (most negative) first
