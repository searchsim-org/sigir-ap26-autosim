import numpy as np


def sorensen_dice(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    num = 2 * sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    den = sum(a.get(k, 0.0) for k in keys) + sum(b.get(k, 0.0) for k in keys)
    return num / den if den else 1.0


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def bootstrap_ci(xs, *, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed); xs = np.asarray(xs, float)
    means = [rng.choice(xs, size=len(xs), replace=True).mean() for _ in range(n)]
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
