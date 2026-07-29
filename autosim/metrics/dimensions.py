import numpy as np
from autosim.metrics.distributions import sorensen_dice

_POLITE = {"please", "thanks", "thank", "could", "would", "sorry", "kindly"}


def _style_features(records: list[dict]) -> dict[str, float]:
    n = max(1, len(records))
    words = [len(r.get("text", "").split()) for r in records]
    polite = sum(any(w in _POLITE for w in r.get("text", "").lower().split()) for r in records)
    short = sum(1 for w in words if w <= 3)
    return {"avg_words": float(np.mean(words)) if words else 0.0,
            "polite_frac": polite / n, "short_frac": short / n}


def d1_comm_style(sim: list[dict], real: list[dict]) -> float:
    fs, fr = _style_features(sim), _style_features(real)
    # normalize avg_words to a rate before Dice
    fs["avg_words"] /= 50.0; fr["avg_words"] /= 50.0
    return sorensen_dice(fs, fr)


def d6_stopping(sim: list[dict], real: list[dict]) -> float:
    def hist(rs):
        lens = [r.get("session_len", 0) for r in rs]
        h, _ = np.histogram(lens, bins=[0, 3, 6, 9, 12, 100], density=False)
        h = h.astype(float); return h / h.sum() if h.sum() else h
    hs, hr = hist(sim), hist(real)
    return 1.0 - 0.5 * float(np.abs(hs - hr).sum())     # 1 - total variation


def d5_satisfaction_ece(sim: list[dict], real: list[dict]) -> float:
    """Return 1 - ECE so higher = better calibrated to real satisfaction."""
    bins = np.linspace(0, 1, 6); ece = 0.0
    sims = np.array([r.get("satisfaction", 0.0) for r in sim])
    reals = np.array([r.get("satisfaction", 0.0) for r in real])
    for lo, hi in zip(bins[:-1], bins[1:]):
        ms = sims[(sims >= lo) & (sims < hi)]
        if len(ms) == 0: continue
        ece += (len(ms) / len(sims)) * abs(ms.mean() - reals.mean())
    return 1.0 - float(ece)
