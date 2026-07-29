import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def discriminator_score(sim: list[dict], human: list[dict], *, seed: int = 0) -> float:
    """Train a human-vs-sim classifier; return 2*|AUC-0.5| (0 = indistinguishable = ideal)."""
    texts = [r.get("text", "") for r in sim] + [r.get("text", "") for r in human]
    y = np.array([0] * len(sim) + [1] * len(human))
    if len(set(y)) < 2 or len(texts) < 4:
        return 0.0
    X = TfidfVectorizer().fit_transform(texts)
    clf = LogisticRegression(max_iter=200, random_state=seed)
    cv = min(3, len(sim), len(human))
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()
    return float(min(1.0, 2 * abs(auc - 0.5)))
