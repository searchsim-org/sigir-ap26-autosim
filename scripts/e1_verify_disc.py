"""Robustness check for E1-fair (no-invisible-walls: reproduce before declaring).
Re-score the SAVED arm utterances with a totally independent LEXICAL discriminator
(TF-IDF char n-grams + logistic, no embeddings) and bootstrap the AUC. If loop_final is
still the most human (lowest AUC), the reversal of E1's conclusion is robust to the
discriminator choice, not an octen-embedding artifact.

Run: .venv/bin/python scripts/e1_verify_disc.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

D = Path("data/ikat/2023")
u = json.loads((D / "e1_discriminator_utts.json").read_text())
real = u["real"]
print(f"LEXICAL discriminator (TF-IDF char 3-5grams + logistic, 5-fold AUC), real n={len(real)}")
rng = np.random.default_rng(0)
for name in ["default", "naive_terse", "oracle_stats", "loop_final"]:
    sim = u[name]
    texts = real + sim
    y = np.array([1] * len(real) + [0] * len(sim))
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(texts)
    auc = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc").mean()
    # bootstrap over the 5-fold point estimate by resampling docs
    boots = []
    Xa = X.toarray()
    for _ in range(200):
        idx = rng.integers(0, len(y), len(y))
        try:
            boots.append(cross_val_score(LogisticRegression(max_iter=1000), Xa[idx], y[idx], cv=3, scoring="roc_auc").mean())
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (auc, auc)
    print(f"  {name:14s} AUC={auc:.3f}  95%CI[{lo:.3f},{hi:.3f}]")
