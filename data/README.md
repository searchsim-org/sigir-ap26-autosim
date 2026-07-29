# Data

This folder holds **derived result files** (JSON/TXT) that back the paper's numbers.
It does **not** redistribute any raw corpus — those come from their official sources.

## Included (derived)

- `ikat/` — per-turn behavioral analyses (dialogue acts, ellipsis, retrieval overlap,
  query–need alignment, consequential effectiveness bias, frontier-judge probes) and
  per-year system-ranking results under `ikat/2023|2024|2025/` (τ, CIs, style stats),
  plus the pooled validity file. `ikat/2024/2024-qrels.txt` is the NIST passage-qrels
  subset used for the held-out effectiveness-bias test.
- `cast/` — cross-corpus (CAsT 2020) system-ranking outputs.

## Raw corpora (obtain from the official sources)

| Resource | Source |
|---|---|
| TREC iKAT 2023/2024/2025 (topics, PTKBs, qrels) | https://www.trecikat.com |
| ClueWeb22-B (retrieval corpus) | https://lemurproject.org/clueweb22 |
| TREC CAsT 2020 | https://www.treccast.ai |
| MS MARCO (passage corpus for CAsT) | https://microsoft.github.io/msmarco |

Retrieval runs against ClueWeb22-B/MS MARCO require your own OpenSearch (and optional
SPLADE) endpoint; configure it in `.env` (see `.env.example`).
