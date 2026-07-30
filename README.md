<div align="center">

# AutoSim: A Self-Improving User Simulator for Conversational Search Evaluation

### Framework, protocol, and models for a self-improving conversational-search user simulator

*A user simulator that self-improves against real users, and code to test whether that self-improvement makes it a more accurate IR evaluator.*

[![SIGIR-AP 2026](https://img.shields.io/badge/SIGIR--AP_2026-Under_double--blind_review-1f6feb?style=flat-square)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square)](#quick-start)
[![Anon code](https://img.shields.io/badge/code-anonymous.4open.science-6f42c1?style=flat-square)](https://anonymous.4open.science/r/autosim-4876)

![Real-versus-simulated discriminator AUC for four simulator configurations on held-out iKAT: default prompt, tuned prompt, few-shot prompt, and weight-level fine-tuning, each scored by an embedding discriminator and a fine-tuned XLM-RoBERTa discriminator. Every prompt configuration stays at AUC 0.81 to 0.94; only weight-level fine-tuning reaches 0.5, where simulated and real utterances are indistinguishable.](images/fig1_substrate.png)

**Anonymous Author(s)** &nbsp;·&nbsp; Under double-blind review at **SIGIR-AP 2026**

</div>

> **What is this?** Simulating users is the standard way to compare conversational search systems without a human study for every design choice, but a simulator is only trustworthy if its behavior matches real users. AutoSim is a conversational-search user simulator that self-improves against a held-out signal from real users. This repository releases the framework, the evaluation protocol, and the trained weight-level models behind the paper.
>
> **Who is it for?** IR researchers evaluating conversational or interactive systems, and practitioners who want a user simulator whose realism is measured against real users.

**Contents:** [Abstract](#abstract) · [Highlights](#highlights) · [Citation](#citation) · [Quick Start](#quick-start) · [Reproduce the paper](#reproduce-the-paper) · [Framework](#framework) · [Models](#models) · [Human study](#human-study) · [Data](#data) · [Anonymization](#anonymization) · [License](#license)

## Abstract

> User simulators promise to evaluate conversational search systems at a scale human studies cannot reach, but only if their behavior is realistic enough to trust. We study AutoSim, a conversational-search user simulator that self-improves against a held-out signal from real users, and find that a more realistic simulator is a more accurate evaluator, provided the realism is trained into the model weights rather than written into the prompt. AutoSim ranks a controlled pool of retrieval systems in agreement with NIST across three TREC iKAT years (Kendall's τ of 0.66, 0.91, and 0.91; all p ≤ 0.002), a validity that comes from its relevance judgments rather than its persona. Behaving like a real user is not a matter of prompting: no prompt we tried lowers a held-out real-versus-simulated classifier below AUC 0.85, whereas fine-tuning the weights drives it to 0.45, where neither a classifier nor a human reading full dialogues can separate it from real users. Only the weight-level simulator removes a bias that matters for evaluation: where real users retrieve little, a prompted simulator over-estimates retrieval effectiveness by +0.136 nDCG and returns a relevant passage on 57% of the turns where real users find none, enough to change a deployment decision. We release the framework, the protocol, and the models.

**Keywords:** user simulation, conversational search, information retrieval evaluation, self-improving agents, evaluation bias

## Highlights

- **A validated evaluator.** AutoSim's persona-conditioned relevance judgments rank a controlled pool of IR systems in agreement with NIST across three iKAT years (Kendall τ = 0.66 / 0.91 / 0.91, all p ≤ 0.002), and the agreement survives removing the persona.
- **Closing the gap needs weight-level training.** No prompt we tried (hand-written, tuned, or few-shot) lowers a held-out real-versus-simulated discriminator below AUC 0.85. Weight-level fine-tuning drives it to about 0.5, where no classifier, no LLM judge, and no human reading full dialogues (8 of 30) can tell the simulator from a real user.
- **Realism changes the evaluation.** Where real users retrieve little, a prompt simulator over-estimates retrieval effectiveness by +0.136 nDCG and returns a relevant passage on 57% of the turns where real users find none. The weight-level simulator is not detectably biased.

## Quick Start

Requires Python 3.11+. For the double-blind period, download the browsable snapshot from the [anonymous.4open.science link](https://anonymous.4open.science/r/autosim-4876); the `git clone` URL is restored at camera-ready.

```bash
pip install -e .            # or: pip install -r requirements.txt
cp .env.example .env        # then fill in your LLM gateway + OpenSearch endpoint
python scripts/live_smoke.py    # end-to-end smoke test against your endpoints
```

<details>
<summary><b>Configuration and optional dependencies</b></summary>

`autosim` reads endpoints and keys from the environment (`.env`); nothing is hardcoded. See [`.env.example`](.env.example) for the full list: an OpenAI-compatible LLM gateway (`CUSTOM_LLM_ENDPOINT`, `CUSTOM_LLM_API_KEY`), an OpenSearch host over the retrieval corpus (`OPENSEARCH_HOST`), and an optional SPLADE endpoint (`SPLADE_URL`).

- **Core** (simulator, analysis, figures): the base install above.
- **Cross-corpus retrieval** (RQ1 CAsT 2020): `pip install ir_datasets pyserini` and set `JAVA_HOME`.
- **Weight-level self-improvement** (RQ2, GPU): `pip install torch transformers peft accelerate sentence-transformers`.

Raw corpora are obtained from their official sources; see [Data](#data).
</details>

## Reproduce the paper

Each research question maps to entry points under `scripts/` and `experiments/`; committed result files under `data/` and `results/` back every number.

| Result | Entry points |
|---|---|
| **RQ1: validity** (τ across iKAT years, CIs, persona-free check, cross-corpus) | `scripts/sim2real_tau.py`, `scripts/pool14_tau.py`, `scripts/tau_ci.py`, `scripts/reassess_backbone.py`, `scripts/cast_crosscorpus_v4.py` |
| **RQ2: prompt vs. weight discriminator** (prompt-level self-improvement, weight-level training, LLM-judge and human probes) | `scripts/self_improve_behavior.py`, `scripts/e1_discriminator.py`, `experiments/exp_ikat_weight.py`, `scripts/frontier_disc.py` |
| **RQ2: behavioral axes** (retrieval overlap, conversational moves, query–need alignment) | `scripts/retrieval_overlap.py`, `scripts/dialogue_act.py`, `scripts/ellipsis_axis.py` |
| **RQ3: evaluation bias** (effectiveness over-estimate on hard turns) | `scripts/consequential_bias.py` |
| **Figures** | `scripts/make_figures.py` |

## Framework

The `autosim` package is the simulator and its self-improvement loop.

| Component | Path | Role |
|---|---|---|
| Simulator | `autosim/simulator/` | persona, orchestration policy, cognitive functions, and the turn-by-turn driver |
| Self-improvement loop | `autosim/metabolism/` | controller, trace clustering, reflective edits, experience base, and `regression_gate.py` (the acceptance check that keeps an edit only if it does not regress) |
| Metrics | `autosim/metrics/` | the sim-to-real gap (`csg.py`), the real-versus-simulated discriminator (`discriminator.py`), ranking agreement, and behavioral distributions |
| Retrieval | `autosim/retrieval/` | BM25, SPLADE, reciprocal-rank fusion, and cross-encoder reranking over OpenSearch |
| Evaluation | `autosim/eval/` | the LLM relevance assessor and TREC metrics (nDCG, Kendall τ) |
| LLM clients | `autosim/llm/` | OpenAI-compatible gateway client (plus a fake for offline tests) |
| Data | `autosim/data/` | iKAT loaders and real-user trace adapters |

## Models

Three trained **weight-level** simulators (LoRA adapters) live under [`models/`](models/): `adapter_qwen2.5-3b` (primary), `adapter_qwen2.5-7b` (scale check), and `adapter_mistral-7b` (cross-family check). Each drives the real-versus-simulated discriminator to about 0.5. See [`models/README.md`](models/README.md) for loading code and training details.

## Human study

[`human_study/`](human_study/) releases the blind 2-alternative forced-choice discrimination **protocol** ([`PROTOCOL.md`](human_study/PROTOCOL.md)), the browser collection **kit** (`index.html`, which records reaction time), and the stimulus **pairs** (`pairs_key.json`). Raw annotations are not included; run the kit with your own annotators and score with `scripts/score_human_study.py`.

## Data

[`data/`](data/) holds derived result files (per-turn behavioral analyses, per-year system-ranking outputs, and the qrels subset for the bias test), not raw corpora. TREC iKAT, ClueWeb22-B, CAsT 2020, and MS MARCO come from their official sources; see [`data/README.md`](data/README.md).

## Anonymization

This snapshot is prepared for anonymous review: no author identity, institution endpoints read from the environment, and no secrets committed. [`ANONYMIZATION.md`](ANONYMIZATION.md) records what was withheld and lists every `TODO(camera-ready)` to restore for the public release.

## License

Released under the [MIT License](LICENSE). The copyright holder is set at camera-ready.
