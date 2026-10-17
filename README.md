<div align="center">

# AutoSim: A Self-Improving User Simulator for Conversational Search Evaluation

### Framework, protocol, and models for a self-improving conversational-search user simulator

*A user simulator that self-improves against real users, and code to test whether that self-improvement makes it a more accurate IR evaluator.*

[![SIGIR-AP 2026](https://img.shields.io/badge/SIGIR--AP_2026-Full_Paper-1f6feb?style=flat-square)](https://doi.org/10.1145/3848000.3850196)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3848000.3850196-1f6feb?style=flat-square)](https://doi.org/10.1145/3848000.3850196)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square)](#quick-start)

![Real-versus-simulated discriminator AUC for four simulator configurations on held-out iKAT: default prompt, tuned prompt, few-shot prompt, and weight-level fine-tuning, each scored by an embedding discriminator and a fine-tuned XLM-RoBERTa discriminator. Every prompt configuration stays at AUC 0.81 to 0.94; only weight-level fine-tuning reaches 0.5, where simulated and real utterances are indistinguishable.](images/fig1_substrate.png)

**[Saber Zerhoudi](mailto:szerhoudi@acm.org), Jelena Mitrović, Michael Granitzer**<br>
University of Passau &nbsp;·&nbsp; Interdisciplinary Transformation University Austria<br>
Published at **SIGIR-AP 2026**, Singapore, December 13–15, 2026 &nbsp;·&nbsp; [Read the paper](https://doi.org/10.1145/3848000.3850196)

</div>

> **What is this?** Simulating users is the standard way to compare conversational search systems without a human study for every design choice, but a simulator is only trustworthy if its behavior matches real users. AutoSim is a conversational-search user simulator that self-improves against a held-out signal from real users. This repository releases the framework, the evaluation protocol, and the trained weight-level models behind the paper.
>
> **Who is it for?** IR researchers evaluating conversational or interactive systems, and practitioners who want a user simulator whose realism is measured against real users.

**Contents:** [Abstract](#abstract) · [Highlights](#highlights) · [Citation](#citation) · [Quick Start](#quick-start) · [Reproduce the paper](#reproduce-the-paper) · [Framework](#framework) · [Models](#models) · [Human study](#human-study) · [Data](#data) · [Related work](#related-work) · [License](#license)

## Abstract

> User simulators promise to evaluate conversational search systems at a scale human studies cannot reach, but only if their behavior is realistic enough to trust. We study AutoSim, a conversational-search user simulator whose relevance judgments rank a controlled pool of retrieval systems in agreement with NIST across three TREC iKAT years (Kendall's τ of 0.66, 0.91, and 0.91; all p ≤ 0.002), an agreement that survives removing its persona. AutoSim self-improves against a held-out signal from real users, and we find that fine-tuning its weights makes it behave like real users, which in turn gives a more accurate estimate of how well a system serves them. Reaching that behavior takes more than prompting: when discriminating between real and simulated user sessions, no prompt we tried lowers a held-out real-versus-simulated classifier below AUC 0.85, whereas fine-tuning the weights drives it to 0.45, where neither the classifier nor a human reading full dialogues can tell it from a real user. That added realism is what makes the evaluation more accurate: where real users retrieve little, a prompted simulator over-estimates retrieval effectiveness by +0.136 nDCG and returns a relevant passage on 57% of the turns where real users find none, enough to change a deployment decision. Only the weight-level simulator escapes this bias. We release the framework and code.

**Keywords:** user simulation, conversational search, evaluation, self-improving agents

## Highlights

- **A validated evaluator.** AutoSim's relevance judgments rank a controlled pool of IR systems in agreement with NIST across three iKAT years (Kendall τ = 0.66 / 0.91 / 0.91, all p ≤ 0.002), and the agreement survives removing the persona.
- **Closing the gap needs weight-level training.** No prompt we tried (hand-written, tuned, or few-shot) lowers a held-out real-versus-simulated discriminator below AUC 0.85. Weight-level fine-tuning drives it to 0.45, where neither the classifier nor a human reading full dialogues can tell the simulator from a real user.
- **Realism changes the evaluation.** Where real users retrieve little, a prompted simulator over-estimates retrieval effectiveness by +0.136 nDCG and returns a relevant passage on 57% of the turns where real users find none. The weight-level simulator is not detectably biased.

## Citation

If you use AutoSim, its protocol, or its models, please cite the paper:

```bibtex
@inproceedings{Zerhoudi:2026:SIGIR-AP,
  author    = {Saber Zerhoudi and Jelena Mitrovi{\'{c}} and Michael Granitzer},
  title     = {{AutoSim}: A Self-Improving User Simulator for Conversational Search
               Evaluation},
  booktitle = {Proceedings of the 2026 Annual International {ACM} {SIGIR} Conference on
               Research and Development in Information Retrieval in the Asia Pacific
               Region ({SIGIR-AP})},
  publisher = {{ACM}},
  year      = {2026},
  doi       = {10.1145/3848000.3850196}
}
```

**ACM Reference Format:** Saber Zerhoudi, Jelena Mitrović, and Michael Granitzer. 2026. AutoSim: A Self-Improving User Simulator for Conversational Search Evaluation. In *Proceedings of the 2026 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP 2026), December 13–15, 2026, Singapore, Singapore.* ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3848000.3850196

GitHub's "Cite this repository" button in the sidebar reads the same record from [`CITATION.cff`](CITATION.cff).

## Quick Start

Requires Python 3.11+.

```bash
git clone https://github.com/searchsim-org/sigir-ap26-autosim.git
cd sigir-ap26-autosim
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
| LLM clients | `autosim/llm/` | OpenAI-compatible gateway client |
| Data | `autosim/data/` | iKAT loaders and real-user trace adapters |

## Models

Three trained **weight-level** simulators (LoRA adapters) live under [`models/`](models/): `adapter_qwen2.5-3b` (primary), `adapter_qwen2.5-7b` (scale check), and `adapter_mistral-7b` (cross-family check). Each drives the real-versus-simulated discriminator to about 0.5. See [`models/README.md`](models/README.md) for loading code and training details.

## Human study

[`human_study/`](human_study/) releases the blind 2-alternative forced-choice discrimination **protocol** ([`PROTOCOL.md`](human_study/PROTOCOL.md)), the browser collection **kit** (`index.html`, which records reaction time), and the stimulus **pairs** (`pairs_key.json`). Raw annotations are not included; run the kit with your own annotators and score with `scripts/score_human_study.py`.

## Data

[`data/`](data/) holds derived result files (per-turn behavioral analyses, per-year system-ranking outputs, and the qrels subset for the bias test), not raw corpora. TREC iKAT, ClueWeb22-B, CAsT 2020, and MS MARCO come from their official sources; see [`data/README.md`](data/README.md).

## Related work

- **[DIAL](https://arxiv.org/abs/2512.20773)** (adversarial user simulation): trains a dialogue user simulator against a realism discriminator in mental-health support. AutoSim reproduces the discriminator effect in conversational search, adds the prompt-level baseline, and measures the effect on retrieval-effectiveness estimates.
- **[User language models](https://arxiv.org/abs/2510.06552)** (Naous et al., 2025): shows trained user models are more realistic than prompted assistants. AutoSim asks what that realism does to evaluation validity.
- **[UXSim](https://doi.org/10.1145/3746252.3761640)** (CIKM 2025): the hybrid orchestration simulator AutoSim builds on.
- **[SimEval-IR](https://doi.org/10.1145/3805712.3808635)** (SIGIR 2026): a toolkit and benchmark for evaluating user simulators. It reports that a discriminator's human-likeness score predicts little of system-ranking validity, so AutoSim uses the discriminator only to certify realism.

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Saber Zerhoudi, Jelena Mitrović, and Michael Granitzer.
