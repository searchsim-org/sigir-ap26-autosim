# Weight-level discriminator results — provenance manifest

Provenance for the RQ2 substrate result (Table `tab:substrate`, §5.2). Each real-versus-simulated
discriminator AUC below is the recorded output of the named script; `0.5` = indistinguishable from
real, higher = more separable. Held-out probe: the 82 iKAT turns disjoint from the SFT split, 2
samples per turn against matched real utterances, five-fold with folds **grouped by turn**.

| Base generator | Self-improvement | Discriminator | AUC (default → weight) | Script |
|---|---|---|---|---|
| Qwen2.5-3B | warm SFT + 4 adversarial rounds | emb-LR (in-loop, MiniLM) | 0.88 → 0.45 (rounds 0.42–0.50) | `v2/scripts/exp_ikat_weight.py` |
| Qwen2.5-3B | warm SFT + 4 adversarial rounds | XLM-RoBERTa audit (grouped) | 0.94 → 0.51 ± 0.06 | `v2/scripts/exp_ikat_strongdisc.py` |
| Qwen2.5-3B | (control) politeness perturbation | XLM-RoBERTa audit | 0.99 (power check) | `v2/scripts/exp_ikat_strongdisc.py` |
| Qwen2.5-3B | few-shot prompt (8 real exemplars) | XLM-RoBERTa audit | 0.94 | `v2/scripts/exp_ikat_fewshot.py` |
| Qwen2.5-7B | warm SFT + 4 adversarial rounds | XLM-RoBERTa audit | 1.00 → 0.50 | `v2/scripts/exp_ikat_strongdisc.py` |
| Mistral-7B-Instruct-v0.3 | warm SFT + 4 adversarial rounds | XLM-RoBERTa audit | 0.97 → 0.40 | `v2/scripts/exp_ikat_strongdisc.py` |

**Fixed config (all rows):** LoRA r=16, α=32, q/v projections, dropout 0.05; warm-start 3 epochs;
4 adversarial rounds × 2 epochs; AdamW, lr 1e-4, batch 8; sampling temp 1.0 (candidates) / 0.7 with
top-p 0.95 (final), ≤48 new tokens. XLM-RoBERTa-base audit: five-fold grouped-by-turn, 4 epochs,
lr 2e-5, batch 16, max-len 64.

**Committed output files (synced from the training cluster 2026-07-14).** Every number above is now
backed by a committed result file, and each has been checked against the paper:

| Committed file | Paper number reconciled |
|---|---|
| `weight_runs/exp_ikat_weight.json` | octen_auc trajectory: base_default 0.878, warm_sft 0.446, adversarial 0.415–0.497 → paper "0.88 → 0.45, holds 0.42–0.50" |
| `weight_runs/exp_ikat_audit.json` | XLM-R grouped default 0.943 / weight 0.511; power (perturbed) 0.992; distinct-2 weight 0.539 vs real 0.84 → paper "0.94 / 0.51 / 0.99 / 0.54 vs 0.84" |
| `weight_runs/exp_ikat_strongdisc_7b.json` | 0.999 / 0.498 → paper "1.00 / 0.50" |
| `weight_runs/exp_ikat_strongdisc_mistral.json` | 0.974 / 0.397 → paper "0.97 / 0.40" |
| `weight_runs/exp_ikat_fewshot.json` | emb 0.806 / XLM-R 0.94 → paper "0.81 / 0.94" |
| `v2/runs/ikat_weight_adapter{,_7b,_mistral}/` | trained LoRA adapters; `adapter_config.json` verified r=16, α=32, dropout 0.05, q/v proj (matches Reproducibility §) |

All 82-turn held-out AUCs and the LoRA config now match the paper from committed files, at the same
coverage as RQ1 (`data/ikat/*/tau_ci.txt`) and RQ3 (`data/ikat/rv_validity_pooled.json`,
`data/ikat/consequential_bias.json`). For public release, the three adapters (14/19/26 MB) are best
deposited to a model host (HuggingFace/Zenodo) and linked, rather than committed to the paper repo.
