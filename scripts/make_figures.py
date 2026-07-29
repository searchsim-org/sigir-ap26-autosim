"""Generate the paper's data figures in ONE clean style matching fig1_substrate:
blue/red palette, thick lines, despined, no redundant in-figure title (captions carry the
message), fonts sized for a tired reviewer. Writes straight into docs/paper-v3/figs/.
  Fig 2 = fig:validity  (14-system sim-to-real scatter)
  Fig 3 = fig:selfimprove (behavioral self-improvement over 12 seeds)
Run: .venv/bin/python scripts/make_figures.py
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

OUT = Path("docs/paper-v3/figs")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("data/ikat/2023")
BLUE, RED, GREY = "#2c7fb8", "#c0392b", "#7f7f7f"   # match fig1_substrate

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# --- Fig 2 (fig:validity): 14-system validity scatter (12 constructed + 2 real official*) ---
constructed = [
    ("rrf(resolved)", 0.3432, 0.6318), ("splade(resolved)", 0.2952, 0.6608),
    ("bm25(resolved)", 0.2759, 0.5531), ("rrf+rerank", 0.2284, 0.5935),
    ("bm25+rerank", 0.1997, 0.5060), ("splade+rerank", 0.1987, 0.6040),
    ("splade(ptkb)", 0.1481, 0.5528), ("rrf(ptkb)", 0.1334, 0.5057),
    ("splade(ptkb)+rerank", 0.1053, 0.5463), ("splade(raw)", 0.0962, 0.3344),
    ("bm25(raw)", 0.0946, 0.2767), ("bm25(ptkb)", 0.0576, 0.2886),
]
official = [("bm25_rm3(manual)", 0.2891, 0.5051), ("bm25_rm3(auto)", 0.0332, 0.1545)]
allp = constructed + official
tau, p = kendalltau([s[1] for s in allp], [s[2] for s in allp])

fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.grid(color="0.90", linewidth=0.7, zorder=0)
ax.scatter([s[1] for s in constructed], [s[2] for s in constructed], s=46, color=BLUE,
           edgecolor="white", linewidth=0.6, zorder=3, label="constructed pool")
ax.scatter([s[1] for s in official], [s[2] for s in official], s=190, marker="*", color=RED,
           edgecolor="white", linewidth=0.6, zorder=4, label="real official runs")
ax.set_xlabel("Condensed nDCG@5 (NIST qrels)")
ax.set_ylabel("AutoSim satisfaction")
ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="0.85", handletextpad=0.3)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig2_validity.pdf")
plt.close(fig)

# --- Fig 3 (fig:selfimprove): behavioral self-improvement over seeds (mean +/- SD) ---
ms = json.loads((DATA / "behavior_multiseed.json").read_text())
res = ms["results"]
cyc = [0, 1, 2]


def band(arm):
    series = [r[arm] for r in res]
    mean = [statistics.mean(x[c] for x in series) for c in cyc]
    sd = [statistics.stdev(x[c] for x in series) if len(series) > 1 else 0.0 for c in cyc]
    return mean, sd


fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.grid(color="0.90", linewidth=0.7, zorder=0)
for arm, lab, col, mk in [("full", "full (gated)", BLUE, "o"), ("nocal", "no-calibration", RED, "s")]:
    m, sd = band(arm)
    ax.plot(cyc, m, mk + "-", color=col, lw=2.6, ms=8, markeredgecolor="white",
            markeredgewidth=0.6, label=lab, zorder=3)
    ax.fill_between(cyc, [a - b for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)],
                    color=col, alpha=0.12, zorder=1)
static_mean = sum(r["static"] for r in res) / len(res)
ax.axhline(static_mean, ls=(0, (1, 1.2)), color=GREY, lw=2.0, label="static (no evolution)", zorder=2)
ax.set_xlabel("self-improvement cycle")
ax.set_ylabel("held-out behavioral CSG")
ax.set_xticks(cyc)
ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="0.85", handletextpad=0.4)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig3_selfimprove.pdf")
plt.close(fig)

print(f"wrote {OUT}/fig2_validity.pdf (tau={tau:.3f}) + fig3_selfimprove.pdf, {len(res)} seeds")
