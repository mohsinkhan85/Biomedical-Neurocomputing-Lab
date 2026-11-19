"""Package J figure: conformal intervals for the combined recalibration (phys3 + PanEcho + EchoPrime), both sites."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
INK2, GRID = "#555555", "#e6e6e6"
cam, ped, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
spec = [("plain", "Split conformal", "#2a78d6", "o"), ("mondrian", "Mondrian (EF <50 / ≥50)", "#eb6834", "s"),
        ("norm", "Error-profile scaled (ours)", "#1baf7a", "^"), ("asym", "Scaled + asymmetric (ours)", "#eda100", "D")]
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.2), constrained_layout=True)
for ax, f, title in ((axs[0], cam, "(a) CAMUS: coverage, reduced EF"), (axs[1], ped, "(b) Pediatric: coverage, reduced EF")):
    d = pd.read_csv(f); g = d[(d.config == "phys3+FM") & (d.selection == "random") & (d.k >= 20)].set_index("k")
    for key, name, col, mk in spec:
        ax.plot(g.index, g[f"cov_{key}_reduced"], color=col, marker=mk, ms=4, mec="white", mew=0.6, label=name)
    ax.axhline(0.9, color=INK2, lw=0.9, ls=(0, (4, 2)))
    ax.set_title(title, loc="left"); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Coverage of 90% interval")
d = pd.read_csv(ped); g = d[(d.config == "phys3+FM") & (d.selection == "random") & (d.k >= 20)].set_index("k")
for key, name, col, mk in spec:
    axs[2].plot(g.index, g[f"width_{key}"], color=col, marker=mk, ms=4, mec="white", mew=0.6)
axs[2].set_title("(c) Pediatric: interval width", loc="left"); axs[2].set_ylabel("Mean width (EF points)")
for ax in axs:
    ax.set_xscale("log"); ks = [20, 50, 100, 250, 1000] if ax is not axs[0] else [20, 50, 100, 250]
    ax.set_xticks(ks); ax.set_xticklabels(map(str, ks)); ax.minorticks_off()
    ax.set_xlabel("Labelled patients (k)"); ax.grid(axis="y", color=GRID, lw=0.6)
axs[0].text(250, 0.905, "target 0.90", ha="right", va="bottom", fontsize=6.2, color=INK2)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=4, frameon=False)
for ext in ("pdf", "png"):
    fig.savefig(out / f"fig_uncertainty.{ext}", dpi=300)
