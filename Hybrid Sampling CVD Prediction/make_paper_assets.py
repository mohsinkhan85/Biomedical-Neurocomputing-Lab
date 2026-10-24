"""Build the figures and result tables of the revised manuscript
(paper/figures/ and paper/tables/).

Needs the outputs of:
    python run_experiments.py                          -> results/paper/
    python run_improved.py                             -> results/improved/
    python run_improved.py --no-baseline --cv --out results/improved_cv
"""

import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "paper", "figures")
TABLES = os.path.join(HERE, "paper", "tables")

# Categorical slots 1 and 2 of the reference palette; text stays in ink colours.
SERIES = {"Paper setup": "#2a78d6", "Improved": "#eb6834"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODELS = ["DT", "SVM", "XGBoost", "CNN", "NN", "MLP"]
METRICS = [("Accuracy (%)", "Accuracy"), ("Recall (%)", "Recall"),
           ("AUC (%)", "AUC")]


def comparison_table():
    paper = pd.read_csv(os.path.join(RESULTS, "improved", "comparison.csv"))
    improved = pd.read_csv(os.path.join(RESULTS, "improved_cv", "comparison.csv"))
    paper = paper[paper["Setup"] == "paper"].set_index("Model")
    improved = improved.set_index("Model")
    return paper, improved


def plot_comparison(path):
    paper, improved = comparison_table()
    x = np.arange(len(MODELS))
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (column, title) in zip(axes, METRICS):
        for offset, (label, frame) in zip((-width / 2, width / 2),
                                          (("Paper setup", paper),
                                           ("Improved", improved))):
            values = frame.loc[MODELS, column].to_numpy()
            bars = ax.bar(x + offset, values, width * 0.94, label=label,
                          color=SERIES[label], edgecolor="white", linewidth=1)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 1,
                        f"{value:.0f}", ha="center", va="bottom",
                        fontsize=7.5, color=INK_MUTED)
        ax.set_title(title, fontsize=12, color=INK)
        ax.set_xticks(x, MODELS, fontsize=9, color=INK)
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_MUTED)
        ax.tick_params(axis="y", colors=INK_MUTED, length=0)
    axes[0].set_ylabel("Percent (real, unseen test set)", color=INK_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _fmt(value):
    return f"{value:.2f}"


def write_test_table(path):
    """Rows of the Protocol B table: paper setup vs. improved, per model."""
    paper, improved = comparison_table()
    lines = []
    for model in MODELS + ["LogReg"]:
        cells = [model.replace("LogReg", "LR")]
        for frame in (paper, improved):
            if model in frame.index:
                row = frame.loc[model]
                cells += [_fmt(row[c]) for c in ("Accuracy (%)", "Precision (%)",
                                                 "Recall (%)", "F1 Score (%)",
                                                 "AUC (%)")]
            else:
                cells += ["--"] * 5
        lines.append(" & ".join(cells) + r" \\")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_cv_table(path):
    """Mean and standard deviation of the 5-fold CV of the improved models."""
    cv = pd.read_csv(os.path.join(RESULTS, "improved_cv", "cv_improved.csv"))
    columns = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)",
               "AUC (%)"]
    lines = []
    for model in ["DT", "SVM", "XGBoost", "LogReg", "CNN", "NN", "MLP"]:
        rows = cv[cv["Model"] == model].set_index("Fold")
        cells = [model.replace("LogReg", "LR")]
        cells += [f"{rows.loc['Mean', c]:.2f} $\\pm$ {rows.loc['Std', c]:.2f}"
                  for c in columns]
        lines.append(" & ".join(cells) + r" \\")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    write_test_table(os.path.join(TABLES, "protocol_b_test.tex"))
    if os.path.exists(os.path.join(RESULTS, "improved_cv", "cv_improved.csv")):
        write_cv_table(os.path.join(TABLES, "cv_improved.tex"))
    plot_comparison(os.path.join(FIGURES, "comparison_paper_vs_improved.png"))
    copies = {
        "eda/fig2_distribution_before_sampling.png": "paper",
        "eda/fig3_age_before_sampling.png": "paper",
        "eda/fig4_distribution_after_sampling.png": "paper",
        "eda/fig5_age_after_sampling.png": "paper",
        "figures/roc_hs_dl.png": "paper",
        "figures/roc_hs_ml.png": "paper",
        "roc_improved.png": "improved_cv",
        "confusion_matrix_XGBoost.png": "improved_cv",
        "confusion_matrix_LogReg.png": "improved_cv",
        "confusion_matrix_MLP.png": "improved_cv",
        "learning_curve_MLP.png": "improved_cv",
        "learning_curve_NN.png": "improved_cv",
    }
    for name, run in copies.items():
        target = os.path.basename(name)
        if run == "improved_cv" and not target.startswith("roc"):
            target = "improved_" + target
        shutil.copy(os.path.join(RESULTS, run, name), os.path.join(FIGURES, target))
    print(f"Figures written to {FIGURES}, tables to {TABLES}")


if __name__ == "__main__":
    main()
