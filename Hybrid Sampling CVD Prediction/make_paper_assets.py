"""Build the figures and result tables of the revised manuscript
(paper/figures/ and paper/tables/).

Needs the outputs of:
    python sampling_analysis.py                        -> results/sampling/
    python run_improved.py                             -> results/improved/
    python run_improved.py --no-baseline --cv --out results/improved_cv
"""

import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "paper", "figures")
TABLES = os.path.join(HERE, "paper", "tables")

# Categorical slots 1-3 of the reference palette; text stays in ink colours.
METRIC_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
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


def plot_improved(path):
    """Accuracy, recall and AUC of the improved models on the real test set."""
    _, improved = comparison_table()
    models = MODELS + ["LogReg"]
    x = np.arange(len(models))
    width = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, ((column, label), color) in enumerate(zip(METRICS, METRIC_COLORS)):
        values = improved.loc[models, column].to_numpy()
        bars = ax.bar(x + (i - 1) * width, values, width * 0.94, label=label,
                      color=color, edgecolor="white", linewidth=1)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1,
                    f"{value:.0f}", ha="center", va="bottom", fontsize=7.5,
                    color=INK_MUTED)
    ax.set_xticks(x, [m.replace("LogReg", "LR") for m in models], fontsize=10,
                  color=INK)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent (real, unseen test set)", color=INK_MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(axis="y", colors=INK_MUTED, length=0)
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=10,
              bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(path, run_dir=os.path.join(RESULTS, "improved_cv"),
                         models=("CNN", "NN", "MLP")):
    """Training and validation accuracy (top) and loss (bottom) per epoch of
    the improved DL models; the dashed line marks the epoch with the lowest
    validation loss, whose weights early stopping restores."""
    fig, axes = plt.subplots(2, len(models), figsize=(4 * len(models), 6.2),
                             squeeze=False)
    for col, name in enumerate(models):
        history = pd.read_csv(os.path.join(run_dir, f"learning_curve_{name}.csv"))
        epochs = history["epoch"] + 1
        best = epochs[history["val_loss"].idxmin()]
        for row, (metric, label) in enumerate((("accuracy", "Accuracy"),
                                               ("loss", "Loss"))):
            ax = axes[row, col]
            ax.plot(epochs, history[metric], color=METRIC_COLORS[0],
                    linewidth=2, label="Training")
            ax.plot(epochs, history[f"val_{metric}"], color=METRIC_COLORS[1],
                    linewidth=2, label="Validation")
            ax.axvline(best, color=INK_MUTED, linestyle="--", linewidth=1,
                       label="Best epoch (weights restored)")
            if row == 0:
                ax.set_title(name, fontsize=12, color=INK)
            if col == 0:
                ax.set_ylabel(label, color=INK_MUTED)
            if row == 1:
                ax.set_xlabel("Epoch", color=INK_MUTED)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(INK_MUTED)
            ax.tick_params(colors=INK_MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.03))
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


def write_literature_rows(path, frame, dataset, models=("XGBoost", "LogReg", "MLP"),
                          std=None):
    """'Proposed' rows of the literature table (Table 8) for one dataset."""
    names = {"XGBoost": "XGBoost", "LogReg": "Logistic regression", "MLP": "MLP"}
    lines = []
    for model in models:
        row = frame.loc[model]
        cells = ["Proposed (improved)", dataset, names[model]]
        for column in ("Accuracy (%)", "Precision (%)", "Recall (%)",
                       "F1 Score (%)", "AUC (%)"):
            cell = _fmt(row[column])
            if std is not None:
                cell += f" $\\pm$ {std.loc[model, column]:.1f}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    write_test_table(os.path.join(TABLES, "test_results.tex"))
    if os.path.exists(os.path.join(RESULTS, "improved_cv", "cv_improved.csv")):
        write_cv_table(os.path.join(TABLES, "cv_improved.tex"))
    _, improved = comparison_table()
    write_literature_rows(os.path.join(TABLES, "literature_brfss.tex"),
                          improved, "BRFSS (229,781)")
    cleveland = os.path.join(RESULTS, "cleveland", "cleveland_summary.csv")
    if os.path.exists(cleveland):
        summary = pd.read_csv(cleveland).set_index("Model")
        means = summary[[c for c in summary.columns if not c.endswith(" std")]]
        stds = summary[[c for c in summary.columns if c.endswith(" std")]]
        stds.columns = [c[:-4] for c in stds.columns]
        write_literature_rows(os.path.join(TABLES, "literature_cleveland.tex"),
                              means, "Cleveland (297)", std=stds)
    plot_improved(os.path.join(FIGURES, "improved_results.png"))
    plot_learning_curves(os.path.join(FIGURES, "improved_learning_curves.png"))
    copies = {
        "distribution_before_sampling.png": "sampling",
        "age_before_sampling.png": "sampling",
        "distribution_after_sampling.png": "sampling",
        "age_after_sampling.png": "sampling",
        "roc_improved.png": "improved_cv",
        "confusion_matrix_XGBoost.png": "improved_cv",
        "confusion_matrix_LogReg.png": "improved_cv",
        "confusion_matrix_MLP.png": "improved_cv",
    }
    for name, run in copies.items():
        target = os.path.basename(name)
        if run == "improved_cv" and not target.startswith("roc"):
            target = "improved_" + target
        shutil.copy(os.path.join(RESULTS, run, name), os.path.join(FIGURES, target))
    print(f"Figures written to {FIGURES}, tables to {TABLES}")


if __name__ == "__main__":
    main()
