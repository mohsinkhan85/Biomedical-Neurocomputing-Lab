"""Figures from the paper: class distributions, age histograms, ROC curves,
confusion matrices and learning curves."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .config import TARGET  # noqa: E402

COLORS = ["blue", "orange"]


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(df, path):
    """Figs. 2 and 4: target distribution, and stroke / smoker shares among
    heart disease cases."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    diseased = df[df[TARGET] == 1]
    panels = [
        ("Target Variable Distribution", df[TARGET],
         ["No Heart Disease", "Heart Disease"]),
        ("Distribution of Stroke Cases", diseased["Stroke"], ["No Stroke", "Stroke"]),
        ("Distribution of Smoker Cases", diseased["Smoker"], ["No Smoker", "Smoker"]),
    ]
    for ax, (title, series, labels) in zip(axes, panels):
        counts = series.value_counts().reindex([0, 1], fill_value=0)
        total = counts.sum()
        ax.bar(labels, counts.values, color=COLORS, width=1)
        for i, count in enumerate(counts.values):
            ax.text(i, count, f"{100 * count / total:.1f}%", ha="center",
                    va="bottom", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(pad=2)
    _save(fig, path)


def plot_age_distribution(df, path):
    """Figs. 3 and 5: age histogram for disease vs. no disease."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df[df[TARGET] == 0], x="Age", color=COLORS[0],
                 label="No Disease", ax=ax)
    sns.histplot(data=df[df[TARGET] == 1], x="Age", color=COLORS[1],
                 label="Disease", ax=ax)
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.legend()
    _save(fig, path)


def plot_roc(curves, path, title="Receiver Operating Characteristic (ROC) Curve"):
    """Figs. 6-9. curves: list of (label, y_true, y_score, auc_percent)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, y_true, y_score, auc in curves:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc / 100:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    _save(fig, path)


def plot_confusion_matrix(cm, path, title=None):
    """Figs. 10-12."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    if title:
        ax.set_title(title)
    _save(fig, path)


def plot_learning_curves(history, name, path):
    """Figs. 13-16: training / validation accuracy and loss per epoch."""
    epochs = range(1, len(history["accuracy"]) + 1)
    marker = "o" if len(epochs) < 3 else None  # keep very short runs visible
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_acc.plot(epochs, history["accuracy"], label="Training Accuracy", marker=marker)
    ax_loss.plot(epochs, history["loss"], label="Training Loss", marker=marker)
    if "val_accuracy" in history:
        ax_acc.plot(epochs, history["val_accuracy"], label="Validation Accuracy", marker=marker)
        ax_loss.plot(epochs, history["val_loss"], label="Validation Loss", marker=marker)
    for ax, metric in ((ax_acc, "Accuracy"), (ax_loss, "Loss")):
        ax.set_title(f"{metric} ({name})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    _save(fig, path)
