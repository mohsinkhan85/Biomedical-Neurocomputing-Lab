"""Uncertainty, risk factors and calibration (Sections 5.4-5.6).

1. Bootstrap 95% confidence intervals for AUC, accuracy and recall on the
   real test set, and paired differences between models (2,000 resamples,
   stratified by class so that every resample keeps 3,000 records per class).
2. Odds ratios with 95% Wald confidence intervals from an unweighted logistic
   regression on the real training set (all 21 features, per unit of each
   variable).
3. Calibration of the improved models on the out-of-fold predictions of the
   five-fold cross-validation (natural prevalence): Brier score, mean
   predicted risk and calibration curves, before and after correcting the
   probabilities for the class weights used in training.

Needs the outputs of:
    python run_improved.py                                  -> results/improved/
    python run_improved.py --no-baseline --cv --out results/improved_cv
    python run_ablation.py                                  -> results/ablation/

Example:
    python run_robustness.py
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (accuracy_score, brier_score_loss,  # noqa: E402
                             recall_score, roc_auc_score)

from hscvd import config  # noqa: E402
from hscvd.data import load_dataset, split_xy  # noqa: E402
from hscvd.sampling import build_splits  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DEFAULT_DATA = os.path.join(HERE, "data", "heart_disease_health_indicators_BRFSS2015.csv")
MODELS = ["DT", "SVM", "XGBoost", "LogReg", "CNN", "NN", "MLP"]
N_BOOT = 2000
# Models whose scores are probabilities; SVM returns decision values.
PROBABILISTIC = ["DT", "XGBoost", "LogReg", "CNN", "NN", "MLP"]
CALIBRATION_FIGURE_MODELS = ["XGBoost", "LogReg", "MLP"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", default=os.path.join(RESULTS, "robustness"))
    return p.parse_args()


# ------------------------------------------------------------ 1. bootstrap

def bootstrap_indices(y, n_boot=N_BOOT, seed=config.RANDOM_STATE):
    """Stratified bootstrap: resample positives and negatives separately."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def metric_samples(y, score, pred, indices):
    auc, acc, rec = [], [], []
    for idx in indices:
        auc.append(roc_auc_score(y[idx], score[idx]))
        acc.append(accuracy_score(y[idx], pred[idx]))
        rec.append(recall_score(y[idx], pred[idx]))
    return {"AUC": np.array(auc), "Accuracy": np.array(acc), "Recall": np.array(rec)}


def ci(samples):
    low, high = np.percentile(samples, [2.5, 97.5])
    return 100 * low, 100 * high


def bootstrap_analysis(out):
    base = pd.read_csv(os.path.join(RESULTS, "improved", "test_predictions.csv"))
    impr = pd.read_csv(os.path.join(RESULTS, "improved_cv", "test_predictions.csv"))
    assert (base["y_true"].to_numpy() == impr["y_true"].to_numpy()).all()
    y = impr["y_true"].to_numpy()
    indices = list(bootstrap_indices(y))

    samples, rows = {}, []
    for setup, frame in (("Baseline HS", base), ("Improved", impr)):
        prefix = "baseline" if setup == "Baseline HS" else "improved"
        for model in MODELS:
            col = f"{prefix}_{model}_score"
            if col not in frame:
                continue
            score = frame[col].to_numpy()
            pred = frame[f"{prefix}_{model}_pred"].to_numpy()
            s = metric_samples(y, score, pred, indices)
            samples[(setup, model)] = s
            row = {"Setup": setup, "Model": model}
            for name, fn in (("AUC", roc_auc_score), ("Accuracy", accuracy_score),
                             ("Recall", recall_score)):
                point = 100 * (fn(y, score) if name == "AUC" else fn(y, pred))
                low, high = ci(s[name])
                row.update({name: point, f"{name} CI low": low,
                            f"{name} CI high": high})
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "bootstrap_ci.csv"), index=False)

    diffs = []
    for model in MODELS:
        if ("Baseline HS", model) not in samples:
            continue
        for name in ("AUC", "Recall"):
            d = samples[("Improved", model)][name] - samples[("Baseline HS", model)][name]
            low, high = ci(d)
            diffs.append({"Comparison": f"Improved - baseline {model}", "Metric": name,
                          "Difference": 100 * d.mean(), "CI low": low, "CI high": high})
    best = samples[("Improved", "XGBoost")]["AUC"]
    for model in MODELS:
        if model == "XGBoost":
            continue
        d = best - samples[("Improved", model)]["AUC"]
        low, high = ci(d)
        diffs.append({"Comparison": f"Improved XGBoost - improved {model}",
                      "Metric": "AUC", "Difference": 100 * d.mean(),
                      "CI low": low, "CI high": high})

    abl = pd.read_csv(os.path.join(RESULTS, "ablation", "ablation_predictions.csv"))
    yab = abl["y_true"].to_numpy()
    assert (yab == y).all()
    pairs = [("XGBoost | 9 features, no resampling", "XGBoost | 9 features, HS (baseline)"),
             ("XGBoost | 21 features, no resampling", "XGBoost | 9 features, no resampling")]
    for a, b in pairs:
        da = np.array([roc_auc_score(y[i], abl[a].to_numpy()[i]) for i in indices])
        db = np.array([roc_auc_score(y[i], abl[b].to_numpy()[i]) for i in indices])
        low, high = ci(da - db)
        diffs.append({"Comparison": f"{a.split(' | ')[1]} - {b.split(' | ')[1]} (XGBoost)",
                      "Metric": "AUC", "Difference": 100 * (da - db).mean(),
                      "CI low": low, "CI high": high})
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(os.path.join(out, "bootstrap_differences.csv"), index=False)
    return table, diffs


# ---------------------------------------------------------- 2. odds ratios

def odds_ratios(data_path, out):
    train = build_splits(load_dataset(data_path), "none", protocol="strict")["train"]
    X, y = split_xy(train)
    fit = sm.Logit(y, sm.add_constant(X.astype(float))).fit(disp=0)
    conf = fit.conf_int()
    table = pd.DataFrame({
        "Variable": fit.params.index,
        "Odds ratio": np.exp(fit.params.to_numpy()),
        "CI low": np.exp(conf[0].to_numpy()),
        "CI high": np.exp(conf[1].to_numpy()),
        "p-value": fit.pvalues.to_numpy(),
    })
    table = table[table["Variable"] != "const"]
    table = table.sort_values("Odds ratio", ascending=False).reset_index(drop=True)
    table.to_csv(os.path.join(out, "odds_ratios.csv"), index=False)
    return table


# ----------------------------------------------------------- 3. calibration

def prior_correct(p, prevalence):
    """Undo balanced class weighting: the model's odds correspond to a
    prevalence of 50%, so multiply them by the training odds of CVD."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    odds = p / (1 - p) * prevalence / (1 - prevalence)
    return odds / (1 + odds)


def calibration_analysis(out):
    oof = pd.read_csv(os.path.join(RESULTS, "improved_cv", "cv_predictions.csv"))
    y = oof["y_true"].to_numpy()
    prevalence = y.mean()
    rows, curves = [], {}
    for model in PROBABILISTIC:
        raw = oof[f"{model}_score"].to_numpy()
        corrected = prior_correct(raw, prevalence)
        rows.append({"Model": model,
                     "Observed prevalence (%)": 100 * prevalence,
                     "Mean predicted, raw (%)": 100 * raw.mean(),
                     "Mean predicted, corrected (%)": 100 * corrected.mean(),
                     "Brier, raw": brier_score_loss(y, raw),
                     "Brier, corrected": brier_score_loss(y, corrected),
                     "Brier, prevalence only": brier_score_loss(
                         y, np.full_like(raw, prevalence))})
        curves[model] = (calibration_curve(y, raw, n_bins=10, strategy="quantile"),
                         calibration_curve(y, corrected, n_bins=10, strategy="quantile"))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "calibration.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, which, title in ((axes[0], 0, "Before correction"),
                             (axes[1], 1, "After prior correction")):
        ax.plot([0, 1], [0, 1], linestyle="--", color=INK_MUTED, linewidth=1,
                label="Perfect calibration")
        for color, model in zip(COLORS, CALIBRATION_FIGURE_MODELS):
            frac_pos, mean_pred = curves[model][which]
            ax.plot(mean_pred, frac_pos, marker="o", markersize=5, linewidth=2,
                    color=color, label=model.replace("LogReg", "LR"))
        ax.set_title(title, fontsize=12, color=INK)
        ax.set_xlabel("Mean predicted probability", color=INK_MUTED)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_MUTED)
    axes[0].set_ylabel("Observed fraction with CVD", color=INK_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "calibration.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return table


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    table, diffs = bootstrap_analysis(args.out)
    print("== Bootstrap 95% CIs (test set)\n" + table.to_string(index=False, float_format=fmt))
    print("\n== Paired differences\n" + diffs.to_string(index=False, float_format=fmt))
    print("\n== Odds ratios\n" + odds_ratios(args.data, args.out).to_string(
        index=False, float_format=fmt))
    print("\n== Calibration (out-of-fold, natural prevalence)\n" + calibration_analysis(
        args.out).to_string(index=False, float_format=fmt))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
