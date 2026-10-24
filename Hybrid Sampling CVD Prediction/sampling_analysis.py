"""How hybrid sampling changes the training data (Sections 3.3 and 5.1).

Hybrid sampling (SMOTEENN followed by ROS) is applied to the real training
set of Protocol B, as in the baseline HS pipeline. The script reports the
records created, removed and duplicated, compares the training data before
and after sampling (share of stroke cases and mean age), and draws Figs. 2
and 3 (full dataset before sampling, training set after sampling).

Example:
    python sampling_analysis.py
"""

import argparse
import os

import pandas as pd
from imblearn.combine import SMOTEENN

from hscvd import config, plots
from hscvd.data import load_dataset, select_features, split_xy
from hscvd.sampling import build_splits, resample_df

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "heart_disease_health_indicators_BRFSS2015.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", default=os.path.join(HERE, "results", "sampling"))
    return p.parse_args()


def profile(df):
    cases = df[df[config.TARGET] == 1]
    return {"Stroke among CVD cases (%)": 100 * cases["Stroke"].mean(),
            "Smoker among CVD cases (%)": 100 * cases["Smoker"].mean(),
            "Mean age category, CVD": cases["Age"].mean(),
            "Mean age category, non-CVD": df[df[config.TARGET] == 0]["Age"].mean()}


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = select_features(load_dataset(args.data))
    train = build_splits(df, "none", protocol="strict")["train"]
    X, y = split_xy(train)
    n_neg, n_pos = (y == 0).sum(), (y == 1).sum()

    _, y_se = SMOTEENN(random_state=config.RANDOM_STATE).fit_resample(X, y)
    kept_neg, kept_pos = (y_se == 0).sum(), (y_se == 1).sum()
    resampled = resample_df(train, "hs")
    stats = {
        "Real training records (negative / positive)": f"{n_neg} / {n_pos}",
        "Synthetic positives created by SMOTE": n_neg - n_pos,
        "Records after SMOTEENN (negative / positive)": f"{kept_neg} / {kept_pos}",
        "Real negatives removed by ENN": n_neg - kept_neg,
        "Real negatives removed by ENN (%)": round(100 * (1 - kept_neg / n_neg), 1),
        "Positives removed by ENN (%)": round(100 * (1 - kept_pos / n_neg), 1),
        "Average copies per positive after ROS": round(kept_neg / kept_pos, 2),
        "Training records after HS": len(resampled),
    }
    before, after = profile(train), profile(resampled)
    for key in before:
        stats[f"{key}: before / after HS"] = f"{before[key]:.2f} / {after[key]:.2f}"

    print("== Hybrid sampling of the training set (Protocol B)")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    pd.Series(stats).to_csv(os.path.join(args.out, "sampling_stats.csv"),
                            header=["value"])
    plots.plot_distributions(df, os.path.join(args.out, "distribution_before_sampling.png"))
    plots.plot_age_distribution(df, os.path.join(args.out, "age_before_sampling.png"))
    plots.plot_distributions(resampled, os.path.join(
        args.out, "distribution_after_sampling.png"))
    plots.plot_age_distribution(resampled, os.path.join(
        args.out, "age_after_sampling.png"))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
