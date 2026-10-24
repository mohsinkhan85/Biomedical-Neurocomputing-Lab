"""Leakage analysis (Table 4) and the statistics of hybrid sampling quoted in
Sections 3.4 and 5.2.

- How SMOTEENN + ROS change the data: records created and removed, the
  average duplication of positive records, and the share of stroke cases and
  the mean age before and after sampling.
- Table 4: the share of test records that have an identical record (all nine
  features and the label) in the training set, under Protocol A and B.

Example:
    python leakage_analysis.py
"""

import argparse
import os

import pandas as pd
from imblearn.combine import SMOTEENN

from hscvd import config
from hscvd.data import load_dataset, select_features, split_xy
from hscvd.sampling import build_splits, resample_df

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "heart_disease_health_indicators_BRFSS2015.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", default=os.path.join(HERE, "results", "leakage"))
    return p.parse_args()


def overlap(train, test):
    """Percentage of test records with an identical record in train, overall
    and per class."""
    seen = set(map(tuple, train.to_numpy()))
    hit = pd.Series([tuple(r) in seen for r in test.to_numpy()], index=test.index)
    positive = test[config.TARGET] == 1
    return {"All test records (%)": 100 * hit.mean(),
            "Positive (%)": 100 * hit[positive].mean(),
            "Negative (%)": 100 * hit[~positive].mean()}


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
    X, y = split_xy(df)
    n_neg, n_pos = (y == 0).sum(), (y == 1).sum()

    X_se, y_se = SMOTEENN(random_state=config.RANDOM_STATE).fit_resample(X, y)
    kept_neg, kept_pos = (y_se == 0).sum(), (y_se == 1).sum()
    resampled = resample_df(df, "hs")
    stats = {
        "Real records (negative / positive)": f"{n_neg} / {n_pos}",
        "Synthetic positives created by SMOTE": n_neg - n_pos,
        "Records after SMOTEENN (negative / positive)": f"{kept_neg} / {kept_pos}",
        "Real negatives removed by ENN": n_neg - kept_neg,
        "Real negatives removed by ENN (%)": round(100 * (1 - kept_neg / n_neg), 1),
        "Average copies per positive after ROS": round(kept_neg / kept_pos, 2),
        "Records after HS": len(resampled),
    }
    before, after = profile(df), profile(resampled)
    for key in before:
        stats[f"{key}: before / after HS"] = f"{before[key]:.2f} / {after[key]:.2f}"

    print("== Hybrid sampling of the full dataset")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    pd.Series(stats).to_csv(os.path.join(args.out, "sampling_stats.csv"),
                            header=["value"])

    rows = []
    for protocol, label in (("paper", "A: resample before split"),
                            ("strict", "B: split before sampling")):
        splits = build_splits(df, "hs", protocol=protocol)
        rows.append({"Protocol": label, **overlap(splits["train"], splits["test"])})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "table4_overlap.csv"), index=False)
    print("\n== Table 4: test records with an identical record in the training set")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main()
