"""Evaluate the improved, leakage-free pipeline on the UCI Cleveland heart
disease data, the dataset used by most of the studies the paper compares
against (Table 8).

Protocol: the 6 patients with missing values are removed (297 remain), the
categorical variables are one-hot encoded, and the models are evaluated with
repeated stratified 5-fold cross-validation. In every fold, 20% of the
training portion is held out for early stopping and threshold tuning; the
test fold is never used for fitting, scaling or tuning.

Outputs (under --out): cleveland_cv.csv (all folds) and cleveland_summary.csv
(mean and standard deviation per model).

Example:
    python run_cleveland.py
    python run_cleveland.py --models LogReg XGBoost --repeats 2   # quick run
"""

import argparse
import os

import pandas as pd

from hscvd import config
from hscvd.evaluation import cross_validate, format_table
from hscvd.models import build_improved_model
from run_experiments import set_seeds

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "cleveland", "heart_disease_cleveland.tsv.gz")
MODELS = ["DT", "SVM", "XGBoost", "LogReg", "CNN", "NN", "MLP"]
CATEGORICAL = ["chest_pain_type", "resting_ecg_results", "exercise_st_slope",
               "thallium_stress_test"]
METRICS = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "AUC (%)"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", default=os.path.join(HERE, "results", "cleveland"))
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--repeats", type=int, default=5,
                   help="number of repetitions of 5-fold cross-validation")
    p.add_argument("--epochs", type=int, default=100,
                   help="maximum epochs for the DL models (early stopping applies)")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def load_cleveland(path):
    df = pd.read_csv(path, sep="\t").dropna().reset_index(drop=True)
    df = pd.get_dummies(df, columns=CATEGORICAL, dtype=int)
    return df.rename(columns={"target": config.TARGET}).astype(float).astype(
        {config.TARGET: int})


def main():
    args = parse_args()
    set_seeds(config.RANDOM_STATE)
    os.makedirs(args.out, exist_ok=True)

    df = load_cleveland(args.data)
    counts = df[config.TARGET].value_counts()
    print(f"Cleveland: {len(df)} patients ({counts[1]} with heart disease), "
          f"{df.shape[1] - 1} features after one-hot encoding")

    folds = []
    for name in args.models:
        for repeat in range(args.repeats):
            print(f"-- {name}, repeat {repeat + 1}/{args.repeats}")
            table = cross_validate(
                lambda y, n=name: build_improved_model(
                    n, y, epochs=args.epochs, batch_size=args.batch_size),
                df, tune=True, random_state=config.RANDOM_STATE + repeat)
            table = table[table["Fold"].str.startswith("Fold")]
            folds.append(table.assign(Model=name, Repeat=repeat + 1))

    folds = pd.concat(folds, ignore_index=True)
    folds.to_csv(os.path.join(args.out, "cleveland_cv.csv"), index=False)
    grouped = folds.groupby("Model", sort=False)[METRICS]
    summary = grouped.mean().join(grouped.std(), rsuffix=" std").reset_index()
    summary.to_csv(os.path.join(args.out, "cleveland_summary.csv"), index=False)
    print("\n" + format_table(summary))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
