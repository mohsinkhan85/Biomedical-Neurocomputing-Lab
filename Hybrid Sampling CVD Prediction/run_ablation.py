"""Ablation study (Table 6): vary one design choice at a time for XGBoost and
the decision tree under Protocol B, on the same real test set as
run_improved.py.

For each configuration it reports the test AUC and the test accuracy at the
default threshold of 0.5 and at the threshold tuned on the validation set
(Youden's J).

Example:
    python run_ablation.py
"""

import argparse
import os

import pandas as pd

from hscvd import config
from hscvd.data import load_dataset, split_xy
from hscvd.evaluation import compute_metrics, format_table, tune_threshold
from hscvd.models import build_improved_model, build_ml_model
from hscvd.sampling import build_splits, resample
from run_experiments import set_seeds

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "heart_disease_health_indicators_BRFSS2015.csv")

ALL_FEATURES = "all"
# (model, configuration label, features, sampling, improved settings)
CONFIGURATIONS = [
    ("XGBoost", "9 features, HS (baseline)", config.SELECTED_FEATURES, "hs", False),
    ("XGBoost", "9 features, ROS only", config.SELECTED_FEATURES, "ros", False),
    ("XGBoost", "9 features, no resampling", config.SELECTED_FEATURES, "none", False),
    ("XGBoost", "21 features, no resampling", ALL_FEATURES, "none", False),
    ("XGBoost", "21 features, class weights, regularised (improved)", ALL_FEATURES,
     "none", True),
    ("DT", "9 features, HS, full depth (baseline)", config.SELECTED_FEATURES, "hs", False),
    ("DT", "21 features, class weights, depth 8 (improved)", ALL_FEATURES, "none", True),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", default=os.path.join(HERE, "results", "ablation"))
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(config.RANDOM_STATE)
    os.makedirs(args.out, exist_ok=True)
    splits = build_splits(load_dataset(args.data), "none", protocol="strict")

    rows = []
    predictions = pd.DataFrame({"y_true": splits["test"][config.TARGET]})
    for model_name, label, features, sampling, improved in CONFIGURATIONS:
        cols = (list(splits["train"].columns) if features == ALL_FEATURES
                else features + [config.TARGET])
        X_train, y_train = split_xy(splits["train"][cols])
        X_train, y_train = resample(X_train, y_train, sampling)
        X_val, y_val = split_xy(splits["val"][cols])
        X_test, y_test = split_xy(splits["test"][cols])

        model = (build_improved_model(model_name, y_train) if improved
                 else build_ml_model(model_name))
        model.fit(X_train, y_train, X_val, y_val)
        scores = model.predict_scores(X_test)
        at_default = compute_metrics(y_test, model.predict(X_test), scores)
        predictions[f"{model_name} | {label}"] = scores
        tune_threshold(model, X_val, y_val)
        at_tuned = compute_metrics(y_test, model.predict(X_test), scores)
        rows.append({"Model": model_name, "Configuration": label,
                     "AUC (%)": at_tuned["AUC (%)"],
                     "Acc @ 0.5 (%)": at_default["Accuracy (%)"],
                     "Acc @ tuned (%)": at_tuned["Accuracy (%)"],
                     "Tuned threshold": model.threshold})
        print(f"{model_name:8s} {label:52s} AUC={rows[-1]['AUC (%)']:.2f}  "
              f"acc@0.5={rows[-1]['Acc @ 0.5 (%)']:.2f}  "
              f"acc@tuned={rows[-1]['Acc @ tuned (%)']:.2f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "ablation.csv"), index=False)
    predictions.to_csv(os.path.join(args.out, "ablation_predictions.csv"),
                       index=False)
    print("\n" + format_table(table))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
