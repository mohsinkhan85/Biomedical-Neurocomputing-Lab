"""Compare the paper's setup with an improved setup on real, unseen patients.

Both setups share the same leakage-free split: the test set (3,000 real
records per class) and a 20% validation set are held out before anything
else, and only the training data is used for fitting.

    paper     9 correlation-selected features, hybrid sampling (SMOTEENN +
              ROS) of the training data, models as in the paper, 0.5 cut-off
    improved  all 21 features, no resampling (class weights instead),
              regularised / early-stopped models, decision cut-off tuned on
              the validation set (Youden's J)

Outputs (under --out): comparison.csv, roc_improved.png and a confusion
matrix per improved model.

Example:
    python run_improved.py
    python run_improved.py --models XGBoost LogReg   # ML models only, fast
"""

import argparse
import os
import time

import pandas as pd

from hscvd import config, plots
from hscvd.data import load_dataset, split_xy
from hscvd.evaluation import evaluate, format_table, tune_threshold
from hscvd.models import build_improved_model, build_model
from hscvd.sampling import build_splits, resample_df
from run_experiments import set_seeds

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "heart_disease_health_indicators_BRFSS2015.csv")
MODELS = ["DT", "SVM", "XGBoost", "LogReg", "CNN", "NN", "MLP"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", default=os.path.join(HERE, "results", "improved"))
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the paper setup and run only the improved one")
    p.add_argument("--epochs", type=int, default=30,
                   help="maximum epochs for the DL models (early stopping applies)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--verbose", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(config.RANDOM_STATE)
    os.makedirs(args.out, exist_ok=True)

    df = load_dataset(args.data)
    splits = build_splits(df, "none", protocol="strict")
    print(f"train={len(splits['train'])}  val={len(splits['val'])}  "
          f"test={len(splits['test'])} (real records only)")

    paper_cols = config.SELECTED_FEATURES + [config.TARGET]
    paper_train = None
    rows, curves = [], []
    for name in args.models:
        if not args.no_baseline and name != "LogReg":
            if paper_train is None:
                print("Resampling the paper training set (SMOTEENN + ROS)...")
                paper_train = resample_df(splits["train"][paper_cols], "hs")
            t0 = time.time()
            model = build_model(name, verbose=args.verbose)
            model.fit(*split_xy(paper_train),
                      *split_xy(splits["val"][paper_cols]))
            metrics, *_ = evaluate(model, splits["test"][paper_cols])
            rows.append({"Setup": "paper", "Model": name, "Threshold": 0.5, **metrics})
            _report(rows[-1], t0)

        t0 = time.time()
        X_train, y_train = split_xy(splits["train"])
        model = build_improved_model(name, y_train, epochs=args.epochs,
                                     batch_size=args.batch_size,
                                     verbose=args.verbose)
        model.fit(X_train, y_train, *split_xy(splits["val"]))
        threshold = tune_threshold(model, splits["val"])
        metrics, y_true, y_score, cm = evaluate(model, splits["test"])
        rows.append({"Setup": "improved", "Model": name,
                     "Threshold": threshold, **metrics})
        curves.append((name, y_true, y_score, metrics["AUC (%)"]))
        plots.plot_confusion_matrix(
            cm, os.path.join(args.out, f"confusion_matrix_{name}.png"),
            title=f"Improved {name} Confusion Matrix (Test)")
        _report(rows[-1], t0)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "comparison.csv"), index=False)
    plots.plot_roc(curves, os.path.join(args.out, "roc_improved.png"),
                   title="ROC of Improved Models (Real Test Set)")
    print("\n" + format_table(table))
    print(f"\nResults written to {args.out}")


def _report(row, t0):
    print(f"{row['Setup']:>8} {row['Model']:<8} "
          + ", ".join(f"{k.split(' ')[0]}={v:.2f}" for k, v in row.items()
                      if k.endswith("(%)"))
          + f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
