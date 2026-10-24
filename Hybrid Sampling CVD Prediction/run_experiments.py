"""Reproduce the experiments of "Integrating Hybrid Sampling with Machine
Learning for Enhanced Cardiovascular Disease Prediction".

Pipeline (Fig. 1): dataset -> attribute selection -> hybrid sampling ->
80/20 train/validation split (+ 3,000-per-class test set) -> train ML/DL
models -> evaluation.

Outputs (under --out):
    eda/        correlation ranking, Figs. 2-5
    splits/     resampled train / validation / test CSVs per sampling method
    tables/     Table 3 (validation), Table 4 (test), Table 5 (DL 5-fold CV),
                Table 7 (ML 5-fold CV)
    figures/    ROC curves (Figs. 6-9), confusion matrices (Figs. 10-12),
                learning curves (Figs. 13-16)

Example:
    python run_experiments.py                       # full run, paper protocol
    python run_experiments.py --protocol strict     # leakage-free evaluation
    python run_experiments.py --sample-frac 0.05 --epochs 2 --skip-cv  # smoke test
"""

import argparse
import os
import time

import pandas as pd

from hscvd import config, plots
from hscvd.data import (correlation_ranking, describe, load_dataset,
                        select_features, split_xy)
from hscvd.evaluation import cross_validate, evaluate, format_table
from hscvd.models import build_model
from hscvd.sampling import build_splits

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "heart_disease_health_indicators_BRFSS2015.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA, help="BRFSS 2015 heart disease CSV")
    p.add_argument("--out", default=os.path.join(HERE, "results"), help="output directory")
    p.add_argument("--protocol", choices=["paper", "strict"], default="paper",
                   help="paper: resample before splitting (as in the paper); "
                        "strict: resample the training set only")
    p.add_argument("--sampling", nargs="+", default=config.SAMPLING_METHODS,
                   choices=config.SAMPLING_METHODS)
    p.add_argument("--sample-frac", type=float, default=1.0,
                   help="use a random fraction of the raw data (for quick runs)")
    p.add_argument("--test-per-class", type=int, default=config.TEST_PER_CLASS,
                   help="rows per class held out as the test set")
    p.add_argument("--epochs", type=int, default=None,
                   help="override the number of training epochs of every DL model")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--no-autoencoder", action="store_true",
                   help="feed scaled features to the DL models directly")
    p.add_argument("--skip-cv", action="store_true", help="skip the 5-fold cross-validation")
    p.add_argument("--verbose", type=int, default=0, help="Keras verbosity")
    return p.parse_args()


def set_seeds(seed):
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        from tensorflow import keras
        keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def main():
    args = parse_args()
    set_seeds(config.RANDOM_STATE)
    out = lambda *parts: os.path.join(args.out, *parts)  # noqa: E731
    for sub in ("eda", "splits", "tables", "figures"):
        os.makedirs(out(sub), exist_ok=True)

    dl_kwargs = {"batch_size": args.batch_size, "verbose": args.verbose,
                 "use_autoencoder": not args.no_autoencoder}
    if args.epochs:
        dl_kwargs["epochs"] = args.epochs
        dl_kwargs["ae_epochs"] = args.epochs

    # ---- 3.1-3.2 Data collection and preprocessing
    print("== Loading data")
    df = load_dataset(args.data)
    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=config.RANDOM_STATE)
        df = df.reset_index(drop=True)
    describe(df)
    plots.plot_distributions(df, out("eda", "fig2_distribution_before_sampling.png"))
    plots.plot_age_distribution(df, out("eda", "fig3_age_before_sampling.png"))

    # ---- 3.3 Feature selection
    ranking = correlation_ranking(df)
    ranking.rename("correlation").to_csv(out("eda", "correlation_with_target.csv"))
    print("\n== Correlation with", config.TARGET)
    print(ranking.to_string(float_format=lambda v: f"{v:.4f}"))
    print("Selected features:", ", ".join(config.SELECTED_FEATURES))
    df = select_features(df)

    val_rows, test_rows, dl_cv, ml_cv = [], [], [], []

    for method in args.sampling:
        label = config.SAMPLING_LABELS[method]
        # ---- 3.4 Hybrid sampling and partitioning
        print(f"\n== Sampling: {label} ({args.protocol} protocol)")
        t0 = time.time()
        splits = build_splits(df, method, protocol=args.protocol,
                              n_per_class=args.test_per_class)
        print(f"   resampled size={len(splits['resampled'])}  train={len(splits['train'])}  "
              f"val={len(splits['val'])}  test={len(splits['test'])}  "
              f"({time.time() - t0:.0f}s)")
        for name in ("train", "val", "test"):
            splits[name].to_csv(out("splits", f"{method}_{name}.csv"), index=False)
        if method == "hs":
            plots.plot_distributions(splits["resampled"],
                                     out("eda", "fig4_distribution_after_sampling.png"))
            plots.plot_age_distribution(splits["resampled"],
                                        out("eda", "fig5_age_after_sampling.png"))

        # The paper evaluates the DL models under every sampling technique and
        # the ML models under hybrid sampling only.
        model_names = config.DL_MODELS + (config.ML_MODELS if method == "hs" else [])
        roc_dl, roc_ml = [], []
        for name in model_names:
            tag = f"{label}_{name}"
            print(f"-- Training {tag}")
            t0 = time.time()
            model = build_model(name, **dl_kwargs)
            model.fit(*split_xy(splits["train"]), *split_xy(splits["val"]))
            for split_name, rows in (("val", val_rows), ("test", test_rows)):
                metrics, y_true, y_score, cm = evaluate(model, splits[split_name])
                rows.append({"Sampling Technique": tag, "Model": name, **metrics})
                if split_name == "test":
                    target = roc_dl if name in config.DL_MODELS else roc_ml
                    target.append((tag if method == "hs" else name, y_true, y_score,
                                   metrics["AUC (%)"]))
                    if method == "hs":
                        plots.plot_confusion_matrix(
                            cm, out("figures", f"confusion_matrix_{tag}.png"),
                            title=f"{tag} Confusion Matrix (Test)")
            print("   test: " + ", ".join(f"{k.split(' ')[0]}={v:.2f}"
                                          for k, v in test_rows[-1].items()
                                          if k.endswith("(%)"))
                  + f"  ({time.time() - t0:.0f}s)")
            if model.history and method == "hs":
                plots.plot_learning_curves(model.history, tag,
                                           out("figures", f"learning_curve_{tag}.png"))

        plots.plot_roc(roc_dl, out("figures", f"roc_{method}_dl.png"),
                       title=f"ROC with {label} Approach (DL)")
        if roc_ml:
            plots.plot_roc(roc_ml, out("figures", f"roc_{method}_ml.png"),
                           title=f"ROC with {label} Approach (ML)")

        # ---- 5-fold cross-validation for the hybrid-sampling models
        if method == "hs" and not args.skip_cv:
            fold_method = method if args.protocol == "strict" else None
            for name in model_names:
                print(f"-- 5-fold CV: {label}_{name}")
                table = cross_validate(lambda y, n=name: build_model(n, **dl_kwargs),
                                       splits["pool"], resample_method=fold_method)
                table.insert(0, "Model", f"{label}_{name}")
                (dl_cv if name in config.DL_MODELS else ml_cv).append(table)

    # ---- Tables
    tables = {
        "table3_validation.csv": pd.DataFrame(val_rows),
        "table4_test.csv": pd.DataFrame(test_rows),
    }
    if dl_cv:
        tables["table5_dl_cross_validation.csv"] = pd.concat(dl_cv, ignore_index=True)
    if ml_cv:
        tables["table7_ml_cross_validation.csv"] = pd.concat(ml_cv, ignore_index=True)
    for fname, table in tables.items():
        table.to_csv(out("tables", fname), index=False)
        print(f"\n== {fname}\n{format_table(table)}")
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
